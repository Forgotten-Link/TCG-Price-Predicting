# Train.py — tuned LogReg + Linear SVM (no RF), NaN-safe, plots, saves artifacts
import pandas as pd, numpy as np, inspect
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay, roc_auc_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import warnings, joblib
from sklearn.exceptions import ConvergenceWarning

# --- quiet liblinear convergence spam (cosmetic) ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---- load full joined table ----
df = pd.read_csv("joined_full_with_prices.csv")

# (optional) strict matches only
# df = df[df["match_level"] == "name+set"]

# keep only rows that have a price
df["price_market"] = pd.to_numeric(df["price_market"], errors="coerce")
df = df.dropna(subset=["price_market"]).copy()

# ---- label: top p% by rank (robust) ----
P_TOP = 0.20
k = max(1, int(np.ceil(len(df) * P_TOP)))
thr = df["price_market"].nlargest(k).min()
df["price_tier"] = (df["price_market"] >= thr).astype(int)
print(f"Label: top {int(P_TOP*100)}% => threshold ${thr:.2f} | positives={df['price_tier'].sum()} / {len(df)}")

# ---- feature sets (use what exists) ----
TEXT = "desc" if "desc" in df.columns else None

cat_candidates = [
    "frameType","attribute","race","archetype","type",
    "rarity","variant","condition","set_name","match_level"
]
CAT = [c for c in cat_candidates if c in df.columns]

num_candidates = ["atk","def","level","linkval","views","upvotes","favorites","comments"]
NUM = [c for c in num_candidates if c in df.columns]

# remove leakage: DO NOT include price columns as features
X_cols = ([TEXT] if TEXT else []) + CAT + NUM
print("Using features:")
print("  text:", TEXT)
print("  cat :", CAT)
print("  num :", NUM)

X = df[X_cols].copy()
y = df["price_tier"].astype(int)

# general safety: convert inf -> NaN; TF-IDF needs strings
X.replace([np.inf, -np.inf], np.nan, inplace=True)
if TEXT:
    X[TEXT] = X[TEXT].astype(str).fillna("")

# group by card name to avoid reprint leakage
groups = df["name"] if "name" in df.columns else pd.Series(np.arange(len(df)))

# ---- preprocessing (with imputers + cross-version OHE) ----
from sklearn.preprocessing import OneHotEncoder
ohe_kwargs = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    ohe_kwargs["sparse_output"] = True   # sklearn >= 1.2
else:
    ohe_kwargs["sparse"] = True          # sklearn < 1.2
cat_encoder = OneHotEncoder(**ohe_kwargs)

steps = []
if TEXT:
    steps.append(("text", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=30000), TEXT))
if CAT:
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", cat_encoder),
    ])
    steps.append(("cat", cat_pipe, CAT))
if NUM:
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("ss",  StandardScaler(with_mean=False)),
    ])
    steps.append(("num", num_pipe, NUM))

pre = ColumnTransformer(steps, remainder="drop", sparse_threshold=0.3)

# ---- split ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups=groups))
Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

# ---- 1) Logistic Regression (tuned) ----
pipe_lr = Pipeline([("pre", pre),
                    ("clf", LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear"
                    ))])

lr = GridSearchCV(
    pipe_lr,
    {"clf__C": [0.1, 0.3, 1, 3, 10]},
    scoring="f1",
    cv=3,
    n_jobs=-1
)
lr.fit(Xtr, ytr)
yp_lr = lr.predict(Xte)
print("\nLogistic Regression:", lr.best_params_)
print(classification_report(yte, yp_lr, target_names=["low","high"], digits=3))

# ---- 2) Linear SVM (tuned; max_iter + tol grid) ----
pipe_svm = Pipeline([("pre", pre),
                     ("clf", LinearSVC(class_weight="balanced"))])

svm = GridSearchCV(
    pipe_svm,
    {
      "clf__C": [0.1, 0.3, 1, 3, 10],
      "clf__max_iter": [5000, 10000, 20000],
      "clf__tol": [1e-3, 1e-4]
    },
    scoring="f1",
    cv=3,
    n_jobs=-1
)
svm.fit(Xtr, ytr)
yp_svm = svm.predict(Xte)
print("\nLinear SVM:", svm.best_params_)
print(classification_report(yte, yp_svm, target_names=["low","high"], digits=3))

# ---- Confusion matrix (SVM) — save + show ----
disp = ConfusionMatrixDisplay.from_predictions(yte, yp_svm, display_labels=["low","high"], cmap="Blues")
plt.title("Linear SVM – Confusion Matrix")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# ---- ROC-AUC (LogReg) + threshold tuning readout ----
yp_lr_proba = lr.best_estimator_.predict_proba(Xte)[:, 1]
print("\nLogReg ROC-AUC:", roc_auc_score(yte, yp_lr_proba))

prec, rec, thr = precision_recall_curve(yte, yp_lr_proba)
f1_vals = 2 * (prec * rec) / (prec + rec + 1e-12)
best_i = np.nanargmax(f1_vals)
best_thr = 0.0 if best_i >= len(thr) else thr[best_i]
best_f1 = np.nanmax(f1_vals)
print(f"\nLogReg best F1 on test ~{best_f1:.3f} at threshold ~{best_thr:.2f}")
pred_best = (yp_lr_proba >= best_thr).astype(int)
print(classification_report(yte, pred_best, target_names=['low','high'], digits=3))

# ---- save models ----
joblib.dump(lr.best_estimator_, "model_logreg.pkl")
joblib.dump(svm.best_estimator_, "model_svm.pkl")
print("\nSaved: svm_confusion_matrix.png, model_logreg.pkl, model_svm.pkl")
