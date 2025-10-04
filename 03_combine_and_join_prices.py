from pathlib import Path
import pandas as pd, numpy as np, re, glob, difflib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ------------- CONFIG -------------
# drop these price files entirely (not in your dataset)
DROP_SLUGS = {
    "2025-mega-pack-yugioh",
    "structure-deck-blue-eyes-white-destiny-yugioh",
    "supreme-darkness-yugioh",
}

# map remaining price slugs -> dataset set names
SET_SLUG_MAP = {
    "magnificent-mavens-yugioh": "Magnificent Mavens",
    "flames-of-destruction-yugioh": "Flames of Destruction",
    "shadows-in-valhalla-yugioh": "Shadows in Valhalla",
    "duel-power-yugioh": "Duel Power",
    "duel-devastator-yugioh": "Duel Devastator",
    "25th-anniversary-rarity-collection-yugioh": "25th Anniversary Rarity Collection",
    # add more mappings here if you add files later
}

TOP_P = 0.20  # price_tier = top 20% by rank
# ------------- /CONFIG -------------

def snake(s): return re.sub(r"[^\w]+","_",str(s).strip()).lower()

# 1) load price csvs you uploaded (pattern: *_data.csv)
price_files = sorted(glob.glob(str(ROOT / "*_data.csv")))
if not price_files:
    raise SystemExit("No *_data.csv price files found next to this script.")

frames = []
for p in price_files:
    df = pd.read_csv(p)
    # normalize typical columns
    cols = {snake(c): c for c in df.columns}
    # common names in your dumps: set, name, variant, condition, rarity, price
    def pick(*cands):
        for c in cands:
            if c in cols: return cols[c]
        return None
    c_set  = pick("set")
    c_name = pick("name")
    c_var  = pick("variant")
    c_cond = pick("condition")
    c_rar  = pick("rarity")
    c_price= pick("price")
    if not (c_set and c_name and c_price):
        print(f"[WARN] Skipping {p} (missing core columns).")
        continue

    tmp = df[[c_set, c_name] + [x for x in [c_var,c_cond,c_rar,c_price] if x]].copy()
    tmp.columns = ["set_slug","name","variant","condition","rarity","price_market"][:tmp.shape[1]]
    # drop slugs we don't want
    if any(s in str(p) for s in DROP_SLUGS) or tmp["set_slug"].isin(DROP_SLUGS).any():
        # filter rows whose slug is in DROP_SLUGS
        tmp = tmp[~tmp["set_slug"].isin(DROP_SLUGS)]
    frames.append(tmp)

if not frames:
    raise SystemExit("No price rows after filtering DROP_SLUGS.")

prices_all = pd.concat(frames, ignore_index=True)

# ensure numeric price
prices_all["price_market"] = pd.to_numeric(prices_all["price_market"], errors="coerce")
prices_all = prices_all.dropna(subset=["price_market"])

# map slug -> dataset set_name
prices_all["set_name_map"] = prices_all["set_slug"].map(SET_SLUG_MAP)
# if any are still unmapped, try fuzzy against dataset set names later
print(f"Loaded {len(prices_all)} price rows from {len(price_files)} files (after filtering).")

# 2) collapse to "best" per (name, mapped set)
best = (prices_all
        .dropna(subset=["set_name_map"])
        .sort_values("price_market", ascending=False)
        .drop_duplicates(["name","set_name_map"])
        .rename(columns={"set_name_map":"set_name"}))

best = best[["set_name","name","price_market","rarity","variant","condition"]]

# 3) load dataset catalog
cards  = pd.read_csv(DATA / "cards.csv")
sets   = pd.read_csv(DATA / "cardsets.csv")
links  = pd.read_csv(DATA / "cards_cardsets.csv")

# attach set_name to each card id
links2 = (links.merge(sets[["id","set_name"]], left_on="set_id", right_on="id", how="left")
               .rename(columns={"id_x":"link_id"})
               .merge(cards, left_on="card_id", right_on="id", how="left"))

# we will preserve ALL columns from cards.csv (+ set_name)
all_card_cols = [c for c in cards.columns]
keep_cols = all_card_cols + ["set_name"]

# normalize for join
def norm(s): return re.sub(r"[^\w\s]","",str(s)).strip().lower()
links2["name_norm"] = links2["name"].map(norm)
links2["set_norm"]  = links2["set_name"].map(norm)

best["name_norm"] = best["name"].map(norm)
best["set_norm"]  = best["set_name"].map(norm)

# 4) primary join on (name,set)
joined = links2.merge(
    best[["name_norm","set_norm","price_market","rarity","variant","condition"]],
    on=["name_norm","set_norm"], how="left"
)

# try to fuzz-map any unmapped slugs (if set_name_map was NaN) – optional safety
if prices_all["set_name_map"].isna().any():
    # build fuzzy slug->dataset set_name map one time
    cat_sets = sorted(sets["set_name"].dropna().unique().tolist())
    slugs_unmapped = sorted(prices_all.loc[prices_all["set_name_map"].isna(),"set_slug"].unique().tolist())
    fuzz_map = {}
    for s in slugs_unmapped:
        m = difflib.get_close_matches(s.replace("-yugioh","").replace("-"," ").title(), cat_sets, n=1, cutoff=0.6)
        if m: fuzz_map[s] = m[0]
    if fuzz_map:
        prices_all["set_name_map2"] = prices_all["set_slug"].map(fuzz_map)
        best2 = (prices_all
                .dropna(subset=["set_name_map2"])
                .sort_values("price_market", ascending=False)
                .drop_duplicates(["name","set_name_map2"])
                .rename(columns={"set_name_map2":"set_name"}))
        best2 = best2[["set_name","name","price_market","rarity","variant","condition"]]
        best2["name_norm"] = best2["name"].map(norm)
        best2["set_norm"]  = best2["set_name"].map(norm)
        # fill only where primary was missing
        missed = joined["price_market"].isna()
        joined.loc[missed, ["price_market","rarity","variant","condition"]] = \
            joined.loc[missed, ["name_norm","set_norm"]].merge(
                best2[["name_norm","set_norm","price_market","rarity","variant","condition"]],
                on=["name_norm","set_norm"], how="left"
            )[["price_market","rarity","variant","condition"]].values

# 5) fallback: name-only join for any still-missing prices
missed = joined["price_market"].isna()
if missed.any():
    name_only = (best.sort_values("price_market", ascending=False)
                      .drop_duplicates("name")[["name_norm","price_market","rarity","variant","condition"]])
    joined.loc[missed, "price_market"] = joined.loc[missed, "name_norm"].map(
        dict(zip(name_only["name_norm"], name_only["price_market"]))
    )
    # optional: bring over the other fields for the fallback too
    for col in ["rarity","variant","condition"]:
        joined.loc[missed, col] = joined.loc[missed, "name_norm"].map(
            dict(zip(name_only["name_norm"], name_only[col]))
        )

# mark match level
joined["match_level"] = np.where(
    joined["price_market"].notna()
    & joined["set_norm"].notna()
    & joined[["name_norm","set_norm"]].merge(
        best[["name_norm","set_norm"]].drop_duplicates(),
        on=["name_norm","set_norm"], how="left"
      )["set_norm"].notna(),
    "name+set",
    np.where(joined["price_market"].notna(), "name-only", "no-match")
)

# keep full card columns + set_name + price fields + match flag (SAFE)
wanted = all_card_cols + ["set_name","price_market","rarity","variant","condition","match_level"]
available = [c for c in wanted if c in joined.columns]
missing   = sorted(set(wanted) - set(available))
if missing:
    print("Note: these columns were not found and were skipped:", missing)

joined_full = joined[available].copy()

# 6) make training label (top p% by rank among matched)
trainable = joined_full.dropna(subset=["price_market"]).copy()
trainable["price_market"] = pd.to_numeric(trainable["price_market"], errors="coerce")
trainable = trainable.dropna(subset=["price_market"])

k = max(1, int(np.ceil(len(trainable) * TOP_P)))
thr = trainable["price_market"].nlargest(k).min()
trainable["price_tier"] = (trainable["price_market"] >= thr).astype(int)

# 7) save artifacts
prices_all.to_csv("prices_combined_filtered.csv", index=False)
best.to_csv("prices_by_card_best.csv", index=False)
joined_full.to_csv("joined_full_with_prices.csv", index=False)

slim_cols = ["name","set_name","desc","frameType","attribute","race","archetype",
             "atk","def","level","linkval","price_market","price_tier"]
slim_cols = [c for c in slim_cols if c in trainable.columns]
trainable[slim_cols].to_csv("joined_with_label.csv", index=False)

print(f"prices_combined_filtered.csv  -> {len(prices_all)} rows")
print(f"prices_by_card_best.csv       -> {len(best)} rows")
print(f"joined_full_with_prices.csv   -> {len(joined_full)} rows "
      f"(matched: {(joined_full['price_market'].notna()).sum()})")
print(f"joined_with_label.csv         -> {len(trainable)} rows; "
      f"positives={trainable['price_tier'].sum()} (thr={thr:.2f})")
