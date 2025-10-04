import os 
import requests
import pandas as pd



dragonemblem_key = "tcg_a34548e6051f4ebb85683a062eeed032"
bhuynh616_key = "tcg_5d7cc9d8d2704c5db1a0d4b750bb7631"

JUSTTCG_API_KEY = bhuynh616_key
BASE_URL = "https://api.justtcg.com/v1"
HEADERS = {"x-api-key": JUSTTCG_API_KEY}

ygo_set_data = []
def get_set_id(set_name:str, game_name:str):
    print(f"Querying for all {game_name} sets...")
    url = f"{BASE_URL}/sets"
    params = {"game": game_name}

    try: 
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()

        sets = response.json()["data"]

        print(f"Found {len(sets)} sets. Searching for {set_name}...")

        #for card_set in sets: #for getting a list of set names and ids
            #print(f"Found set '{card_set['name']}' with ID: {card_set['id']}")

        for s in sets:
            if s["name"].lower() == set_name.lower():
                print(f"Found set '{s['name']}' with ID: {s['id']}")
                return s['id']
        
        print("Set not found.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occured: {e}")
        return None
    
def find_top_cards(set_id: str, limit: int = 10):
    print(f"\nSearching for the top {limit} most valuable cards...")
    url = f"{BASE_URL}/cards"

    params = {
        "set": set_id, 
        "orderBy": "price",
        "order": "desc", 
        "limit": limit,
    }

    try: 
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()

        cards = response.json()["data"]

        print(f"\n--- Top {limit} Most Valuable Cards in {set_id}---")
        for i, card in enumerate(cards, 1):
            card_name = card["name"]
            for variant in card["variants"]:
                market_price = variant.get("price")
                printing = variant.get("printing")
                condition = variant.get("condition")
                rarity = card.get("rarity")
            print(
                f"{i}, {card_name} ({printing}) - {rarity} - {condition} - ${market_price}"
            )
            ygo_set_data.append({
                "name": card_name, 
                "variant": printing, 
                "condition": condition, 
                "rarity": rarity, 
                "price": market_price
                })
        df = pd.DataFrame(ygo_set_data)
        df.to_csv(f"{set_id}_data.csv", index=False)

    except requests.exceptions.RequestExceptions as e:
        print(f"An error occured: {e}")


def main():
    print("Starting Scanner")
    target_set_name = "2025 Mega-Pack"
    target_game = "YuGiOh"
    set_id = get_set_id(target_set_name, target_game)
    
    if not set_id:
        print("Could not find set ID. Exiting")
        return

    find_top_cards(set_id)

if __name__ == "__main__":
    main()