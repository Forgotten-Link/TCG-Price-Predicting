import os 
import requests
import pandas as pd



dragonemblem_key = "tcg_a34548e6051f4ebb85683a062eeed032"
bhuynh616_key = "tcg_5d7cc9d8d2704c5db1a0d4b750bb7631"
alex_key = "tcg_b68b0d0de26544019b12fca272991c98"

JUSTTCG_API_KEY = alex_key
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
    
def find_top_cards(set_id: str, limit: int = 10, offset:int = 0):
    print(f"\nSearching for the top {limit} most valuable cards...")
    url = f"{BASE_URL}/cards"

    while True: 
        params = {
            "set": set_id, 
            "limit": limit,
            "offset": offset,
        }


        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()

        cards = response.json().get("data", [])
        if not cards:
            break
        print(f"\n---{limit} Recently Scanned Cards in {set_id}---")
        for card in cards:
            card_name = card["name"]
            rarity = card.get("rarity")
            for variant in card["variants"]:
                market_price = variant.get("price")
                printing = variant.get("printing")
                condition = variant.get("condition")   
            print(f"{card_name} ({printing}) - {rarity} - {condition} - ${market_price}")
            ygo_set_data.append({
                "set": set_id, 
                "name": card_name, 
                "variant": printing, 
                "condition": condition, 
                "rarity": rarity, 
                "price": market_price
                })
        offset += limit

    df = pd.DataFrame(ygo_set_data)
    df.to_csv(f"{set_id}_data.csv", index=False)


def main():
    print("Starting Scanner")
    target_set_name = "Magnificent Mavens"
    target_game = "YuGiOh"
    set_id = get_set_id(target_set_name, target_game)
    
    if not set_id:
        print("Could not find set ID. Exiting")
        return

    find_top_cards(set_id)

if __name__ == "__main__":
    main()