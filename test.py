# market_scanner.py  
import os  
import requests  
  
# Your JustTCG API Key  
# Best practice is to store this as an environment variable  
API_KEY = "tcg_a34548e6051f4ebb85683a062eeed032"
BASE_URL = "https://api.justtcg.com/v1"  
HEADERS = {"x-api-key": API_KEY}  
  
  
def get_set_id(set_name: str, game_name: str) -> str | None:  
    """Fetches the set ID for a given set name."""  
    print(f"Querying for all {game_name} sets...")  
    url = f"{BASE_URL}/sets"  
  
    params = {"game": game_name}  
  
    try:  
        response = requests.get(url, headers=HEADERS, params=params)  
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)  
  
        sets = response.json()["data"]  
  
        print(f"Found {len(sets)} sets. Searching for {set_name}...")  
  
        for s in sets:  
            if s["name"].lower() == set_name.lower():  
                print(f"Found set '{s['name']}' with ID: {s['id']}")  
                return s["id"]  
  
        print("Set not found.")  
        return None  
    except requests.exceptions.RequestException as e:  
        print(f"An error occurred: {e}")  
        return None  
  
  
def find_top_cards(set_id: str, limit: int = 10):  
    """Finds the most valuable cards in a given set."""  
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
            highest_priced_variant = max(  
                (variant for variant in card["variants"]), key=lambda x: x["price"]  
            )  
            market_price = highest_priced_variant["price"]  
            variant_printing = highest_priced_variant["printing"]  
            variant_condition = highest_priced_variant["condition"]  
            print(  
                f"{i}. {card_name} ({variant_printing}) - {variant_condition} - ${market_price}"  
            )  
    except requests.exceptions.RequestException as e:  
        print(f"An error occurred: {e}")  
  
  
def main():  
    """Main function to run our market scanner."""  
    if not API_KEY:  
        print("Error: JUSTTCG_API_KEY environment variable not set.")  
        return  
    print("Starting the JustTCG Market Scanner...")  
    target_set_name = "SV04: Paradox Rift"  
    target_game = "Pokemon"  
    set_id = get_set_id(target_set_name, target_game)  
  
    if not set_id:  
        print("Could not find set ID. Exiting.")  
        return  
  
    find_top_cards(set_id)  
  
  
if __name__ == "__main__":  
    main()
