import os
import pandas as pd
from config import RAW_PATH, PROCESSED_DIR
from household_generator import generate_households, generate_food_dislikes
from simulation_engine import run_simulation

def main():
    """Orchestrates the end-to-end synthetic data generation pipeline."""
    print("=== Starting Synthetic Data Generation (90 Days, ~300 Households) ===")
    
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Recipe catalog not found at {RAW_PATH}. Please place recipe_list.csv in data/raw/")
    
    recipes_df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(recipes_df)} recipes from catalog.")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    households_df = generate_households(num_households=300)
    dislikes_df = generate_food_dislikes(households_df)
    
    plans_df, items_df, interactions_df, orders_df = run_simulation(recipes_df, households_df, dislikes_df)
    
    households_df.to_csv(os.path.join(PROCESSED_DIR, "households.csv"), index=False)
    dislikes_df.to_csv(os.path.join(PROCESSED_DIR, "food_dislikes.csv"), index=False)
    plans_df.to_csv(os.path.join(PROCESSED_DIR, "meal_plans.csv"), index=False)
    items_df.to_csv(os.path.join(PROCESSED_DIR, "meal_plan_items.csv"), index=False)
    interactions_df.to_csv(os.path.join(PROCESSED_DIR, "interactions.csv"), index=False)
    orders_df.to_csv(os.path.join(PROCESSED_DIR, "orders.csv"), index=False)
    
    print("=== Synthetic Data Generation Complete ===")
    print(f"Generated:")
    print(f" - Households: {len(households_df)}")
    print(f" - Food Dislikes: {len(dislikes_df)}")
    print(f" - Meal Plans: {len(plans_df)}")
    print(f" - Meal Plan Items: {len(items_df)}")
    print(f" - Interactions: {len(interactions_df)}")
    print(f" - Orders: {len(orders_df)}")
    print(f"All files exported successfully to {PROCESSED_DIR}/")

if __name__ == "__main__":
    main()