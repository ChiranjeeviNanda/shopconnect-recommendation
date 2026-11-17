import os
import logging
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, LOG_DIR
from household_generator import generate_households, generate_food_dislikes
from simulation.simulation_engine import run_simulation

# Configure logging for data generation monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "synthetic_data.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Orchestrates the end-to-end synthetic data generation pipeline."""
    logger.info("=== Starting Synthetic Data Generation (90 Days, ~300 Households) ===")
    
    raw_path = os.path.join(RAW_DIR, "recipe_list.csv")
    if not os.path.exists(raw_path):
        logger.error(f"Recipe catalog not found at {raw_path}.")
        raise FileNotFoundError(f"Recipe catalog not found at {raw_path}. Please place recipe_list.csv in data/raw/")
    
    recipes_df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(recipes_df)} recipes successfully from catalog.")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info(f"Ensured processed directory exists at: {PROCESSED_DIR}")
    
    # Generate households and their respective ingredient/dish dislikes
    logger.info("Generating household profiles...")
    households_df = generate_households(num_households=300)
    logger.info(f"Generated {len(households_df)} household profiles.")

    logger.info("Generating household food dislikes...")
    dislikes_df = generate_food_dislikes(households_df)
    logger.info(f"Generated {len(dislikes_df)} food dislike entries.")
    
    # Run the simulation engine to generate meal plans, items, user interactions, and orders
    logger.info("Starting simulation engine (90-day horizon)...")
    plans_df, items_df, interactions_df, orders_df = run_simulation(recipes_df, households_df, dislikes_df)
    logger.info("Simulation engine execution completed successfully.")
    
    # Export all generated datasets to CSV files in the processed data directory
    logger.info("Exporting generated datasets to CSV...")
    
    datasets = [
        (households_df, "households.csv"),
        (dislikes_df, "food_dislikes.csv"),
        (plans_df, "meal_plans.csv"),
        (items_df, "meal_plan_items.csv"),
        (interactions_df, "interactions.csv"),
        (orders_df, "orders.csv")
    ]
    
    for df, filename in datasets:
        file_path = os.path.join(PROCESSED_DIR, filename)
        df.to_csv(file_path, index=False)
        logger.info(f" - Exported {filename} ({len(df)} rows) to {file_path}")
    
    logger.info("=== Synthetic Data Generation Complete ===")
    logger.info(f"Generated:")
    logger.info(f" - Households: {len(households_df)}")
    logger.info(f" - Food Dislikes: {len(dislikes_df)}")
    logger.info(f" - Meal Plans: {len(plans_df)}")
    logger.info(f" - Meal Plan Items: {len(items_df)}")
    logger.info(f" - Interactions: {len(interactions_df)}")
    logger.info(f" - Orders: {len(orders_df)}")
    logger.info(f"All files exported successfully to {PROCESSED_DIR}/")

if __name__ == "__main__":
    main()