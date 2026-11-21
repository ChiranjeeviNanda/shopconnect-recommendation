import random
import numpy as np
import pandas as pd
from datetime import datetime

def generate_households(num_households=300):
    """Generate household profiles with region, diet, spice level, and health goal attributes."""
    regions = ['south_india', 'north_india', 'west_india', 'east_india']
    region_weights = [0.55, 0.20, 0.20, 0.05]

    diets = ['vegetarian', 'non-vegetarian', 'vegan', 'jain']
    diet_weights = [0.60, 0.25, 0.10, 0.05]

    spice_levels = ['mild', 'medium', 'spicy']
    health_goals = ['everyday', 'high-protein', 'weight-loss']

    households = []
    # Generate demographic records for each household with randomized attributes
    for i in range(1, num_households + 1):
        h_id = f"H{i:03d}"
        households.append({
            'household_id': h_id,
            'household_size': random.choice([1, 2, 3, 4, 5]),
            'home_region': np.random.choice(regions, p=region_weights),
            'diet': np.random.choice(diets, p=diet_weights),
            'spice_level': random.choice(spice_levels),
            'health_goal': np.random.choice(health_goals),
            'created_at': datetime(2026, 1, 1).strftime('%Y-%m-%d %H:%M:%S')
        })

    return pd.DataFrame(households)


def generate_food_dislikes(households_df):
    """Generate disliked ingredients or dishes for households, returning a consistent schema even when empty."""
    common_dislikes = ['mushroom', 'brinjal', 'capsicum', 'paneer', 'seafood', 'gourd', 'jackfruit']
    food_dislikes = []
    dislike_id_counter = 1

    for _, h in households_df.iterrows():
        if random.random() < 0.4:
            num_dislikes = random.choice([1, 2])
            sampled_dislikes = random.sample(common_dislikes, num_dislikes)
            for d in sampled_dislikes:
                food_dislikes.append({
                    'dislike_id': dislike_id_counter,
                    'household_id': h['household_id'],
                    'ingredient_or_dish': d
                })
                dislike_id_counter += 1
                
    if not food_dislikes:
        # Preserve expected columns so downstream groupby operations do not fail on empty data
        return pd.DataFrame(columns=['dislike_id', 'household_id', 'ingredient_or_dish'])

    return pd.DataFrame(food_dislikes)