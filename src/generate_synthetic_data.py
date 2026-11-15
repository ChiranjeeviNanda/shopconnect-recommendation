import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_synthetic_data():
    print("=== Starting Synthetic Data Generation (90 Days, ~300 Households) ===")
    
    # 1. Load Recipe Catalog
    raw_path = os.path.join("data", "raw", "recipe_list.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Recipe catalog not found at {raw_path}. Please place recipe_list.csv in data/raw/")
    
    recipes_df = pd.read_csv(raw_path)
    print(f"Loaded {len(recipes_df)} recipes from catalog.")
    
    # Ensure processed directory exists
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    
    # 2. Generate Households (~300 households)
    num_households = 300
    regions = ['south_india', 'north_india', 'west_india', 'east_india']
    region_weights = [0.55, 0.20, 0.20, 0.05] # Reflecting recipe distribution skew
    
    diets = ['vegetarian', 'non-vegetarian', 'vegan', 'jain']
    diet_weights = [0.60, 0.25, 0.10, 0.05]
    
    spice_levels = ['mild', 'medium', 'spicy']
    health_goals = ['everyday', 'high-protein', 'weight-loss']
    
    households = []
    for i in range(1, num_households + 1):
        h_id = f"H{i:03d}"
        households.append({
            'household_id': h_id,
            'household_size': random.choice([1, 2, 3, 4, 5]),
            'home_region': np.random.choice(regions, p=region_weights),
            'diet': np.random.choice(diets, p=diet_weights),
            'spice_level': random.choice(spice_levels),
            'health_goal': np.random.choice(health_goals),
            'created_at': datetime(2025, 1, 1).strftime('%Y-%m-%d %H:%M:%S')
        })
    households_df = pd.DataFrame(households)
    
    # 3. Generate Food Dislikes
    common_dislikes = ['mushroom', 'brinjal', 'capsicum', 'paneer', 'seafood', 'gourd', 'jackfruit']
    food_dislikes = []
    dislike_id_counter = 1
    
    for _, h in households_df.iterrows():
        # 30% chance of having 1-2 dislikes
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
    dislikes_df = pd.DataFrame(food_dislikes)
    
    # Identify cold-start households (15-20% = ~45 to 60 households)
    cold_start_count = int(num_households * 0.18)
    cold_start_households = set(random.sample(list(households_df['household_id']), cold_start_count))
    
    # 4. Simulate 90 Days of Meal Plans & Interactions
    start_date = datetime(2025, 10, 1)
    days_to_simulate = 90
    slots = ['breakfast', 'lunch', 'dinner']
    
    meal_plans = []
    meal_plan_items = []
    interactions = []
    orders = []
    
    plan_id_counter = 1
    item_id_counter = 1
    interaction_id_counter = 1
    order_id_counter = 1
    
    # Pre-classify recipes for efficient filtering
    recipes_df['diet_clean'] = recipes_df['diet'].fillna('vegetarian').str.lower()
    recipes_df['region_clean'] = recipes_df['region'].fillna('south_india').str.lower()
    recipes_df['meal_role_clean'] = recipes_df['meal_role'].fillna('main').str.lower()
    
    for _, h in households_df.iterrows():
        h_id = h['household_id']
        h_diet = h['diet']
        h_region = h['home_region']
        h_goal = h['health_goal']
        is_cold_start = h_id in cold_start_households
        
        # Get household dislikes
        h_dislikes = set(dislikes_df[dislikes_df['household_id'] == h_id]['ingredient_or_dish'].str.lower())
        
        # Build weekly plans (every 7 days)
        for week_start_day in range(0, days_to_simulate, 7):
            plan_start = start_date + timedelta(days=week_start_day)
            plan_end = plan_start + timedelta(days=6)
            plan_id = f"P{plan_id_counter:05d}"
            plan_id_counter += 1
            
            meal_plans.append({
                'plan_id': plan_id,
                'household_id': h_id,
                'template_goal': h_goal,
                'start_date': plan_start.strftime('%Y-%m-%d'),
                'end_date': plan_end.strftime('%Y-%m-%d'),
                'version': 1,
                'created_at': (plan_start - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Track recent recipes for fatigue calculation within the week
            recent_recipes_in_week = []
            plan_has_interaction = False
            interaction_count_for_plan = 0
            
            # Generate days for this weekly plan
            for day_offset in range(7):
                current_date = plan_start + timedelta(days=day_offset)
                is_weekday = current_date.weekday() < 5
                
                for slot in slots:
                    # 1. Diet constraint
                    if h_diet == 'vegetarian':
                        role_candidates = recipes_df[recipes_df['diet_clean'].isin(['vegetarian', 'vegan', 'jain'])]
                    elif h_diet == 'vegan':
                        role_candidates = recipes_df[recipes_df['diet_clean'] == 'vegan']
                    elif h_diet == 'jain':
                        role_candidates = recipes_df[recipes_df['diet_clean'] == 'jain']
                    else:
                        role_candidates = recipes_df.copy()
                    
                    if role_candidates.empty:
                        role_candidates = recipes_df.copy()

                    # 2. Slot / Meal Role fitting
                    if slot == 'breakfast':
                        slot_filtered = role_candidates[role_candidates['meal_role_clean'].isin(['anchor', 'main', 'snack'])]
                        if is_weekday:
                            quick_filtered = slot_filtered[slot_filtered['cook_time_min'] <= 20]
                            if not quick_filtered.empty:
                                slot_filtered = quick_filtered
                        if not slot_filtered.empty:
                            role_candidates = slot_filtered
                    elif slot == 'lunch':
                        slot_filtered = role_candidates[role_candidates['meal_role_clean'].isin(['main', 'lentil', 'side', 'anchor'])]
                        if not slot_filtered.empty:
                            role_candidates = slot_filtered
                    else: # dinner
                        slot_filtered = role_candidates[role_candidates['meal_role_clean'].isin(['main', 'lentil', 'condiment', 'dessert', 'anchor'])]
                        if not slot_filtered.empty:
                            role_candidates = slot_filtered
                            
                    # 3. Dislike filtering
                    if h_dislikes:
                        def passes_dislikes(row):
                            text_blob = f"{row['name']} {row['ingredients_json']} {row['dish_family']}".lower()
                            return not any(d in text_blob for d in h_dislikes)
                        dislike_filtered = role_candidates[role_candidates.apply(passes_dislikes, axis=1)]
                        if not dislike_filtered.empty:
                            role_candidates = dislike_filtered
                            
                    # Ultimate safety fallback if filters ever empty the pool
                    if role_candidates.empty:
                        role_candidates = recipes_df.copy()
                            
                    # 4. Scoring / Weighting for selection
                    weights = []
                    for _, rec in role_candidates.iterrows():
                        w = 1.0
                        # Home region preference weighting
                        if rec['region_clean'] == h_region:
                            w *= 2.5
                        
                        # Template goal shifts
                        if h_goal == 'high-protein' and ('paneer' in str(rec['dish_family']).lower() or 'dal' in str(rec['dish_family']).lower() or 'protein' in str(rec.get('health_tags', '')).lower()):
                            w *= 2.0
                        elif h_goal == 'weight-loss' and rec.get('total_time_min', 30) < 25:
                            w *= 1.5
                            
                        # Fatigue penalty if repeated in the recent week
                        if rec['recipe_code'] in recent_recipes_in_week:
                            w *= 0.1 # Heavily penalize repetition
                            
                        weights.append(max(w, 0.01))
                        
                    weights = np.array(weights)
                    weight_sum = weights.sum()
                    if weight_sum == 0:
                        weights = np.ones(len(role_candidates)) / len(role_candidates)
                    else:
                        weights /= weight_sum
                        
                    selected_recipe = role_candidates.sample(weights=weights).iloc[0]
                    recipe_code = selected_recipe['recipe_code']
                    meal_role = selected_recipe['meal_role_clean']
                    
                    recent_recipes_in_week.append(recipe_code)
                    if len(recent_recipes_in_week) > 5:
                        recent_recipes_in_week.pop(0)
                        
                    meal_plan_items.append({
                        'item_id': item_id_counter,
                        'plan_id': plan_id,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'slot': slot,
                        'recipe_code': recipe_code,
                        'meal_role': meal_role
                    })
                    item_id_counter += 1
                    
                    # Simulate Interactions (Skip for cold-start households if interaction limit reached)
                    if is_cold_start and interaction_count_for_plan >= 4:
                        continue # Keep cold-start total interactions < 5
                        
                    # Determine action: accepted, swapped, skipped
                    # Fatigue logic: swap probability rises if recipe repeats or is disliked
                    swap_prob = 0.08
                    if recipe_code in recent_recipes_in_week[:-1]:
                        swap_prob = 0.35 # Fatigue trigger
                        
                    skip_prob = 0.05
                    
                    rand_val = random.random()
                    if rand_val < swap_prob:
                        action = 'swapped'
                        # Pick a swap alternative
                        alt_candidates = role_candidates[role_candidates['recipe_code'] != recipe_code]
                        if not alt_candidates.empty:
                            swapped_to = alt_candidates.sample(1).iloc[0]['recipe_code']
                        else:
                            swapped_to = recipe_code
                    elif rand_val < (swap_prob + skip_prob):
                        action = 'skipped'
                        swapped_to = None
                    else:
                        action = 'accepted'
                        swapped_to = None
                        
                    # Cook noise: 10% of accepted items are never cooked
                    if action == 'accepted' and random.random() > 0.10:
                        # Log cooked interaction or supplementary action
                        pass
                        
                    interactions.append({
                        'interaction_id': interaction_id_counter,
                        'household_id': h_id,
                        'plan_id': plan_id,
                        'recipe_code': recipe_code,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'action': action,
                        'swapped_to_recipe_code': swapped_to,
                        'created_at': current_date.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    interaction_id_counter += 1
                    interaction_count_for_plan += 1
                    plan_has_interaction = True
                    
            # Order generation with noise (some plans abandoned without orders)
            if plan_has_interaction and not is_cold_start and random.random() > 0.12:
                orders.append({
                    'order_id': f"ORD{order_id_counter:05d}",
                    'household_id': h_id,
                    'plan_id': plan_id,
                    'source': 'meal_plan',
                    'order_date': plan_start.strftime('%Y-%m-%d'),
                    'created_at': plan_start.strftime('%Y-%m-%d %H:%M:%S')
                })
                order_id_counter += 1

    # Convert to DataFrames
    households_df = pd.DataFrame(households)
    dislikes_df = pd.DataFrame(food_dislikes)
    plans_df = pd.DataFrame(meal_plans)
    items_df = pd.DataFrame(meal_plan_items)
    interactions_df = pd.DataFrame(interactions)
    orders_df = pd.DataFrame(orders)
    
    # Export to CSV
    processed_dir = os.path.join("data", "processed")
    households_df.to_csv(os.path.join(processed_dir, "households.csv"), index=False)
    dislikes_df.to_csv(os.path.join(processed_dir, "food_dislikes.csv"), index=False)
    plans_df.to_csv(os.path.join(processed_dir, "meal_plans.csv"), index=False)
    items_df.to_csv(os.path.join(processed_dir, "meal_plan_items.csv"), index=False)
    interactions_df.to_csv(os.path.join(processed_dir, "interactions.csv"), index=False)
    orders_df.to_csv(os.path.join(processed_dir, "orders.csv"), index=False)
    
    print("=== Synthetic Data Generation Complete ===")
    print(f"Generated:")
    print(f" - Households: {len(households_df)}")
    print(f" - Food Dislikes: {len(dislikes_df)}")
    print(f" - Meal Plans: {len(plans_df)}")
    print(f" - Meal Plan Items: {len(items_df)}")
    print(f" - Interactions: {len(interactions_df)}")
    print(f" - Orders: {len(orders_df)}")
    print(f"All files exported successfully to {processed_dir}/")

if __name__ == "__main__":
    generate_synthetic_data()