import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def run_simulation(recipes_df, households_df, dislikes_df):
    """Simulates 90 days of meal plans, recipe scoring, interactions, and orders."""
    # Identify a subset of households to simulate cold-start behavior (18% of total)
    cold_start_count = int(len(households_df) * 0.18)
    cold_start_households = set(random.sample(list(households_df['household_id']), cold_start_count))
    
    start_date = datetime(2026, 9, 1)
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
    
    # Normalize recipe columns for robust filtering and matching during simulation
    recipes_df['diet_clean'] = recipes_df['diet'].fillna('vegetarian').str.lower()
    recipes_df['region_clean'] = recipes_df['region'].fillna('south_india').str.lower()
    recipes_df['meal_role_clean'] = recipes_df['meal_role'].fillna('main').str.lower()
    
    # Generate weekly meal plans and behavioral interactions
    for _, h in households_df.iterrows():
        h_id = h['household_id']
        h_diet = h['diet']
        h_region = h['home_region']
        h_goal = h['health_goal']
        is_cold_start = h_id in cold_start_households
        
        h_dislikes = set(dislikes_df[dislikes_df['household_id'] == h_id]['ingredient_or_dish'].str.lower())
        
        # Generate meal plans in 7-day increments over the 90-day simulation window
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
            
            recent_recipes_in_week = []
            plan_has_interaction = False
            interaction_count_for_plan = 0
            
            for day_offset in range(7):
                current_date = plan_start + timedelta(days=day_offset)
                is_weekday = current_date.weekday() < 5
                
                for slot in slots:
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

                    # Filter recipe roles appropriate for the specific meal slot and weekday/weekend timing
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
                    else:
                        slot_filtered = role_candidates[role_candidates['meal_role_clean'].isin(['main', 'lentil', 'condiment', 'dessert', 'anchor'])]
                        if not slot_filtered.empty:
                            role_candidates = slot_filtered
                            
                    # Exclude recipes containing ingredients or dishes disliked by the household
                    if h_dislikes:
                        def passes_dislikes(row):
                            text_blob = f"{row['name']} {row['ingredients_json']} {row['dish_family']}".lower()
                            return not any(d in text_blob for d in h_dislikes)
                        dislike_filtered = role_candidates[role_candidates.apply(passes_dislikes, axis=1)]
                        if not dislike_filtered.empty:
                            role_candidates = dislike_filtered
                            
                    if role_candidates.empty:
                        role_candidates = recipes_df.copy()
                            
                    # Calculate recommendation weights based on regional match, health goals, and recent repetition penalty
                    weights = []
                    for _, rec in role_candidates.iterrows():
                        w = 1.0
                        if rec['region_clean'] == h_region:
                            w *= 2.5
                        
                        if h_goal == 'high-protein' and ('paneer' in str(rec['dish_family']).lower() or 'dal' in str(rec['dish_family']).lower() or 'protein' in str(rec.get('health_tags', '')).lower()):
                            w *= 2.0
                        elif h_goal == 'weight-loss' and rec.get('total_time_min', 30) < 25:
                            w *= 1.5
                            
                        if rec['recipe_code'] in recent_recipes_in_week:
                            w *= 0.1
                            
                        weights.append(max(w, 0.01))
                        
                    weights = np.array(weights)
                    weight_sum = weights.sum()
                    if weight_sum == 0:
                        weights = np.ones(len(role_candidates)) / len(role_candidates)
                    else:
                        weights /= weight_sum
                        
                    # Sample a recipe probabilistically from the weighted candidate pool
                    selected_recipe = role_candidates.sample(weights=weights).iloc[0]
                    recipe_code = selected_recipe['recipe_code']
                    meal_role = selected_recipe['meal_role_clean']
                    
                    # Track recently selected recipes to prevent excessive repetition within the week
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
                    
                    if is_cold_start and interaction_count_for_plan >= 4:
                        continue
                        
                    # Determine user interaction action (accepted, swapped, or skipped) based on probabilities
                    swap_prob = 0.08
                    if recipe_code in recent_recipes_in_week[:-1]:
                        swap_prob = 0.35
                        
                    skip_prob = 0.05
                    rand_val = random.random()
                    
                    if rand_val < swap_prob:
                        action = 'swapped'
                        alt_candidates = role_candidates[role_candidates['recipe_code'] != recipe_code]
                        swapped_to = alt_candidates.sample(1).iloc[0]['recipe_code'] if not alt_candidates.empty else recipe_code
                    elif rand_val < (swap_prob + skip_prob):
                        action = 'skipped'
                        swapped_to = None
                    else:
                        action = 'accepted'
                        swapped_to = None
                        
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
                    
            # Generate grocery orders based on plan engagement and non-cold-start status
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

    return (
        pd.DataFrame(meal_plans),
        pd.DataFrame(meal_plan_items),
        pd.DataFrame(interactions),
        pd.DataFrame(orders)
    )