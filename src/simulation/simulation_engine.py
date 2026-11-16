import random
import pandas as pd
from datetime import datetime, timedelta

from .recipe_filters import filter_by_diet, filter_by_slot, filter_by_dislikes
from .scoring import calculate_recipe_weights
from .behavior_simulator import simulate_interaction
from .order_generator import should_generate_order


def prepare_recipes(recipes_df):
    """Normalizes recipe columns for robust filtering and matching during simulation."""
    df = recipes_df.copy()
    df['diet_clean'] = df['diet'].fillna('vegetarian').str.lower()
    df['region_clean'] = df['region'].fillna('south_india').str.lower()
    df['meal_role_clean'] = df['meal_role'].fillna('main').str.lower()
    return df


def run_simulation(recipes_df, households_df, dislikes_df):
    """Orchestrates 90 days of meal plans, recipe scoring, interactions, and orders."""
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

    clean_recipes_df = prepare_recipes(recipes_df)

    for _, h in households_df.iterrows():
        h_id = h['household_id']
        h_diet = h['diet']
        h_region = h['home_region']
        h_goal = h['health_goal']
        is_cold_start = h_id in cold_start_households

        h_dislikes = set(dislikes_df[dislikes_df['household_id'] == h_id]['ingredient_or_dish'].str.lower())
        diet_filtered_recipes = filter_by_diet(clean_recipes_df, h_diet)

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

            last_eaten_date = {}
            plan_interactions = []
            plan_has_interaction = False
            interaction_count_for_plan = 0

            for day_offset in range(7):
                current_date = plan_start + timedelta(days=day_offset)
                is_weekday = current_date.weekday() < 5

                for slot in slots:
                    # 1. Filter candidate recipes
                    role_candidates = filter_by_slot(diet_filtered_recipes, slot, is_weekday)
                    role_candidates = filter_by_dislikes(role_candidates, h_dislikes)
                    if role_candidates.empty:
                        role_candidates = clean_recipes_df.copy()

                    # 2. Score and sample recipe
                    weights = calculate_recipe_weights(
                        role_candidates, h_region, h_goal, last_eaten_date, current_date
                    )
                    selected_recipe = role_candidates.sample(weights=weights).iloc[0]
                    recipe_code = selected_recipe['recipe_code']
                    meal_role = selected_recipe['meal_role_clean']

                    meal_plan_items.append({
                        'item_id': item_id_counter,
                        'plan_id': plan_id,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'slot': slot,
                        'recipe_code': recipe_code,
                        'meal_role': meal_role
                    })
                    item_id_counter += 1

                    # 3. Simulate user interaction (respecting cold-start interaction limits)
                    if not (is_cold_start and interaction_count_for_plan >= 4):
                        action, swapped_to = simulate_interaction(
                            clean_recipes_df, role_candidates, recipe_code, meal_role,
                            h_diet, h_dislikes, last_eaten_date, current_date
                        )

                        interaction_record = {
                            'interaction_id': interaction_id_counter,
                            'household_id': h_id,
                            'plan_id': plan_id,
                            'recipe_code': recipe_code,
                            'date': current_date.strftime('%Y-%m-%d'),
                            'action': action,
                            'swapped_to_recipe_code': swapped_to,
                            'created_at': current_date.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        interactions.append(interaction_record)
                        plan_interactions.append(interaction_record)

                        interaction_id_counter += 1
                        interaction_count_for_plan += 1
                        plan_has_interaction = True

                    # Update last eaten date after interaction check so recency swap logic remains accurate
                    last_eaten_date[recipe_code] = current_date

            # 4. Evaluate weekly order conversion
            if should_generate_order(h_goal, plan_interactions, plan_has_interaction, is_cold_start):
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