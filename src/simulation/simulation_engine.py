import random
import pandas as pd
from datetime import datetime, timedelta

from .recipe_filters import prepare_recipes, filter_by_diet, filter_by_slot, filter_by_dislikes
from .scoring import calculate_recipe_weights
from .behavior_simulator import simulate_interaction
from .order_generator import should_generate_order


def run_simulation(recipes_df, households_df, dislikes_df):
    """Orchestrates 90 days of meal plans, recipe scoring with time-decayed history, interactions, and orders."""
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

    # Pre-group food dislikes into a dictionary for O(1) lookup per household
    dislikes_dict = {
        h_id: set(group['ingredient_or_dish'].str.lower())
        for h_id, group in dislikes_df.groupby('household_id')
    }

    for _, h in households_df.iterrows():
        h_id = h['household_id']
        h_diet = h['diet']
        h_region = h['home_region']
        h_goal = h['health_goal']
        is_cold_start = h_id in cold_start_households

        h_dislikes = dislikes_dict.get(h_id, set())
        diet_filtered_recipes = filter_by_diet(clean_recipes_df, h_diet)

        # Track household-specific historical interactions across the simulation window for time decay
        household_interaction_history = []
        cached_history_df = None

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

                    # 2. Prepare history DataFrame for time-decayed scoring (memoized safely per household)
                    history_df = None
                    if not is_cold_start and household_interaction_history:
                        if cached_history_df is None or len(household_interaction_history) != len(cached_history_df):
                            cached_history_df = pd.DataFrame(household_interaction_history)
                        history_df = cached_history_df

                    # 3. Score and sample recipe using centralized scoring with time decay
                    weights = calculate_recipe_weights(
                        role_candidates, h_region, h_goal, last_eaten_date, current_date, history_df=history_df
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

                    # 4. Simulate user interaction (respecting cold-start interaction limits)
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

                        # Append to rolling household history for subsequent weeks' time-decayed scoring
                        household_interaction_history.append({
                            'recipe_code': recipe_code,
                            'dish_family': selected_recipe.get('dish_family'),
                            'region': selected_recipe.get('region'),
                            'interaction_date': current_date.strftime('%Y-%m-%d'),
                            'action': action
                        })

                        interaction_id_counter += 1
                        interaction_count_for_plan += 1
                        plan_has_interaction = True

                    # Update last eaten date after interaction check so recency swap logic remains accurate
                    last_eaten_date[recipe_code] = current_date

            # 5. Evaluate weekly order conversion
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