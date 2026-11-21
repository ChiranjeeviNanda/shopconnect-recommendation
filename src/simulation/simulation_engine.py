import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .recipe_filters import prepare_recipes, filter_by_diet, filter_by_slot, filter_by_dislikes
from .behavior_simulator import simulate_interaction
from .order_generator import should_generate_order


def calculate_generation_weights(
    candidates_df,
    home_region,
    health_goal,
    last_eaten_date,
    current_date,
    spice_level=None,
):
    """Calculate weights for sampling synthetic meal plan items during simulation."""
    weights = np.ones(len(candidates_df), dtype=float)

    if "region_clean" in candidates_df.columns:
        region_values = candidates_df["region_clean"].fillna("").astype(str).str.lower().str.strip().values
    elif "region" in candidates_df.columns:
        region_values = candidates_df["region"].fillna("south_india").astype(str).str.lower().str.strip().values
    else:
        region_values = np.array([""] * len(candidates_df))

    region_mask = region_values == str(home_region).lower().strip()
    weights[region_mask] *= 1.35

    if spice_level and "spice_level" in candidates_df.columns:
        spice_norm = str(spice_level).lower().strip()
        spice_values = candidates_df["spice_level"].fillna("").astype(str).str.lower().str.strip().values
        spice_mask = spice_values == spice_norm
        weights[spice_mask] *= 1.10

    dish_families = candidates_df.get("dish_family", pd.Series([""] * len(candidates_df))).str.lower().fillna("").values
    health_tags = candidates_df.get("health_tags", pd.Series([""] * len(candidates_df))).str.lower().fillna("").values
    total_times = candidates_df.get("total_time_min", pd.Series([30] * len(candidates_df))).fillna(30).values

    if health_goal == "high-protein":
        hp_mask = ["paneer" in df or "dal" in df or "protein" in ht for df, ht in zip(dish_families, health_tags)]
        weights[hp_mask] *= 1.15
    elif health_goal == "weight-loss":
        wl_mask = total_times < 25
        weights[wl_mask] *= 1.15

    # Penalize recently eaten recipes so daily plans rotate dishes naturally
    if last_eaten_date:
        for idx, rec_code in enumerate(candidates_df["recipe_code"].values):
            if rec_code in last_eaten_date:
                days_since = (current_date - last_eaten_date[rec_code]).days
                if days_since < 3:
                    weights[idx] *= 0.08
                elif days_since < 7:
                    weights[idx] *= 0.50
                elif days_since < 14:
                    weights[idx] *= 0.80

    weights = np.nan_to_num(weights, nan=0.01, posinf=1.0, neginf=0.01)
    weights = np.maximum(weights, 0.01)
    weight_sum = float(weights.sum())

    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.ones(len(candidates_df), dtype=float) / len(candidates_df)

    return weights / weight_sum


def run_simulation(recipes_df, households_df, dislikes_df, seed: int = 42):
    """Run a 90-day simulation for meal plans, interactions, and orders."""
    random.seed(seed)
    np.random.seed(seed)

    # Keep cold-start households at 18% (within 15-20%)
    if len(households_df) > 0:
        cold_start_count = max(1, round(len(households_df) * 0.18))
    else:
        cold_start_count = 0

    cold_start_households = set(random.sample(list(households_df["household_id"]), cold_start_count))

    start_date = datetime(2026, 9, 1)
    days_to_simulate = 90
    slots = ["breakfast", "lunch", "dinner"]

    # Spread cold-start households across day 35 to day 86 so some appear in the test window
    household_start_dates = {}
    household_engagement = {}
    for h_id in households_df["household_id"]:
        if h_id in cold_start_households:
            join_offset = random.randint(35, 86)
            household_start_dates[h_id] = start_date + timedelta(days=join_offset)
        else:
            household_start_dates[h_id] = start_date
        household_engagement[h_id] = round(random.uniform(0.55, 1.0), 2)

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
        h_id: set(group["ingredient_or_dish"].str.lower())
        for h_id, group in dislikes_df.groupby("household_id")
    }

    for _, h in households_df.iterrows():
        h_id = h["household_id"]
        h_diet = h["diet"]
        h_region = h["home_region"]
        h_goal = h["health_goal"]
        h_spice = h.get("spice_level")
        is_cold_start = h_id in cold_start_households
        h_start_date = household_start_dates[h_id]
        h_engagement = household_engagement[h_id]

        h_dislikes = dislikes_dict.get(h_id, set())
        diet_filtered_recipes = filter_by_diet(clean_recipes_df, h_diet)
        safe_recipes_df = filter_by_dislikes(diet_filtered_recipes, h_dislikes)

        total_household_interactions = 0
        last_eaten_date = {}

        # Generate meal plans in 7-day increments over the 90-day simulation window
        for week_start_day in range(0, days_to_simulate, 7):
            plan_start = start_date + timedelta(days=week_start_day)
            plan_end = plan_start + timedelta(days=6)

            if plan_end < h_start_date:
                continue

            active_plan_start = max(plan_start, h_start_date)
            plan_id = f"P{plan_id_counter:05d}"
            plan_id_counter += 1

            meal_plans.append({
                "plan_id": plan_id,
                "household_id": h_id,
                "template_goal": h_goal,
                "start_date": active_plan_start.strftime("%Y-%m-%d"),
                "end_date": plan_end.strftime("%Y-%m-%d"),
                "version": 1,
                "created_at": (active_plan_start - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            })

            week_eaten_counts = {}
            plan_interactions = []
            plan_has_interaction = False

            for day_offset in range(7):
                current_date = plan_start + timedelta(days=day_offset)

                if current_date < h_start_date:
                    continue

                is_weekday = current_date.weekday() < 5

                for slot in slots:
                    role_candidates = filter_by_slot(diet_filtered_recipes, slot, is_weekday)
                    role_candidates = filter_by_dislikes(role_candidates, h_dislikes)
                    if role_candidates.empty:
                        role_candidates = filter_by_slot(diet_filtered_recipes, slot, is_weekday)
                    if role_candidates.empty:
                        role_candidates = diet_filtered_recipes.copy()
                    if role_candidates.empty:
                        role_candidates = filter_by_dislikes(clean_recipes_df, h_dislikes)
                    if role_candidates.empty:
                        role_candidates = clean_recipes_df.copy()

                    weights = calculate_generation_weights(
                        role_candidates,
                        h_region,
                        h_goal,
                        last_eaten_date,
                        current_date,
                        spice_level=h_spice,
                    )
                    if (
                        len(role_candidates) == 0
                        or weights is None
                        or len(weights) != len(role_candidates)
                        or not np.isfinite(weights).all()
                        or float(np.sum(weights)) <= 0.0
                    ):
                        selected_recipe = role_candidates.sample(n=1).iloc[0]
                    else:
                        selected_recipe = role_candidates.sample(weights=weights).iloc[0]

                    recipe_code = selected_recipe["recipe_code"]
                    meal_role = selected_recipe.get("meal_role_clean", selected_recipe.get("meal_role", "main"))

                    meal_plan_items.append({
                        "item_id": item_id_counter,
                        "plan_id": plan_id,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "slot": slot,
                        "recipe_code": recipe_code,
                        "meal_role": meal_role,
                    })
                    item_id_counter += 1

                    # Cap cold-start households at fewer than 5 interactions (max 4)
                    can_interact = True
                    if is_cold_start and total_household_interactions >= 4:
                        can_interact = False

                    if can_interact:
                        action, swapped_to = simulate_interaction(
                            safe_recipes_df,
                            role_candidates,
                            recipe_code,
                            meal_role,
                            h_diet,
                            h_dislikes,
                            last_eaten_date,
                            current_date,
                            h_goal=h_goal,
                            h_region=h_region,
                            week_eaten_counts=week_eaten_counts,
                            slot=slot,
                            engagement_rate=h_engagement,
                        )

                        interaction_record = {
                            "interaction_id": interaction_id_counter,
                            "household_id": h_id,
                            "plan_id": plan_id,
                            "recipe_code": recipe_code,
                            "date": current_date.strftime("%Y-%m-%d"),
                            "action": action,
                            "swapped_to_recipe_code": swapped_to,
                            "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        interactions.append(interaction_record)
                        plan_interactions.append(interaction_record)

                        interaction_id_counter += 1
                        total_household_interactions += 1
                        plan_has_interaction = True

                    actual_eaten_code = swapped_to if (can_interact and action == "swapped" and swapped_to) else recipe_code
                    if not (can_interact and action == "skipped"):
                        last_eaten_date[actual_eaten_code] = current_date
                        week_eaten_counts[actual_eaten_code] = week_eaten_counts.get(actual_eaten_code, 0) + 1

            if plan_interactions and should_generate_order(h_goal, plan_interactions, plan_has_interaction, is_cold_start):
                orders.append({
                    "order_id": f"ORD{order_id_counter:05d}",
                    "household_id": h_id,
                    "plan_id": plan_id,
                    "source": "meal_plan",
                    "order_date": active_plan_start.strftime("%Y-%m-%d"),
                    "created_at": active_plan_start.strftime("%Y-%m-%d %H:%M:%S"),
                })
                order_id_counter += 1

    return (
        pd.DataFrame(meal_plans),
        pd.DataFrame(meal_plan_items),
        pd.DataFrame(interactions),
        pd.DataFrame(orders),
    )