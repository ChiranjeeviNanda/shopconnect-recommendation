import random
import numpy as np
import pandas as pd
from .recipe_filters import filter_by_diet, filter_by_dislikes

ROLE_BEHAVIOR_MAP = {
    'main': {'swap': 0.08, 'skip': 0.05},
    'anchor': {'swap': 0.06, 'skip': 0.04},
    'lentil': {'swap': 0.07, 'skip': 0.05},
    'snack': {'swap': 0.15, 'skip': 0.10},
    'dessert': {'swap': 0.12, 'skip': 0.08},
    'condiment': {'swap': 0.04, 'skip': 0.03},
    'beverage': {'swap': 0.05, 'skip': 0.02},
    'salad': {'swap': 0.10, 'skip': 0.07},
    'yogurt_side': {'swap': 0.05, 'skip': 0.04}
}

COMPATIBLE_ROLES = {
    'breakfast': ['anchor', 'main', 'snack', 'beverage', 'yogurt_side'],
    'lunch': ['main', 'lentil', 'side', 'anchor', 'salad', 'yogurt_side', 'beverage'],
    'dinner': ['main', 'lentil', 'condiment', 'dessert', 'anchor', 'salad', 'yogurt_side', 'beverage']
}


def calculate_swap_weights(candidates_df, home_region, health_goal):
    """Calculates balanced, non-extreme selection weights to ensure recipe diversity during swaps."""
    weights = np.ones(len(candidates_df))

    # Milder multipliers prevent a few recipes from monopolizing all selections
    if home_region:
        region_mask = candidates_df['region_clean'].values == home_region
        weights[region_mask] *= 1.4

    if health_goal:
        dish_families = candidates_df['dish_family'].str.lower().fillna('').values
        health_tags = candidates_df.get('health_tags', pd.Series([''] * len(candidates_df))).str.lower().fillna('').values
        total_times = candidates_df.get('total_time_min', pd.Series([30] * len(candidates_df))).fillna(30).values

        if health_goal == 'high-protein':
            hp_mask = ['paneer' in df or 'dal' in df or 'protein' in ht for df, ht in zip(dish_families, health_tags)]
            weights[hp_mask] *= 1.3
        elif health_goal == 'weight-loss':
            wl_mask = total_times < 25
            weights[wl_mask] *= 1.3

    weight_sum = weights.sum()
    if weight_sum == 0:
        return np.ones(len(candidates_df)) / len(candidates_df)

    return weights / weight_sum


def simulate_interaction(
    recipes_df, role_candidates, recipe_code, meal_role,
    h_diet, h_dislikes, last_eaten_date, current_date,
    h_goal=None, h_region=None, current_slot='dinner'
):
    """Determines user action and selects a diverse swap target using role-aware fallbacks."""
    role_probs = ROLE_BEHAVIOR_MAP.get(meal_role, {'swap': 0.08, 'skip': 0.05})
    swap_prob = role_probs['swap']
    skip_prob = role_probs['skip']

    if recipe_code in last_eaten_date:
        days_since = (current_date - last_eaten_date[recipe_code]).days
        if days_since < 5:
            swap_prob = max(swap_prob, 0.35)

    rand_val = random.random()

    if rand_val < swap_prob:
        action = 'swapped'
        alt_candidates = role_candidates[role_candidates['recipe_code'] != recipe_code]

        # Role-Aware Fallback: Restrict to slot-compatible roles instead of dropping constraints
        if alt_candidates.empty:
            allowed_roles = COMPATIBLE_ROLES.get(current_slot, ['main', 'anchor', 'lentil'])
            broader_pool = recipes_df[
                (recipes_df['recipe_code'] != recipe_code) & 
                (recipes_df['meal_role_clean'].isin(allowed_roles))
            ]
            broader_pool = filter_by_diet(broader_pool, h_diet)
            broader_pool = filter_by_dislikes(broader_pool, h_dislikes)
            
            alt_candidates = (
                broader_pool if not broader_pool.empty
                else recipes_df[recipes_df['recipe_code'] != recipe_code]
            )

        if not alt_candidates.empty:
            if h_goal or h_region:
                weights = calculate_swap_weights(alt_candidates, h_region, h_goal)
                selected_alt = alt_candidates.sample(weights=weights).iloc[0]
            else:
                random_idx = random.randrange(len(alt_candidates))
                selected_alt = alt_candidates.iloc[random_idx]
            swapped_to = selected_alt['recipe_code']
        else:
            swapped_to = recipe_code

    elif rand_val < (swap_prob + skip_prob):
        action = 'skipped'
        swapped_to = None
    else:
        action = 'accepted'
        if random.random() < 0.10:
            action = 'not_cooked'
        swapped_to = None

    return action, swapped_to