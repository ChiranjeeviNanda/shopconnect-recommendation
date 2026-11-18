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


def calculate_swap_weights(candidates_df, home_region, health_goal, last_eaten_date=None, current_date=None):
    """Calculates balanced swap weights with recency cooldowns."""
    weights = np.ones(len(candidates_df))

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

    # Apply cooldown penalty so recently eaten meals aren't repeatedly chosen as swaps
    if last_eaten_date and current_date:
        for idx, rec_code in enumerate(candidates_df['recipe_code'].values):
            if rec_code in last_eaten_date:
                days_since = (current_date - last_eaten_date[rec_code]).days
                if days_since < 7:
                    weights[idx] *= 0.05
                elif days_since < 14:
                    weights[idx] *= 0.25

    # Downweight recipes that appear multiple times under the same name in the catalog
    name_counts = candidates_df['name_clean'].map(candidates_df['name_clean'].value_counts()).values
    weights = weights / np.maximum(name_counts, 1)

    weight_sum = weights.sum()
    if weight_sum == 0:
        return np.ones(len(candidates_df)) / len(candidates_df)

    return weights / weight_sum


def simulate_interaction(
    recipes_df, role_candidates, recipe_code, meal_role,
    h_diet, h_dislikes, last_eaten_date, current_date,
    h_goal=None, h_region=None
):
    """Determines user action and selects a realistic, same-role swap target."""
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
        
        # Look up current recipe name to prevent swapping into duplicate recipe names
        current_row = recipes_df[recipes_df['recipe_code'] == recipe_code]
        current_name = current_row['name_clean'].iloc[0] if not current_row.empty else ''

        # 1. Primary pool: same slot candidates AND exact same meal_role, excluding same recipe/name
        alt_candidates = role_candidates[
            (role_candidates['recipe_code'] != recipe_code) &
            (role_candidates['name_clean'] != current_name) &
            (role_candidates['meal_role_clean'] == meal_role)
        ]

        # 2. Fallback pool: catalog-wide same meal_role matching diet & dislikes
        if alt_candidates.empty:
            broader_pool = recipes_df[
                (recipes_df['recipe_code'] != recipe_code) &
                (recipes_df['name_clean'] != current_name) &
                (recipes_df['meal_role_clean'] == meal_role)
            ]
            broader_pool = filter_by_diet(broader_pool, h_diet)
            broader_pool = filter_by_dislikes(broader_pool, h_dislikes)
            alt_candidates = broader_pool

        if not alt_candidates.empty:
            weights = calculate_swap_weights(
                alt_candidates, h_region, h_goal, last_eaten_date, current_date
            )
            selected_alt = alt_candidates.sample(weights=weights).iloc[0]
            swapped_to = selected_alt['recipe_code']
        else:
            # If no valid alternative exists, keep original recipe as accepted
            action = 'accepted'
            swapped_to = None

    elif rand_val < (swap_prob + skip_prob):
        action = 'skipped'
        swapped_to = None
    else:
        action = 'accepted'
        if random.random() < 0.10:
            action = 'not_cooked'
        swapped_to = None

    return action, swapped_to