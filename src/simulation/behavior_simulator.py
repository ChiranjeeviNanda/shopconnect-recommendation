import random
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


def simulate_interaction(
    recipes_df, role_candidates, recipe_code, meal_role,
    h_diet, h_dislikes, last_eaten_date, current_date
):
    """Determines user action (accepted, swapped, skipped) and selects a swap target if needed."""
    role_probs = ROLE_BEHAVIOR_MAP.get(meal_role, {'swap': 0.08, 'skip': 0.05})
    swap_prob = role_probs['swap']
    skip_prob = role_probs['skip']

    # Increase swap likelihood if the recipe was eaten very recently
    if recipe_code in last_eaten_date:
        days_since = (current_date - last_eaten_date[recipe_code]).days
        if days_since < 5:
            swap_prob = max(swap_prob, 0.35)

    rand_val = random.random()

    if rand_val < swap_prob:
        action = 'swapped'
        alt_candidates = role_candidates[role_candidates['recipe_code'] != recipe_code]

        # Broaden the search pool if strict slot/time filtering yielded no alternatives
        if alt_candidates.empty:
            broader_pool = recipes_df[recipes_df['recipe_code'] != recipe_code]
            broader_pool = filter_by_diet(broader_pool, h_diet)
            broader_pool = filter_by_dislikes(broader_pool, h_dislikes)
            alt_candidates = (
                broader_pool if not broader_pool.empty
                else recipes_df[recipes_df['recipe_code'] != recipe_code]
            )

        swapped_to = (
            alt_candidates.sample(1).iloc[0]['recipe_code']
            if not alt_candidates.empty else recipe_code
        )
    elif rand_val < (swap_prob + skip_prob):
        action = 'skipped'
        swapped_to = None
    else:
        action = 'accepted'
        swapped_to = None

    return action, swapped_to