import numpy as np
import pandas as pd

ACTION_WEIGHTS = {
    'cooked': 2.0,
    'accepted': 1.0,
    'swapped': -0.5,
    'skipped': -1.0,
    'not_cooked': -0.5,
}

SWAP_TARGET_REWARD = 1.5
SMOOTHING_K = 5.0


def calculate_popularity_scores(recipes_df, interactions_df, meal_plan_items_df=None, cutoff_date=None):
    """Calculate a normalized popularity_score for each recipe using data strictly before cutoff_date."""
    df = recipes_df.copy()
    n_recipes = len(df)

    if n_recipes == 0:
        df['popularity_score'] = pd.Series(dtype=float)
        return df

    filtered_interactions = interactions_df
    if filtered_interactions is not None and not filtered_interactions.empty and cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        inter_dates = pd.to_datetime(filtered_interactions['date'], errors='coerce')
        filtered_interactions = filtered_interactions[inter_dates < cutoff_ts]

    filtered_plans = meal_plan_items_df
    if filtered_plans is not None and not filtered_plans.empty and cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        plan_dates = pd.to_datetime(filtered_plans['date'], errors='coerce')
        filtered_plans = filtered_plans[plan_dates < cutoff_ts]

    code_to_idx = pd.Series(np.arange(n_recipes, dtype=np.intp), index=df['recipe_code'])
    scores = np.zeros(n_recipes, dtype=np.float64)
    exposure = np.zeros(n_recipes, dtype=np.float64)

    has_interactions = filtered_interactions is not None and not filtered_interactions.empty

    if has_interactions:
        action_weights = filtered_interactions['action'].map(ACTION_WEIGHTS).fillna(0.0).to_numpy(dtype=np.float64)
        recipe_idx = filtered_interactions['recipe_code'].map(code_to_idx).to_numpy()
        valid_recipe = ~pd.isna(recipe_idx)

        if np.any(valid_recipe):
            scores += np.bincount(
                recipe_idx[valid_recipe].astype(np.intp),
                weights=action_weights[valid_recipe],
                minlength=n_recipes,
            )

        swap_mask = (filtered_interactions['action'].to_numpy() == 'swapped') & filtered_interactions[
            'swapped_to_recipe_code'
        ].notna().to_numpy()

        if np.any(swap_mask):
            swap_idx = (
                filtered_interactions.loc[swap_mask, 'swapped_to_recipe_code']
                .map(code_to_idx)
                .to_numpy()
            )
            valid_swap = ~pd.isna(swap_idx)
            if np.any(valid_swap):
                valid_swap_idx = swap_idx[valid_swap].astype(np.intp)
                scores += np.bincount(valid_swap_idx, minlength=n_recipes) * SWAP_TARGET_REWARD
                exposure += np.bincount(valid_swap_idx, minlength=n_recipes)

    if filtered_plans is not None and not filtered_plans.empty:
        plan_idx = filtered_plans['recipe_code'].map(code_to_idx).to_numpy()
        valid_plan = ~pd.isna(plan_idx)
        if np.any(valid_plan):
            exposure += np.bincount(plan_idx[valid_plan].astype(np.intp), minlength=n_recipes)

        raw_rate = scores / (exposure + SMOOTHING_K)
    else:
        raw_rate = scores

    min_val = float(np.min(raw_rate))
    max_val = float(np.max(raw_rate))

    if max_val > min_val:
        normalized = ((raw_rate - min_val) / (max_val - min_val)) * 10.0
    else:
        normalized = np.full(n_recipes, 5.0, dtype=np.float64)

    df['popularity_score'] = np.round(normalized, 4)
    return df