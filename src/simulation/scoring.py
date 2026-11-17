import numpy as np
import pandas as pd

def calculate_recipe_weights(
    candidates_df, home_region, health_goal, last_eaten_date, current_date, history_df=None
):
    """
    Calculates normalized selection probabilities for candidate recipes efficiently,
    incorporating profile alignment, cooldown penalties, and vectorized time-decayed affinity.
    """
    weights = np.ones(len(candidates_df))

    # 1. Home Region Multiplier (Vectorized)
    region_mask = candidates_df['region_clean'].values == home_region
    weights[region_mask] *= 2.5

    # 2. Health Goal Multiplier (Vectorized)
    dish_families = candidates_df['dish_family'].str.lower().fillna('').values
    health_tags = candidates_df.get('health_tags', pd.Series([''] * len(candidates_df))).str.lower().fillna('').values
    total_times = candidates_df.get('total_time_min', pd.Series([30] * len(candidates_df))).fillna(30).values

    if health_goal == 'high-protein':
        hp_mask = ['paneer' in df or 'dal' in df or 'protein' in ht for df, ht in zip(dish_families, health_tags)]
        weights[hp_mask] *= 2.0
    elif health_goal == 'weight-loss':
        wl_mask = total_times < 25
        weights[wl_mask] *= 1.5

    # 3. Cooldown Penalties for Recent Recipes
    if last_eaten_date:
        for idx, rec_code in enumerate(candidates_df['recipe_code'].values):
            if rec_code in last_eaten_date:
                days_since = (current_date - last_eaten_date[rec_code]).days
                if days_since < 4:
                    weights[idx] *= 0.01
                elif days_since < 10:
                    weights[idx] *= 0.15
                elif days_since < 18:
                    weights[idx] *= 0.50

    # 4. Vectorized Time-Decayed Interaction Affinity for Active Households
    if history_df is not None and not history_df.empty:
        # Pre-parse interaction dates once if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(history_df['interaction_date']):
            history_df = history_df.copy()
            history_df['interaction_date'] = pd.to_datetime(history_df['interaction_date'])

        # Calculate days ago vectorized
        days_ago = (current_date - history_df['interaction_date']).dt.days
        valid_mask = days_ago >= 0
        
        if valid_mask.any():
            valid_history = history_df[valid_mask].copy()
            valid_days = days_ago[valid_mask]
            
            # Compute recency weights vector
            recency_w = np.select(
                [valid_days <= 14, valid_days <= 45],
                [1.0, 0.6],
                default=0.2
            )
            
            # Compute action multiplier vector
            action_mult = np.where(
                valid_history['action'].isin(['cooked', 'accepted']), 
                2.0, 
                0.5
            )
            
            valid_history['effective_weight'] = recency_w * action_mult

            # Aggregate affinity boosts per family and region vectorized
            fam_boosts = valid_history.groupby('dish_family')['effective_weight'].sum() * 1.4
            reg_boosts = valid_history.groupby('region')['effective_weight'].sum() * 1.2

            # Map boosts directly to candidates_df using vectorized pandas mapping
            fam_boost_series = candidates_df['dish_family'].map(fam_boosts).fillna(0)
            reg_boost_series = candidates_df['region'].map(reg_boosts).fillna(0)
            
            total_boost = fam_boost_series + reg_boost_series
            weights *= (1.0 + (total_boost * 0.01))

    weights = np.maximum(weights, 0.01)
    weight_sum = weights.sum()

    if weight_sum == 0:
        return np.ones(len(candidates_df)) / len(candidates_df)

    return weights / weight_sum