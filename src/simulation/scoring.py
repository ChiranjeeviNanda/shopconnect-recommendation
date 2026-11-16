import numpy as np

def calculate_recipe_weights(candidates_df, home_region, health_goal, last_eaten_date, current_date):
    """Calculates normalized selection probabilities for candidate recipes."""
    weights = []

    for _, rec in candidates_df.iterrows():
        w = 1.0
        if rec['region_clean'] == home_region:
            w *= 2.5

        dish_family_str = str(rec['dish_family']).lower()
        health_tags_str = str(rec.get('health_tags', '')).lower()

        if health_goal == 'high-protein' and (
            'paneer' in dish_family_str or 'dal' in dish_family_str or 'protein' in health_tags_str
        ):
            w *= 2.0
        elif health_goal == 'weight-loss' and rec.get('total_time_min', 30) < 25:
            w *= 1.5

        # Apply a gradual cooldown penalty based on how many days ago the recipe was last used
        if rec['recipe_code'] in last_eaten_date:
            days_since = (current_date - last_eaten_date[rec['recipe_code']]).days
            if days_since < 4:
                w *= 0.01  # Strong penalty within 4 days
            elif days_since < 10:
                w *= 0.15  # Moderate penalty between 4 and 9 days
            elif days_since < 18:
                w *= 0.50  # Light penalty between 10 and 17 days

        weights.append(max(w, 0.01))

    weights = np.array(weights)
    weight_sum = weights.sum()

    if weight_sum == 0:
        return np.ones(len(candidates_df)) / len(candidates_df)

    return weights / weight_sum