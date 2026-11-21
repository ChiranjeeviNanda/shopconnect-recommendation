import numpy as np
import pandas as pd

def calculate_recipe_weights(
    candidates_df,
    home_region,
    health_goal,
    last_eaten_date,
    current_date,
    history_df=None,
    target_role=None,
    spice_level=None,
):
    """Calculate normalized recommendation weights for ranking candidate recipes."""
    weights = np.ones(len(candidates_df), dtype=float)

    if target_role and "meal_role_clean" in candidates_df.columns:
        role_mask = candidates_df["meal_role_clean"].values == str(target_role).lower().strip()
        weights[role_mask] *= 1.15

    if "region_clean" in candidates_df.columns:
        region_values = candidates_df["region_clean"].fillna("").astype(str).str.lower().str.strip().values
    elif "region" in candidates_df.columns:
        region_values = candidates_df["region"].fillna("south_india").astype(str).str.lower().str.strip().values
    else:
        region_values = np.array([""] * len(candidates_df))

    region_mask = region_values == str(home_region).lower().strip()
    weights[region_mask] *= 1.20

    if spice_level and "spice_level" in candidates_df.columns:
        spice_norm = str(spice_level).lower().strip()
        spice_values = candidates_df["spice_level"].fillna("").astype(str).str.lower().str.strip().values
        spice_mask = spice_values == spice_norm
        weights[spice_mask] *= 1.08

    dish_families = candidates_df.get("dish_family", pd.Series([""] * len(candidates_df))).str.lower().fillna("").values
    health_tags = candidates_df.get("health_tags", pd.Series([""] * len(candidates_df))).str.lower().fillna("").values
    total_times = candidates_df.get("total_time_min", pd.Series([30] * len(candidates_df))).fillna(30).values

    if health_goal == "high-protein":
        hp_mask = ["paneer" in df or "dal" in df or "protein" in ht for df, ht in zip(dish_families, health_tags)]
        weights[hp_mask] *= 1.10
    elif health_goal == "weight-loss":
        wl_mask = total_times < 25
        weights[wl_mask] *= 1.10

    if last_eaten_date:
        for idx, rec_code in enumerate(candidates_df["recipe_code"].values):
            if rec_code in last_eaten_date:
                days_since = (current_date - last_eaten_date[rec_code]).days
                if days_since < 3:
                    weights[idx] *= 0.10
                elif days_since < 7:
                    weights[idx] *= 0.55
                elif days_since < 14:
                    weights[idx] *= 0.85

    # Apply time-decayed affinity for positive history and a gentle penalty for skips or not_cooked
    if history_df is not None and not history_df.empty:
        if not pd.api.types.is_datetime64_any_dtype(history_df["interaction_date"]):
            history_df = history_df.copy()
            history_df["interaction_date"] = pd.to_datetime(history_df["interaction_date"])

        days_ago = (current_date - history_df["interaction_date"]).dt.days
        valid_mask = days_ago >= 0

        if valid_mask.any():
            valid_history = history_df[valid_mask].copy()
            valid_days = days_ago[valid_mask]

            recency_w = np.select(
                [valid_days <= 10, valid_days <= 30],
                [1.2, 0.7],
                default=0.25,
            )

            action_mult = np.select(
                [
                    valid_history["action"] == "cooked",
                    valid_history["action"] == "accepted",
                    valid_history["action"].isin(["skipped", "not_cooked"]),
                ],
                [1.5, 1.0, -0.4],
                default=0.0,
            )

            valid_history["effective_weight"] = recency_w * action_mult
            positive_history = valid_history[valid_history["effective_weight"] > 0]

            if not positive_history.empty:
                fam_boosts = positive_history.groupby("dish_family")["effective_weight"].sum() * 1.2
                reg_boosts = positive_history.groupby("region")["effective_weight"].sum() * 1.1

                fam_boost_series = candidates_df["dish_family"].map(fam_boosts).fillna(0)
                reg_boost_series = candidates_df["region"].map(reg_boosts).fillna(0)

                total_boost = fam_boost_series + reg_boost_series
                weights *= (1.0 + (total_boost * 0.025))

            negative_history = valid_history[valid_history["effective_weight"] < 0]
            if not negative_history.empty and "recipe_code" in candidates_df.columns:
                neg_counts = negative_history.groupby("recipe_code")["effective_weight"].sum().abs()
                neg_series = candidates_df["recipe_code"].map(neg_counts).fillna(0.0).values
                weights *= np.maximum(0.35, 1.0 - (neg_series * 0.10))

    weights = np.nan_to_num(weights, nan=0.01, posinf=1.0, neginf=0.01)
    weights = np.maximum(weights, 0.01)
    weight_sum = float(weights.sum())

    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.ones(len(candidates_df), dtype=float) / len(candidates_df)

    return weights / weight_sum