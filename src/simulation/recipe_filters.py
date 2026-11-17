import re

def prepare_recipes(recipes_df):
    """Normalizes recipe columns for robust filtering and matching during simulation."""
    df = recipes_df.copy()
    df['diet_clean'] = df['diet'].fillna('vegetarian').str.lower()
    df['region_clean'] = df['region'].fillna('south_india').str.lower()
    df['meal_role_clean'] = df['meal_role'].fillna('main').str.lower()
    return df


def filter_by_diet(recipes_df, diet):
    """Filters the recipe catalog based on household dietary restrictions."""
    if diet == 'vegetarian':
        filtered = recipes_df[recipes_df['diet_clean'].isin(['vegetarian', 'vegan', 'jain'])]
    elif diet == 'vegan':
        filtered = recipes_df[recipes_df['diet_clean'] == 'vegan']
    elif diet == 'jain':
        filtered = recipes_df[recipes_df['diet_clean'] == 'jain']
    else:
        filtered = recipes_df.copy()

    return filtered if not filtered.empty else recipes_df.copy()


def filter_by_slot(candidates_df, slot, is_weekday):
    """Filters candidate recipes based on the meal slot and weekday cook-time limits."""
    if slot == 'breakfast':
        slot_filtered = candidates_df[
            candidates_df['meal_role_clean'].isin(['anchor', 'main', 'snack', 'beverage', 'yogurt_side'])
        ]
        if is_weekday:
            quick_filtered = slot_filtered[slot_filtered['cook_time_min'] <= 20]
            if not quick_filtered.empty:
                slot_filtered = quick_filtered
    elif slot == 'lunch':
        slot_filtered = candidates_df[
            candidates_df['meal_role_clean'].isin(['main', 'lentil', 'side', 'anchor', 'salad', 'yogurt_side', 'beverage'])
        ]
    else:
        slot_filtered = candidates_df[
            candidates_df['meal_role_clean'].isin(['main', 'lentil', 'condiment', 'dessert', 'anchor', 'salad', 'yogurt_side', 'beverage'])
        ]

    return slot_filtered if not slot_filtered.empty else candidates_df


def filter_by_dislikes(candidates_df, dislikes_set):
    """Excludes recipes containing ingredients or dishes disliked by the household efficiently."""
    if not dislikes_set or candidates_df.empty:
        return candidates_df

    # Create a vectorized text blob series to avoid slow row-by-row apply
    text_blobs = (
        candidates_df['name'].str.lower().fillna('') + " " +
        candidates_df['ingredients_json'].str.lower().fillna('') + " " +
        candidates_df['dish_family'].str.lower().fillna('')
    )

    # Use a vectorized regex search instead of row-by-row apply
    pattern = '|'.join(map(re.escape, dislikes_set))
    mask = ~text_blobs.str.contains(pattern, regex=True, na=False)
    dislike_filtered = candidates_df[mask]
    
    return dislike_filtered if not dislike_filtered.empty else candidates_df