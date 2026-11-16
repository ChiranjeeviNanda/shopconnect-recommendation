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


def passes_dislikes(row, dislikes_set):
    """Checks if a recipe avoids all disliked ingredients or dish families."""
    text_blob = f"{row['name']} {row['ingredients_json']} {row['dish_family']}".lower()
    return not any(d in text_blob for d in dislikes_set)


def filter_by_dislikes(candidates_df, dislikes_set):
    """Excludes recipes containing ingredients or dishes disliked by the household."""
    if not dislikes_set:
        return candidates_df

    dislike_filtered = candidates_df[
        candidates_df.apply(lambda row: passes_dislikes(row, dislikes_set), axis=1)
    ]
    return dislike_filtered if not dislike_filtered.empty else candidates_df