import re
import pandas as pd

def prepare_recipes(recipes_df):
    """Normalizes recipe columns and infers missing meal_role values from course/name."""
    df = recipes_df.copy()
    df['diet_clean'] = df['diet'].fillna('vegetarian').str.lower().str.strip()
    df['region_clean'] = df['region'].fillna('south_india').str.lower().str.strip()
    df['name_clean'] = df['name'].fillna('').str.lower().str.strip()

    def resolve_role(row):
        if pd.notna(row.get('meal_role')) and str(row['meal_role']).strip().lower() not in ('', 'none', 'nan'):
            return str(row['meal_role']).strip().lower()
        
        course = str(row.get('course', '')).lower()
        name = str(row.get('name', '')).lower()
        family = str(row.get('dish_family', '')).lower()
        
        if 'dessert' in course or any(w in name for w in ['cake', 'halwa', 'payasam', 'kheer', 'sweet', 'ladoo']):
            return 'dessert'
        if 'beverage' in course or any(w in name for w in ['tea', 'coffee', 'juice', 'smoothie', 'lassi']):
            return 'beverage'
        if 'salad' in course or 'salad' in name or 'raita' in name:
            return 'salad'
        if 'side' in course or 'chutney' in name or 'pickle' in name:
            return 'condiment'
        if 'dal' in family or 'dal' in name or 'sambar' in name or 'rasam' in name:
            return 'lentil'
        return 'main'

    df['meal_role_clean'] = df.apply(resolve_role, axis=1)
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