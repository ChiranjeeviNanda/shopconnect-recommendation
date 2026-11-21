import re
import pandas as pd

# Shared role lists per meal slot so filtering and swap pools match
SLOT_ROLES = {
    'breakfast': ['anchor', 'main', 'snack', 'beverage', 'yogurt_side'],
    'lunch': ['main', 'lentil', 'side', 'anchor', 'salad', 'yogurt_side', 'beverage'],
    'dinner': ['main', 'lentil', 'condiment', 'dessert', 'anchor', 'salad', 'yogurt_side', 'beverage'],
}


def prepare_recipes(recipes_df):
    """Prepare recipe catalog: filter active and published items, clean fields, infer roles, and drop duplicate names."""
    df = recipes_df.copy()

    # Keep only active recipes so inactive items do not enter simulation or evaluation
    if 'is_active' in df.columns:
        active_vals = df['is_active'].astype(str).str.strip().str.lower()
        df = df[active_vals.isin(['true', 't', '1', 'yes'])].copy()

    # Keep only published recipes so draft and review items do not enter analysis
    if 'status' in df.columns:
        df = df[df['status'].astype(str).str.strip().str.upper() == 'PUBLISHED'].copy()

    df['diet_clean'] = df['diet'].fillna('vegetarian').str.lower().str.strip()
    df['region_clean'] = df['region'].fillna('south_india').str.lower().str.strip()
    df['name_clean'] = df['name'].fillna('').str.lower().str.strip()

    # Drop duplicate dish names to prevent catalog duplication bias
    df = df.drop_duplicates(subset=['name_clean'], keep='first')

    def resolve_role(row):
        """Infer a clean meal role from meal_role or fallback text fields."""
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
    """Filter recipes strictly by household diet without falling back to disallowed diets."""
    if recipes_df.empty or 'diet_clean' not in recipes_df.columns:
        return recipes_df.iloc[0:0].copy()

    diet_norm = str(diet).lower().strip() if diet else ''
    if diet_norm == 'vegetarian':
        return recipes_df[recipes_df['diet_clean'].isin(['vegetarian', 'vegan', 'jain'])].copy()
    if diet_norm == 'vegan':
        return recipes_df[recipes_df['diet_clean'] == 'vegan'].copy()
    if diet_norm == 'jain':
        return recipes_df[recipes_df['diet_clean'] == 'jain'].copy()
    if diet_norm in ('eggetarian', 'egg'):
        return recipes_df[recipes_df['diet_clean'].isin(['vegetarian', 'vegan', 'jain', 'eggetarian', 'egg'])].copy()
    if diet_norm in ('non-vegetarian', 'non_vegetarian', 'non vegetarian'):
        return recipes_df.copy()

    return recipes_df[recipes_df['diet_clean'] == diet_norm].copy()


def filter_by_slot(candidates_df, slot, is_weekday):
    """Filter candidate recipes by meal slot and favor short cook times on weekday breakfasts."""
    if candidates_df.empty:
        return candidates_df.copy()

    slot_norm = str(slot).lower().strip() if slot else ''
    allowed_roles = SLOT_ROLES.get(slot_norm, SLOT_ROLES['dinner'])
    slot_filtered = candidates_df[candidates_df['meal_role_clean'].isin(allowed_roles)]

    if slot_norm == 'breakfast' and is_weekday and not slot_filtered.empty and 'cook_time_min' in slot_filtered.columns:
        quick_filtered = slot_filtered[slot_filtered['cook_time_min'].fillna(999) <= 20]
        if not quick_filtered.empty:
            slot_filtered = quick_filtered

    return slot_filtered.copy() if not slot_filtered.empty else candidates_df.copy()


def filter_by_dislikes(candidates_df, dislikes_set):
    """Remove recipes that match disliked ingredients or dish names."""
    if candidates_df.empty:
        return candidates_df.copy()

    clean_dislikes = {str(item).lower().strip() for item in (dislikes_set or set()) if str(item).strip()}
    if not clean_dislikes:
        return candidates_df.copy()

    # Create a vectorized text blob series to avoid slow row-by-row apply
    name_col = candidates_df['name'].str.lower().fillna('') if 'name' in candidates_df.columns else ''
    ing_col = candidates_df['ingredients_json'].str.lower().fillna('') if 'ingredients_json' in candidates_df.columns else ''
    fam_col = candidates_df['dish_family'].str.lower().fillna('') if 'dish_family' in candidates_df.columns else ''
    text_blobs = name_col + " " + ing_col + " " + fam_col

    # Use a vectorized regex search instead of row-by-row apply
    pattern = '|'.join(sorted(map(re.escape, clean_dislikes), key=len, reverse=True))
    mask = ~text_blobs.str.contains(pattern, regex=True, na=False)
    return candidates_df[mask].copy()