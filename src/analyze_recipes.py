import os
import pandas as pd

def analyze_recipes(file_path):
    """
    Analyzes the recipe catalog CSV file, providing dataset shape, missing values,
    max string lengths, categorical distributions, tag formats, and numerical summaries.
    """
    print(f"=== Analyzing Recipe Catalog: {file_path} ===")
    
    # Check if the CSV file exists before attempting to load it
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Please ensure recipe_list.csv is placed in data/raw/")
        return

    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    print(f"\n1. Basic Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n2. Column Data Types & Missing Values:")
    summary_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Null Count': df.isnull().sum(),
        'Null Percentage (%)': (df.isnull().sum() / len(df)) * 100
    })
    print(summary_df)
    
    print("\n3. Maximum String Lengths (for VARCHAR sizing optimization):")
    for col in df.columns:
        # Filter out nulls and convert non-strings safely to measure length
        max_len = df[col].dropna().astype(str).map(len).max() if not df[col].dropna().empty else 0
        print(f"   - {col}: max length = {max_len} chars")

    print("\n4. Categorical Field Distributions & Cardinality:")
    categorical_cols = ['meal_role', 'region', 'course', 'cost_tier', 'dish_family']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n--- {col} (Unique values: {df[col].nunique()}) ---")
            print(df[col].value_counts().head(10))

    print("\n5. Array / Tag Field Formatting Check:")
    tag_cols = ['cuisines', 'diet_tags', 'health_tags']
    for col in tag_cols:
        if col in df.columns:
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            print(f"   - {col}: Sample value -> {repr(sample_val)} (Type: {type(sample_val)})")
            
            # Check if stored as string representation of lists (e.g., "['veg', 'gluten-free']")
            if isinstance(sample_val, str) and (sample_val.startswith('[') or ',' in sample_val):
                print(f"    [Note] {col} appears to be serialized text. Ingestion will require parsing (e.g., ast.literal_eval or splitting by comma).")

    print("\n6. Numerical Field Summary (Cook Time & Popularity):")
    num_cols = ['cook_time_min', 'popularity_score']
    for col in num_cols:
        if col in df.columns:
            print(f"   - {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    recipes_path = os.path.join("data", "raw", "recipe_list.csv")
    analyze_recipes(recipes_path)