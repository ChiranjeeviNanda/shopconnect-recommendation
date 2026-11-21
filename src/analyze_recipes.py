import os
import traceback
import pandas as pd
from logging_config import configure_logging

logger = configure_logging("analysis_pipeline.log", __name__)


def analyze_recipes(file_path):
    """
    Read the recipe CSV file and log dataset shape, missing values,
    max string lengths, categorical counts (including status and diet),
    tag formats, and numeric summaries.
    """
    logger.info(f"=== Analyzing Recipe Catalog: {file_path} ===")

    if not os.path.exists(file_path):
        logger.error(f"Error: File not found at {file_path}. Place recipe_list.csv in data/raw/")
        raise FileNotFoundError(f"Missing required dataset: {file_path}")

    try:
        df = pd.read_csv(file_path)

        logger.info(f"1. Basic Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        logger.info("2. Column Data Types & Missing Values:")
        summary_df = pd.DataFrame({
            'Data Type': df.dtypes,
            'Null Count': df.isnull().sum(),
            'Null Percentage (%)': (df.isnull().sum() / len(df)) * 100
        })
        logger.info(f"\n{summary_df}")

        logger.info("3. Maximum String Lengths (for VARCHAR sizing optimization):")
        for col in df.columns:
            # Convert non-null values to string to find max character length
            max_len = df[col].dropna().astype(str).map(len).max() if not df[col].dropna().empty else 0
            logger.info(f"   - {col}: max length = {max_len} chars")

        logger.info("4. Categorical Field Distributions & Cardinality:")
        categorical_cols = ['status', 'diet', 'meal_role', 'region', 'course', 'cost_tier', 'dish_family']
        for col in categorical_cols:
            if col in df.columns:
                # Include NaN values so missing statuses or roles remain visible
                counts = df[col].value_counts(dropna=False)
                logger.info(f"\n--- {col} (Unique values: {df[col].nunique(dropna=False)}) ---\n{counts.head(10)}")

        logger.info("5. Array / Tag Field Formatting Check:")
        tag_cols = ['cuisines', 'diet_tags', 'health_tags']
        for col in tag_cols:
            if col in df.columns:
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
                logger.info(f"   - {col}: Sample value -> {repr(sample_val)} (Type: {type(sample_val)})")

                # Check if stored as string representation of lists (e.g., "['veg', 'gluten-free']")
                if isinstance(sample_val, str) and (sample_val.startswith('[') or ',' in sample_val):
                    logger.info(f"    [Note] {col} appears to be serialized text. Ingestion will require parsing (e.g., ast.literal_eval or splitting by comma).")

        logger.info("6. Numerical Field Summary (Cook Time & Popularity):")
        num_cols = ['cook_time_min', 'popularity_score']
        for col in num_cols:
            if col in df.columns:
                logger.info(f"   - {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

        logger.info("=== Analysis Complete ===")

    except Exception:
        logger.error("Failed to analyze recipe catalog due to error:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    recipes_path = os.path.join("data", "raw", "recipe_list.csv")
    analyze_recipes(recipes_path)