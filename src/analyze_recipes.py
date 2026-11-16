import os
import logging
import traceback
import pandas as pd
from config import LOG_DIR

# Configure logging for analysis monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "analysis_pipeline.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def analyze_recipes(file_path):
    """
    Analyzes the recipe catalog CSV file, providing dataset shape, missing values,
    max string lengths, categorical distributions, tag formats, and numerical summaries.
    """
    logger.info(f"=== Analyzing Recipe Catalog: {file_path} ===")
    
    # Check if the CSV file exists before attempting to load it
    if not os.path.exists(file_path):
        logger.error(f"Error: File not found at {file_path}. Please ensure recipe_list.csv is placed in data/raw/")
        raise FileNotFoundError(f"Missing required dataset: {file_path}")

    try:
        # Load the dataset into a pandas DataFrame
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
            # Filter out nulls and convert non-strings safely to measure length
            max_len = df[col].dropna().astype(str).map(len).max() if not df[col].dropna().empty else 0
            logger.info(f"   - {col}: max length = {max_len} chars")

        logger.info("4. Categorical Field Distributions & Cardinality:")
        categorical_cols = ['meal_role', 'region', 'course', 'cost_tier', 'dish_family']
        for col in categorical_cols:
            if col in df.columns:
                logger.info(f"\n--- {col} (Unique values: {df[col].nunique()}) ---\n{df[col].value_counts().head(10)}")

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
        
    except Exception as e:
        logger.error(f"Failed to analyze recipe catalog due to error:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    recipes_path = os.path.join("data", "raw", "recipe_list.csv")
    analyze_recipes(recipes_path)