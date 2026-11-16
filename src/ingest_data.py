import os
import logging
import traceback
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from config import DATABASE_URL, RAW_DIR, PROCESSED_DIR, SCHEMA_PATH, DB_NAME, DB_HOST, DB_PORT

# Configure logging for ingestion monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_db_engine():
    """Create and return a SQLAlchemy database engine."""
    logger.info(f"Connecting to database {DB_NAME} on {DB_HOST}:{DB_PORT}...")
    engine = create_engine(DATABASE_URL)
    return engine

def run_schema_setup(engine):
    """Execute schema DDL script to set up tables and constraints."""
    if os.path.exists(SCHEMA_PATH):
        logger.info(f"Applying schema DDL from {SCHEMA_PATH}...")
        with open(SCHEMA_PATH, "r") as f:
            sql_script = f.read()
        with engine.begin() as connection:
            connection.execute(text(sql_script))
        logger.info("Schema DDL successfully applied.")
    else:
        logger.warning(f"Schema file not found at {SCHEMA_PATH}. Assuming tables already exist.")

def ingest_file(engine, file_path, table_name, dtype_mapping=None, parse_dates=None):
    """Read CSV file from a given path and bulk insert rows into a PostgreSQL table."""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Missing required dataset: {file_path}")

    logger.info(f"Loading data from {file_path} into table '{table_name}'...")
    
    df = pd.read_csv(file_path, dtype=dtype_mapping, parse_dates=parse_dates)

    # Drop unused CSV columns that are not defined in the database table schema.
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        db_columns = [col['name'] for col in inspector.get_columns(table_name)]
        dropped_cols = [col for col in df.columns if col not in db_columns]
        if dropped_cols:
            logger.info(f"Dropping unused columns for '{table_name}': {dropped_cols}")
            df = df[[col for col in df.columns if col in db_columns]]
            
    # Diagnostic check
    logger.info(f"DataFrame columns for {table_name}: {list(df.columns)}")
    logger.info(f"Read {len(df):,} rows from {os.path.basename(file_path)}.")

    try:
        with engine.begin() as connection:
            df.to_sql(
                name=table_name,
                con=connection,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
        logger.info(f"Successfully loaded {len(df):,} rows into '{table_name}'.")
    except Exception as e:
        logger.error(f"Failed to load table '{table_name}' due to error:")
        traceback.print_exc()
        raise

def ingest_processed_table(engine, file_name, table_name, dtype_mapping=None, parse_dates=None):
    """Read CSV file from the processed directory and bulk insert rows into a PostgreSQL table."""
    file_path = os.path.join(PROCESSED_DIR, file_name)
    ingest_file(engine, file_path, table_name, dtype_mapping, parse_dates)

def main():
    """Run the end-to-end database ingestion workflow."""
    engine = get_db_engine()
    
    run_schema_setup(engine)

    logger.info("--- Phase 1: Ingesting Independent Parent Tables ---")
    raw_path = os.path.join(RAW_DIR, "recipe_list.csv")
    ingest_file(engine, raw_path, "recipes")
    ingest_processed_table(engine, "households.csv", "households")

    logger.info("--- Phase 2: Ingesting Dependent Child Tables ---")
    ingest_processed_table(engine, "food_dislikes.csv", "food_dislikes")
    ingest_processed_table(engine, "meal_plans.csv", "meal_plans")
    ingest_processed_table(engine, "meal_plan_items.csv", "meal_plan_items")
    ingest_processed_table(engine, "interactions.csv", "interactions")
    ingest_processed_table(engine, "orders.csv", "orders")

    logger.info("--- All tables are successfully ingested ---")

if __name__ == "__main__":
    main()