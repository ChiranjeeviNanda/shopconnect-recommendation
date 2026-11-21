import os
import traceback
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from config import DATABASE_URL, RAW_DIR, PROCESSED_DIR, SCHEMA_PATH, DB_NAME, DB_HOST, DB_PORT
from logging_config import configure_logging

logger = configure_logging("ingest_pipeline.log", __name__)


def get_db_engine():
    """Create and return a SQLAlchemy database engine."""
    logger.info(f"Connecting to database {DB_NAME} on {DB_HOST}:{DB_PORT}...")
    engine = create_engine(DATABASE_URL)
    return engine


def run_schema_setup(engine):
    """Run the schema DDL script to create tables and constraints."""
    if os.path.exists(SCHEMA_PATH):
        logger.info(f"Applying schema DDL from {SCHEMA_PATH}...")
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            sql_script = f.read()
        with engine.begin() as connection:
            connection.execute(text(sql_script))
        logger.info("Schema DDL successfully applied.")
    else:
        logger.warning(f"Schema file not found at {SCHEMA_PATH}. Assuming tables already exist.")


def reset_tables(engine, table_names):
    """Clear target tables in reverse dependency order before a fresh ingestion run."""
    inspector = inspect(engine)
    existing_tables = [name for name in reversed(table_names) if inspector.has_table(name)]
    if not existing_tables:
        return

    logger.info(f"Clearing existing rows from tables: {existing_tables}...")
    quoted_tables = ", ".join(f'"{name}"' for name in existing_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {quoted_tables} RESTART IDENTITY CASCADE;"))
    logger.info("Target tables cleared.")


def ingest_file(engine, file_path, table_name, dtype_mapping=None, parse_dates=None):
    """Read a CSV file, align columns with the target table, and insert rows."""
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

    logger.info(f"DataFrame columns for {table_name}: {list(df.columns)}")
    logger.info(f"Read {len(df):,} rows from {os.path.basename(file_path)}.")

    if df.empty:
        logger.info(f"No rows to insert for '{table_name}'. Skipping insert.")
        return

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
    except Exception:
        logger.error(f"Failed to load table '{table_name}' due to error:")
        traceback.print_exc()
        raise


def ingest_processed_table(engine, file_name, table_name, dtype_mapping=None, parse_dates=None):
    """Read a CSV file from PROCESSED_DIR and insert rows into the database."""
    file_path = os.path.join(PROCESSED_DIR, file_name)
    ingest_file(engine, file_path, table_name, dtype_mapping, parse_dates)


def main():
    """Run the database ingestion workflow for parent tables first, then child tables."""
    engine = get_db_engine()

    run_schema_setup(engine)

    table_order = [
        "recipes",
        "households",
        "food_dislikes",
        "meal_plans",
        "meal_plan_items",
        "interactions",
        "orders"
    ]
    reset_tables(engine, table_order)

    logger.info("--- Phase 1: Ingesting Independent Parent Tables ---")
    processed_recipes_path = os.path.join(PROCESSED_DIR, "recipes.csv")
    raw_recipes_path = os.path.join(RAW_DIR, "recipe_list.csv")
    recipes_source_path = processed_recipes_path if os.path.exists(processed_recipes_path) else raw_recipes_path
    ingest_file(engine, recipes_source_path, "recipes")
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