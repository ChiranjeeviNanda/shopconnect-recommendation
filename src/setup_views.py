import os
import logging
from sqlalchemy import create_engine, text
from config import DATABASE_URL, LOG_DIR, VIEWS_PATH, DB_NAME, DB_HOST, DB_PORT

# Configure logging for views setup monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "views_pipeline.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def get_db_engine():
    """Create and return a SQLAlchemy database engine."""
    logger.info(f"Connecting to database {DB_NAME} on {DB_HOST}:{DB_PORT}...")
    engine = create_engine(DATABASE_URL)
    return engine

def run_views_setup(engine):
    """Execute views DDL script to create or refresh database views."""
    if os.path.exists(VIEWS_PATH):
        logger.info(f"Applying views DDL from {VIEWS_PATH}...")
        with open(VIEWS_PATH, "r") as f:
            sql_script = f.read()
        with engine.begin() as connection:
            connection.execute(text(sql_script))
        logger.info("Views DDL successfully applied.")
    else:
        logger.error(f"Views file not found at {VIEWS_PATH}.")
        raise FileNotFoundError(f"Missing required views file: {VIEWS_PATH}")

def main():
    """Run the end-to-end database views setup workflow."""
    engine = get_db_engine()
    
    logger.info("--- Starting Database Views Setup ---")
    run_views_setup(engine)
    logger.info("--- All database views are successfully created/refreshed ---")

if __name__ == "__main__":
    main()