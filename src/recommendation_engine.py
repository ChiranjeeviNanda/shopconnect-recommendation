import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
from .config import DATABASE_URL, LOG_DIR

from .simulation.recipe_filters import filter_by_diet, filter_by_slot, filter_by_dislikes
from .simulation.scoring import calculate_recipe_weights
from .simulation.simulation_engine import prepare_recipes

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "recommendation_pipeline.log")

# Setup dedicated logger to avoid pytest / global logging conflicts
logger = logging.getLogger("recommendation_engine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # File Handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)
logging.getLogger("recommendation_engine")


class RecommendationEngine:
    def __init__(self, db_url: str = DATABASE_URL, repeat_window_days: int = 7):
        """
        Initializes the Recommendation Engine connected to PostgreSQL.
        :param db_url: Database connection string from config.py.
        :param repeat_window_days: N-day window to prevent dish repetition (Default N=7).
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.repeat_window_days = repeat_window_days
        logger.info(f"Initialized RecommendationEngine with repeat window: {repeat_window_days} days.")

    def _get_household_profile(self, household_id: str) -> dict:
        """Fetches household profile details from PostgreSQL."""
        query = text("""
            SELECT household_id, household_size, home_region, diet, spice_level, health_goal 
            FROM households 
            WHERE household_id = :hid
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"hid": household_id}).mappings().fetchone()
            if not result:
                logger.error(f"Household ID '{household_id}' not found in database.")
                raise ValueError(f"Household ID '{household_id}' not found.")
            return dict(result)

    def _get_household_dislikes(self, household_id: str) -> set:
        """Fetches the set of ingredients or dishes disliked by the household."""
        query = text("SELECT ingredient_or_dish FROM food_dislikes WHERE household_id = :hid")
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"hid": household_id}).fetchall()
            dislikes = {row[0].lower().strip() for row in rows}
            logger.debug(f"Loaded {len(dislikes)} food dislikes for household {household_id}.")
            return dislikes

    def _get_recent_recipe_codes(self, household_id: str, current_date: str) -> set:
        """Retrieves recipe codes cooked/planned within the N-day repeat window."""
        curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
        window_start = (curr_dt - timedelta(days=self.repeat_window_days)).strftime("%Y-%m-%d")

        query = text("""
            SELECT DISTINCT mpi.recipe_code 
            FROM meal_plan_items mpi
            JOIN meal_plans mp ON mpi.plan_id = mp.plan_id
            WHERE mp.household_id = :hid 
              AND mpi.date >= :start_date 
              AND mpi.date < :current_date
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query, {
                "hid": household_id,
                "start_date": window_start,
                "current_date": current_date
            }).fetchall()
            recent = {row[0] for row in rows}
            logger.debug(f"Found {len(recent)} recently repeated recipes within {self.repeat_window_days} days for {household_id}.")
            return recent

    def _get_interaction_count(self, household_id: str) -> int:
        """Counts total historical interactions to determine cold-start status (<5 interactions)."""
        query = text("SELECT COUNT(*) FROM interactions WHERE household_id = :hid")
        with self.engine.connect() as conn:
            count = conn.execute(query, {"hid": household_id}).scalar()
            return count

    def _get_interaction_history(self, household_id: str) -> pd.DataFrame:
        """Fetches historical positive/swap interactions with dates for time-decayed scoring."""
        history_query = text("""
            SELECT r.recipe_code, r.dish_family, r.region, i.date AS interaction_date, i.action 
            FROM interactions i
            JOIN recipes r ON i.recipe_code = r.recipe_code
            WHERE i.household_id = :hid AND i.action IN ('accepted', 'swapped', 'cooked')
        """)
        with self.engine.connect() as conn:
            return pd.read_sql(history_query, conn, params={"hid": household_id})

    def suggest(self, household_id: str, date: str, slot: str, meal_role: str) -> list:
        """
        Suggests top 5 recipes for a given household, date, slot, and meal_role 
        leveraging modular filters, centralized time-decayed scoring weights, and PostgreSQL.
        """
        logger.info(f"=== Generating suggestions for Household: {household_id} | Date: {date} | Slot: {slot} | Role: {meal_role} ===")

        profile = self._get_household_profile(household_id)
        dislikes = self._get_household_dislikes(household_id)
        recent_recipes = self._get_recent_recipe_codes(household_id, date)

        interaction_count = self._get_interaction_count(household_id)
        is_cold_start = interaction_count < 5
        logger.info(f"Household Profile -> Diet: {profile['diet']}, Region: {profile['home_region']}, Goal: {profile['health_goal']}, Cold-Start: {is_cold_start} ({interaction_count} interactions)")

        curr_dt = datetime.strptime(date, "%Y-%m-%d")
        is_weekday = curr_dt.weekday() < 5

        # 1. Load active recipes from PostgreSQL and prepare normalized columns
        with self.engine.connect() as conn:
            recipes_df = pd.read_sql("SELECT * FROM recipes WHERE is_active = 'Y'", conn)

        clean_recipes = prepare_recipes(recipes_df)
        logger.info(f"Loaded {len(clean_recipes)} active recipes from database catalog.")

        # 2. Apply modular filter pipeline (`recipe_filters.py`)
        diet_filtered = filter_by_diet(clean_recipes, profile["diet"].lower())
        slot_filtered = filter_by_slot(diet_filtered, slot.lower(), is_weekday)

        # Enforce exact meal_role match (falling back to slot pool if restricted)
        role_filtered = slot_filtered[slot_filtered['meal_role_clean'] == meal_role.lower()]
        if role_filtered.empty:
            logger.warning(f"No exact matches for meal_role '{meal_role}' under slot '{slot}'. Falling back to slot pool.")
            role_filtered = slot_filtered

        dislike_filtered = filter_by_dislikes(role_filtered, dislikes)
        logger.info(f"Candidates remaining after diet, slot, and dislike filters: {len(dislike_filtered)}")

        # 3. Enforce N-day Repeat Window Constraint (Hard constraint)
        candidates_df = dislike_filtered[~dislike_filtered['recipe_code'].isin(recent_recipes)].copy()

        if candidates_df.empty:
            logger.warning("N-day repeat window filter cleared all candidates. Relaxing repeat constraint as fallback.")
            candidates_df = dislike_filtered.copy()
            if candidates_df.empty:
                logger.error("All recipe candidates exhausted after hard constraints.")
                return []

        # Map recent dates back into a lookup dictionary for scoring cool-down penalties
        last_eaten_map = {rc: curr_dt - timedelta(days=1) for rc in recent_recipes}

        # 4. Fetch interaction history for active households to pass into centralized scoring
        history_df = None
        if not is_cold_start:
            history_df = self._get_interaction_history(household_id)
            if not history_df.empty:
                logger.info(f"Loaded {len(history_df)} historical interactions for time-decayed scoring.")

        # 5. Calculate unified scoring weights via `scoring.py`
        weights = calculate_recipe_weights(
            candidates_df,
            home_region=profile["home_region"].lower(),
            health_goal=profile["health_goal"].lower(),
            last_eaten_date=last_eaten_map,
            current_date=curr_dt,
            history_df=history_df
        )

        candidates_df['computed_score'] = weights
        top_recommendations = candidates_df.sort_values(by='computed_score', ascending=False).head(5)

        logger.info(f"Successfully generated {len(top_recommendations)} recommendations.")

        # Return formatted recommendation records for UI slot/swap rendering
        return top_recommendations[[
            'recipe_code', 'name', 'meal_role_clean', 'region_clean', 'cook_time_min', 'computed_score'
        ]].to_dict(orient='records')