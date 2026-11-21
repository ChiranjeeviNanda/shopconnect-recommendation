from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
from .config import DATABASE_URL
from .logging_config import configure_logging

from .simulation.recipe_filters import (
    prepare_recipes,
    filter_by_diet,
    filter_by_slot,
    filter_by_dislikes,
)
from .simulation.scoring import calculate_recipe_weights

logger = configure_logging("recommendation_pipeline.log", "recommendation_engine")

class RecommendationEngine:
    def __init__(self, db_url: str = DATABASE_URL, repeat_window_days: int = 7):
        """Set up the engine with a database connection and repeat window."""
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.repeat_window_days = repeat_window_days
        logger.info(f"Initialized RecommendationEngine with repeat window: {repeat_window_days} days.")

    def _get_household_profile(self, household_id: str) -> dict:
        """Load household profile fields from the database."""
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
        """Load disliked ingredients or dishes for a household."""
        query = text("SELECT ingredient_or_dish FROM food_dislikes WHERE household_id = :hid")
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"hid": household_id}).fetchall()
            dislikes = {str(row[0]).lower().strip() for row in rows if row[0] is not None}
            logger.debug(f"Loaded {len(dislikes)} food dislikes for household {household_id}.")
            return dislikes

    def _get_recent_recipe_codes(self, household_id: str, current_date: str) -> set:
        """Load recipe codes planned within the repeat window strictly before current_date."""
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

    def _get_last_eaten_dates(self, household_id: str, current_date: str, lookback_days: int = 30) -> dict:
        """Load the most recent date each recipe was eaten or accepted before current_date."""
        curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
        window_start = (curr_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        query = text("""
            SELECT 
                CASE 
                    WHEN i.action = 'swapped' AND i.swapped_to_recipe_code IS NOT NULL 
                    THEN i.swapped_to_recipe_code 
                    ELSE i.recipe_code 
                END AS eaten_code,
                MAX(i.date) AS last_date
            FROM interactions i
            WHERE i.household_id = :hid
              AND i.date >= :start_date
              AND i.date < :current_date
              AND i.action IN ('accepted', 'cooked', 'swapped')
            GROUP BY eaten_code
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query, {
                "hid": household_id,
                "start_date": window_start,
                "current_date": current_date
            }).fetchall()

        last_eaten_map = {}
        for code, last_dt in rows:
            if code and last_dt:
                parsed_dt = last_dt if isinstance(last_dt, datetime) else datetime.strptime(str(last_dt)[:10], "%Y-%m-%d")
                last_eaten_map[code] = parsed_dt
        return last_eaten_map

    def _get_interaction_count(self, household_id: str, before_date: str | None = None) -> int:
        """Count historical interactions for a household before before_date when given."""
        if before_date:
            query = text("""
                SELECT COUNT(*) 
                FROM interactions 
                WHERE household_id = :hid AND date < :before_date
            """)
            params = {"hid": household_id, "before_date": before_date}
        else:
            query = text("SELECT COUNT(*) FROM interactions WHERE household_id = :hid")
            params = {"hid": household_id}

        with self.engine.connect() as conn:
            count = conn.execute(query, params).scalar()
            return int(count or 0)

    def _get_interaction_history(self, household_id: str, before_date: str | None = None) -> pd.DataFrame:
        """Load interaction history strictly before before_date when given."""
        if before_date:
            history_query = text("""
                SELECT r.recipe_code, r.dish_family, r.region, i.date AS interaction_date, i.action 
                FROM interactions i
                JOIN recipes r ON i.recipe_code = r.recipe_code
                WHERE i.household_id = :hid 
                  AND i.date < :before_date
                  AND i.action IN ('accepted', 'swapped', 'cooked', 'skipped', 'not_cooked')
            """)
            params = {"hid": household_id, "before_date": before_date}
        else:
            history_query = text("""
                SELECT r.recipe_code, r.dish_family, r.region, i.date AS interaction_date, i.action 
                FROM interactions i
                JOIN recipes r ON i.recipe_code = r.recipe_code
                WHERE i.household_id = :hid AND i.action IN ('accepted', 'swapped', 'cooked', 'skipped', 'not_cooked')
            """)
            params = {"hid": household_id}

        with self.engine.connect() as conn:
            return pd.read_sql(history_query, conn, params=params)

    def suggest(self, household_id: str, date: str, slot: str, meal_role: str, k: int = 5) -> list:
        """Return up to k recommended recipes for a household, date, slot, and meal role."""
        logger.info(f"=== Generating suggestions for Household: {household_id} | Date: {date} | Slot: {slot} | Role: {meal_role} ===")

        profile = self._get_household_profile(household_id)
        dislikes = self._get_household_dislikes(household_id)
        recent_recipes = self._get_recent_recipe_codes(household_id, date)
        last_eaten_map = self._get_last_eaten_dates(household_id, date)

        interaction_count = self._get_interaction_count(household_id, before_date=date)
        is_cold_start = interaction_count < 5

        curr_dt = datetime.strptime(date, "%Y-%m-%d")
        is_weekday = curr_dt.weekday() < 5

        # 1. Load active recipes from PostgreSQL
        with self.engine.connect() as conn:
            recipes_df = pd.read_sql("SELECT * FROM recipes", conn)

        clean_recipes = prepare_recipes(recipes_df)

        # 2. Apply modular filter pipeline (Diet, Slot, Dislikes only - NO hard role drop)
        diet_filtered = filter_by_diet(clean_recipes, profile["diet"].lower())
        slot_filtered = filter_by_slot(diet_filtered, slot.lower(), is_weekday)

        role_clean = str(meal_role).lower().strip() if meal_role else ""
        if role_clean and "meal_role_clean" in slot_filtered.columns:
            role_filtered = slot_filtered[slot_filtered["meal_role_clean"] == role_clean]
            if role_filtered.empty:
                role_filtered = slot_filtered
        else:
            role_filtered = slot_filtered

        dislike_filtered = filter_by_dislikes(role_filtered, dislikes)

        candidates_df = dislike_filtered[~dislike_filtered['recipe_code'].isin(recent_recipes)].copy()
        if candidates_df.empty:
            candidates_df = dislike_filtered.copy()
            if candidates_df.empty:
                return []

        # 4. Fetch interaction history (including swap targets)
        history_df = None
        if not is_cold_start:
            history_df = self._get_interaction_history(household_id, before_date=date)

        # 5. Calculate unified scoring weights with soft role matching
        weights = calculate_recipe_weights(
            candidates_df,
            home_region=profile["home_region"].lower(),
            health_goal=profile["health_goal"].lower(),
            last_eaten_date=last_eaten_map,
            current_date=curr_dt,
            history_df=history_df,
            target_role=role_clean or None,
            spice_level=profile.get("spice_level"),
        )

        if "popularity_score" in candidates_df.columns:
            pop_scores = pd.to_numeric(candidates_df["popularity_score"], errors="coerce").fillna(0.0).values
        else:
            pop_scores = 0.0

        pop_multiplier = 1.0 + (pop_scores / 10.0) * 0.05
        candidates_df['computed_score'] = weights * pop_multiplier

        top_recommendations = candidates_df.sort_values(by='computed_score', ascending=False).head(k)

        return top_recommendations[[
            'recipe_code', 'name', 'meal_role_clean', 'region_clean', 'cook_time_min', 'computed_score'
        ]].to_dict(orient='records')