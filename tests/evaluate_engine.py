import argparse
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, LOG_DIR
from src.recommendation_engine import RecommendationEngine
from src.simulation.recipe_filters import filter_by_diet, filter_by_slot, filter_by_dislikes
from src.simulation.simulation_engine import prepare_recipes

# Configure evaluation logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "evaluate_engine.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def compute_metrics_at_k(recommended_codes: list, actual_code: str, k: int = 5) -> tuple[int, float]:
    """
    Computes Recall@K (Hit Rate) and NDCG@K for a single interaction.
    """
    rec_k = recommended_codes[:k]
    hit = 1 if actual_code in rec_k else 0

    ndcg = 0.0
    if hit:
        idx = rec_k.index(actual_code)
        ndcg = 1.0 / np.log2(idx + 2)  # idx 0 maps to log2(2) = 1.0

    return hit, ndcg


class PopularityBaseline:
    """
    Baseline recommender that ranks active recipes by popularity_score,
    respecting core hard constraints (diet, slot, meal_role, dislikes, and N-day repeat window)
    and prioritizing the household's home region.
    """

    def __init__(self, db_engine, repeat_window_days: int = 7):
        self.db_engine = db_engine
        self.repeat_window_days = repeat_window_days
        with self.db_engine.connect() as conn:
            recipes_df = pd.read_sql("SELECT * FROM recipes WHERE is_active = TRUE", conn)
        self.clean_recipes = prepare_recipes(recipes_df)

    def _get_household_info(self, household_id: str) -> tuple[dict, set]:
        with self.db_engine.connect() as conn:
            prof_row = conn.execute(
                text("SELECT household_id, home_region, diet FROM households WHERE household_id = :hid"),
                {"hid": household_id}
            ).mappings().fetchone()
            if not prof_row:
                raise ValueError(f"Household ID '{household_id}' not found.")

            dislike_rows = conn.execute(
                text("SELECT ingredient_or_dish FROM food_dislikes WHERE household_id = :hid"),
                {"hid": household_id}
            ).fetchall()
            dislikes = {r[0].lower().strip() for r in dislike_rows}

        return dict(prof_row), dislikes

    def _get_recent_recipes(self, household_id: str, current_date: str) -> set:
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
        with self.db_engine.connect() as conn:
            rows = conn.execute(
                query,
                {"hid": household_id, "start_date": window_start, "current_date": current_date}
            ).fetchall()
        return {r[0] for r in rows}

    def suggest(self, household_id: str, date: str, slot: str, meal_role: str, k: int = 5) -> list[str]:
        profile, dislikes = self._get_household_info(household_id)
        recent_recipes = self._get_recent_recipes(household_id, date)

        curr_dt = datetime.strptime(date, "%Y-%m-%d")
        is_weekday = curr_dt.weekday() < 5

        # Apply same hard constraint filtering pipeline
        candidates = filter_by_diet(self.clean_recipes, profile["diet"].lower())
        candidates = filter_by_slot(candidates, slot.lower(), is_weekday)

        role_filtered = candidates[candidates["meal_role_clean"] == meal_role.lower()]
        if not role_filtered.empty:
            candidates = role_filtered

        candidates = filter_by_dislikes(candidates, dislikes)
        non_repeated = candidates[~candidates["recipe_code"].isin(recent_recipes)]
        if not non_repeated.empty:
            candidates = non_repeated

        if candidates.empty:
            return []

        candidates = candidates.copy()
        home_region = profile["home_region"].lower()
        candidates["is_home_region"] = (candidates["region_clean"] == home_region).astype(int)

        # Sort by home region match first, then global popularity_score
        top_k = candidates.sort_values(
            by=["is_home_region", "popularity_score"],
            ascending=[False, False]
        ).head(k)

        return top_k["recipe_code"].tolist()


def get_dynamic_split_date(db_engine, train_days: int = 75) -> str:
    """
    Dynamically determines the chronological train/test cutoff date from the interactions table.
    """
    query = text("SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM interactions")
    with db_engine.connect() as conn:
        row = conn.execute(query).mappings().fetchone()

    if not row or not row["min_date"]:
        raise ValueError("No interactions found in the database to determine split date.")

    min_dt = pd.to_datetime(row["min_date"])
    max_dt = pd.to_datetime(row["max_date"])
    split_dt = min_dt + timedelta(days=train_days)

    if split_dt > max_dt:
        # Fallback to an 80/20 temporal split if dataset spans fewer than `train_days`
        total_days = max(1, (max_dt - min_dt).days)
        split_dt = min_dt + timedelta(days=int(total_days * 0.8))

    return split_dt.strftime("%Y-%m-%d")


def run_offline_evaluation(
    test_start_date: str | None = None,
    train_days: int = 75,
    sample_limit: int | None = None,
    k: int = 5
):
    db_engine = create_engine(DATABASE_URL)
    personalized_engine = RecommendationEngine(db_url=DATABASE_URL)
    baseline_engine = PopularityBaseline(db_engine=db_engine)

    if not test_start_date:
        test_start_date = get_dynamic_split_date(db_engine, train_days=train_days)

    logger.info(f"Running offline evaluation for test window starting: {test_start_date} (K={k})")

    # Resolve target ground-truth recipe:
    # - For 'accepted' or 'cooked', target is i.recipe_code
    # - For 'swapped', target is i.swapped_to_recipe_code (the dish the user actually wanted)
    limit_clause = f"LIMIT {int(sample_limit)}" if sample_limit else ""
    query = text(f"""
        SELECT 
            i.household_id,
            i.date,
            i.action,
            CASE 
                WHEN i.action = 'swapped' AND i.swapped_to_recipe_code IS NOT NULL 
                THEN i.swapped_to_recipe_code 
                ELSE i.recipe_code 
            END AS target_recipe_code,
            mpi.slot,
            mpi.meal_role
        FROM interactions i
        JOIN meal_plan_items mpi 
          ON i.plan_id = mpi.plan_id 
         AND i.date = mpi.date 
         AND i.recipe_code = mpi.recipe_code
        WHERE i.date >= :test_start 
          AND i.action IN ('accepted', 'swapped', 'cooked')
        ORDER BY i.date ASC, i.interaction_id ASC
        {limit_clause}
    """)

    with db_engine.connect() as conn:
        test_interactions = pd.read_sql(query, conn, params={"test_start": test_start_date})

    if test_interactions.empty:
        logger.warning(f"No test interactions found on or after {test_start_date}.")
        return

    # Track metrics overall and segmented by interaction type (slot-fill vs. swap)
    results = {
        "personalized": {"hits": [], "ndcgs": []},
        "baseline": {"hits": [], "ndcgs": []},
        "swap_only": {"pers_hits": [], "pers_ndcgs": [], "base_hits": [], "base_ndcgs": []}
    }

    for _, row in test_interactions.iterrows():
        hid = row["household_id"]
        dt_str = str(row["date"])[:10]
        slot = row["slot"]
        role = row["meal_role"]
        target_code = row["target_recipe_code"]
        is_swap = row["action"] == "swapped"

        # 1. Personalized Engine Suggestions
        pers_suggestions = personalized_engine.suggest(
            household_id=hid,
            date=dt_str,
            slot=slot,
            meal_role=role
        )
        pers_codes = [s["recipe_code"] for s in pers_suggestions]
        p_hit, p_ndcg = compute_metrics_at_k(pers_codes, target_code, k=k)
        results["personalized"]["hits"].append(p_hit)
        results["personalized"]["ndcgs"].append(p_ndcg)

        # 2. Popularity Baseline Suggestions
        base_codes = baseline_engine.suggest(
            household_id=hid,
            date=dt_str,
            slot=slot,
            meal_role=role,
            k=k
        )
        b_hit, b_ndcg = compute_metrics_at_k(base_codes, target_code, k=k)
        results["baseline"]["hits"].append(b_hit)
        results["baseline"]["ndcgs"].append(b_ndcg)

        if is_swap:
            results["swap_only"]["pers_hits"].append(p_hit)
            results["swap_only"]["pers_ndcgs"].append(p_ndcg)
            results["swap_only"]["base_hits"].append(b_hit)
            results["swap_only"]["base_ndcgs"].append(b_ndcg)

    # Compute summary aggregates
    pers_recall = np.mean(results["personalized"]["hits"])
    pers_ndcg = np.mean(results["personalized"]["ndcgs"])
    base_recall = np.mean(results["baseline"]["hits"])
    base_ndcg = np.mean(results["baseline"]["ndcgs"])

    recall_lift = ((pers_recall - base_recall) / base_recall * 100) if base_recall > 0 else 0.0
    ndcg_lift = ((pers_ndcg - base_ndcg) / base_ndcg * 100) if base_ndcg > 0 else 0.0

    print("\n" + "=" * 62)
    print(f"OFFLINE EVALUATION REPORT (Test Start: {test_start_date} | N={len(test_interactions)})")
    print("=" * 62)
    print(f"{'Model':<26} | {f'Recall@{k}':<12} | {f'NDCG@{k}':<12}")
    print("-" * 62)
    print(f"{'Popularity Baseline':<26} | {base_recall:<12.4f} | {base_ndcg:<12.4f}")
    print(f"{'Personalized Engine':<26} | {pers_recall:<12.4f} | {pers_ndcg:<12.4f}")
    print("-" * 62)
    print(f"{'Relative Lift (%)':<26} | {recall_lift:>+11.2f}% | {ndcg_lift:>+11.2f}%")
    print("=" * 62)

    swap_count = len(results["swap_only"]["pers_hits"])
    if swap_count > 0:
        s_pers_rec = np.mean(results["swap_only"]["pers_hits"])
        s_pers_ndcg = np.mean(results["swap_only"]["pers_ndcgs"])
        s_base_rec = np.mean(results["swap_only"]["base_hits"])
        s_base_ndcg = np.mean(results["swap_only"]["base_ndcgs"])
        print(f"\nSwap-Only Subset Breakdown (N={swap_count} swap events):")
        print(f"  Popularity Baseline -> Recall@{k}: {s_base_rec:.4f} | NDCG@{k}: {s_base_ndcg:.4f}")
        print(f"  Personalized Engine -> Recall@{k}: {s_pers_rec:.4f} | NDCG@{k}: {s_pers_ndcg:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation benchmark for ShopConnect Recommendation Engine.")
    parser.add_argument("--test-start", type=str, default=None, help="Explicit test start date (YYYY-MM-DD). Defaults to dynamic 75-day split.")
    parser.add_argument("--train-days", type=int, default=75, help="Number of initial days to treat as training window (default: 75).")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of test interactions to evaluate.")
    parser.add_argument("--k", type=int, default=5, help="Top-K cutoff for Recall@K and NDCG@K (default: 5).")

    args = parser.parse_args()
    run_offline_evaluation(
        test_start_date=args.test_start,
        train_days=args.train_days,
        sample_limit=args.limit,
        k=args.k
    )
