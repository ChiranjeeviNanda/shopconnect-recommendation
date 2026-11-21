import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL
from src.logging_config import configure_logging
from src.recommendation_engine import RecommendationEngine
from src.simulation.popularity import calculate_popularity_scores
from src.simulation.recipe_filters import (
    prepare_recipes,
    filter_by_diet,
    filter_by_slot,
    filter_by_dislikes,
)

logger = configure_logging("evaluate_engine.log", __name__)


def compute_metrics_at_k(recommended_codes: list, actual_code: str, k: int = 5) -> tuple[int, float]:
    """Calculate Hit@K (Recall@K for single target) and NDCG@K for one target interaction."""
    rec_k = recommended_codes[:k]
    hit = 1 if actual_code in rec_k else 0

    ndcg = 0.0
    if hit:
        idx = rec_k.index(actual_code)
        ndcg = 1.0 / np.log2(idx + 2)

    return hit, ndcg


class PopularityBaseline:
    """Rank active recipes by popularity_score, optionally prioritizing home region."""
    def __init__(self, db_engine, repeat_window_days: int = 7, use_region_boost: bool = False, cutoff_date: str | None = None):
        self.db_engine = db_engine
        self.repeat_window_days = repeat_window_days
        self.use_region_boost = use_region_boost

        with self.db_engine.connect() as conn:
            recipes_df = pd.read_sql("SELECT * FROM recipes", conn)
            if cutoff_date:
                interactions_df = pd.read_sql("SELECT * FROM interactions", conn)
                plan_items_df = pd.read_sql("SELECT * FROM meal_plan_items", conn)
                recipes_df = calculate_popularity_scores(
                    recipes_df,
                    interactions_df,
                    meal_plan_items_df=plan_items_df,
                    cutoff_date=cutoff_date,
                )

        self.clean_recipes = prepare_recipes(recipes_df)
        if "popularity_score" not in self.clean_recipes.columns:
            self.clean_recipes["popularity_score"] = 0.0
        else:
            self.clean_recipes["popularity_score"] = pd.to_numeric(
                self.clean_recipes["popularity_score"], errors="coerce"
            ).fillna(0.0)

    def _get_household_info(self, household_id: str) -> tuple[dict, set]:
        """Load household profile and food dislikes."""
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
            dislikes = {str(r[0]).lower().strip() for r in dislike_rows if r[0] is not None}

        return dict(prof_row), dislikes

    def _get_recent_recipes(self, household_id: str, current_date: str) -> set:
        """Load recipes planned within repeat_window_days strictly before current_date."""
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
        """Return up to k recipe codes ranked by popularity_score (and home region if enabled)."""
        profile, dislikes = self._get_household_info(household_id)
        recent_recipes = self._get_recent_recipes(household_id, date)

        curr_dt = datetime.strptime(date, "%Y-%m-%d")
        is_weekday = curr_dt.weekday() < 5

        # Apply same hard constraint filtering pipeline
        candidates = filter_by_diet(self.clean_recipes, profile["diet"].lower())
        candidates = filter_by_slot(candidates, slot.lower(), is_weekday)

        role_clean = str(meal_role).lower().strip() if meal_role else ""
        if role_clean and "meal_role_clean" in candidates.columns:
            role_filtered = candidates[candidates["meal_role_clean"] == role_clean]
            if not role_filtered.empty:
                candidates = role_filtered

        candidates = filter_by_dislikes(candidates, dislikes)
        non_repeated = candidates[~candidates["recipe_code"].isin(recent_recipes)]
        if not non_repeated.empty:
            candidates = non_repeated

        if candidates.empty:
            return []

        candidates = candidates.copy()
        if self.use_region_boost:
            home_region = profile["home_region"].lower()
            is_home = (candidates["region_clean"] == home_region).astype(float)
            candidates["blended_score"] = candidates["popularity_score"] * (1.0 + (0.15 * is_home))
            top_k = candidates.sort_values(
                by=["blended_score", "popularity_score"],
                ascending=[False, False]
            ).head(k)
        else:
            top_k = candidates.sort_values(
                by=["popularity_score"],
                ascending=[False]
            ).head(k)

        return top_k["recipe_code"].tolist()


def get_dynamic_split_date(db_engine, train_days: int = 75) -> str:
    """Compute the chronological train/test cutoff date from interactions."""
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


def _summarize_bucket(hits: list, ndcgs: list) -> tuple[int, float, float]:
    """Return total hits, recall/hit rate, and mean NDCG for a metric list."""
    hits_sum = int(np.sum(hits))
    recall = float(np.mean(hits)) if hits else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    return hits_sum, recall, ndcg


def run_offline_evaluation(
    test_start_date: str | None = None,
    train_days: int = 75,
    sample_limit: int | None = None,
    k: int = 5
):
    """Run offline evaluation comparing RecommendationEngine against global and region popularity baselines."""
    db_engine = create_engine(DATABASE_URL)
    personalized_engine = RecommendationEngine(db_url=DATABASE_URL)

    if not test_start_date:
        test_start_date = get_dynamic_split_date(db_engine, train_days=train_days)

    global_baseline = PopularityBaseline(
        db_engine=db_engine,
        use_region_boost=False,
        cutoff_date=test_start_date,
    )
    region_baseline = PopularityBaseline(
        db_engine=db_engine,
        use_region_boost=True,
        cutoff_date=test_start_date,
    )

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

    buckets = {
        "all": {"pers_h": [], "pers_n": [], "glob_h": [], "glob_n": [], "reg_h": [], "reg_n": []},
        "swap": {"pers_h": [], "pers_n": [], "glob_h": [], "glob_n": [], "reg_h": [], "reg_n": []},
        "confirm": {"pers_h": [], "pers_n": [], "glob_h": [], "glob_n": [], "reg_h": [], "reg_n": []},
        "cold_start": {"pers_h": [], "pers_n": [], "glob_h": [], "glob_n": [], "reg_h": [], "reg_n": []},
        "warm": {"pers_h": [], "pers_n": [], "glob_h": [], "glob_n": [], "reg_h": [], "reg_n": []},
    }

    for _, row in test_interactions.iterrows():
        hid = row["household_id"]
        dt_str = str(row["date"])[:10]
        slot = row["slot"]
        role = row["meal_role"]
        target_code = row["target_recipe_code"]
        bucket_key = "swap" if row["action"] == "swapped" else "confirm"

        pers_suggestions = personalized_engine.suggest(household_id=hid, date=dt_str, slot=slot, meal_role=role, k=k)
        pers_codes = [s["recipe_code"] for s in pers_suggestions]
        p_hit, p_ndcg = compute_metrics_at_k(pers_codes, target_code, k=k)

        g_codes = global_baseline.suggest(household_id=hid, date=dt_str, slot=slot, meal_role=role, k=k)
        g_hit, g_ndcg = compute_metrics_at_k(g_codes, target_code, k=k)

        r_codes = region_baseline.suggest(household_id=hid, date=dt_str, slot=slot, meal_role=role, k=k)
        r_hit, r_ndcg = compute_metrics_at_k(r_codes, target_code, k=k)

        is_cold = personalized_engine._get_interaction_count(hid, before_date=dt_str) < 5
        cohort_key = "cold_start" if is_cold else "warm"

        for key in ("all", bucket_key, cohort_key):
            buckets[key]["pers_h"].append(p_hit)
            buckets[key]["pers_n"].append(p_ndcg)
            buckets[key]["glob_h"].append(g_hit)
            buckets[key]["glob_n"].append(g_ndcg)
            buckets[key]["reg_h"].append(r_hit)
            buckets[key]["reg_n"].append(r_ndcg)

    total_n = len(test_interactions)
    g_sum, g_rec, g_ndcg = _summarize_bucket(buckets["all"]["glob_h"], buckets["all"]["glob_n"])
    r_sum, r_rec, r_ndcg = _summarize_bucket(buckets["all"]["reg_h"], buckets["all"]["reg_n"])
    p_sum, p_rec, p_ndcg = _summarize_bucket(buckets["all"]["pers_h"], buckets["all"]["pers_n"])

    rec_lift_vs_global = ((p_rec - g_rec) / g_rec * 100) if g_rec > 0 else 0.0
    ndcg_lift_vs_global = ((p_ndcg - g_ndcg) / g_ndcg * 100) if g_ndcg > 0 else 0.0

    print("\n" + "=" * 78)
    print(f"OFFLINE EVALUATION REPORT (Test Start: {test_start_date} | N={total_n} | K={k})")
    print("=" * 78)
    print(f"{'Model':<28} | {f'Hits@{k}':<12} | {f'Recall@{k}':<12} | {f'NDCG@{k}':<12}")
    print("-" * 78)
    print(f"{'Global Popularity':<28} | {f'{g_sum}/{total_n}':<12} | {g_rec:<12.4f} | {g_ndcg:<12.4f}")
    print(f"{'Region + Popularity':<28} | {f'{r_sum}/{total_n}':<12} | {r_rec:<12.4f} | {r_ndcg:<12.4f}")
    print(f"{'Personalized Engine':<28} | {f'{p_sum}/{total_n}':<12} | {p_rec:<12.4f} | {p_ndcg:<12.4f}")
    print("-" * 78)
    print(f"{'Lift vs Global Pop (%)':<28} | {'-':<12} | {rec_lift_vs_global:>+11.2f}% | {ndcg_lift_vs_global:>+11.2f}%")
    print("=" * 78)

    for label, key in [("Swap-Only Events", "swap"), ("Accepted/Cooked Events", "confirm")]:
        cnt = len(buckets[key]["pers_h"])
        if cnt > 0:
            sg_sum, sg_rec, sg_ndcg = _summarize_bucket(buckets[key]["glob_h"], buckets[key]["glob_n"])
            sr_sum, sr_rec, sr_ndcg = _summarize_bucket(buckets[key]["reg_h"], buckets[key]["reg_n"])
            sp_sum, sp_rec, sp_ndcg = _summarize_bucket(buckets[key]["pers_h"], buckets[key]["pers_n"])
            print(f"\n{label} (N={cnt}):")
            print(f"  Global Popularity   -> Hits@{k}: {sg_sum}/{cnt} | Recall@{k}: {sg_rec:.4f} | NDCG@{k}: {sg_ndcg:.4f}")
            print(f"  Region + Popularity -> Hits@{k}: {sr_sum}/{cnt} | Recall@{k}: {sr_rec:.4f} | NDCG@{k}: {sr_ndcg:.4f}")
            print(f"  Personalized Engine -> Hits@{k}: {sp_sum}/{cnt} | Recall@{k}: {sp_rec:.4f} | NDCG@{k}: {sp_ndcg:.4f}")

    for label, key in [("Cold-Start Events (<5 prior interactions)", "cold_start"), ("Warm Events (>=5 prior interactions)", "warm")]:
        cnt = len(buckets[key]["pers_h"])
        if cnt > 0:
            sg_sum, sg_rec, sg_ndcg = _summarize_bucket(buckets[key]["glob_h"], buckets[key]["glob_n"])
            sr_sum, sr_rec, sr_ndcg = _summarize_bucket(buckets[key]["reg_h"], buckets[key]["reg_n"])
            sp_sum, sp_rec, sp_ndcg = _summarize_bucket(buckets[key]["pers_h"], buckets[key]["pers_n"])
            print(f"\n{label} (N={cnt}):")
            print(f"  Global Popularity   -> Hits@{k}: {sg_sum}/{cnt} | Recall@{k}: {sg_rec:.4f} | NDCG@{k}: {sg_ndcg:.4f}")
            print(f"  Region + Popularity -> Hits@{k}: {sr_sum}/{cnt} | Recall@{k}: {sr_rec:.4f} | NDCG@{k}: {sr_ndcg:.4f}")
            print(f"  Personalized Engine -> Hits@{k}: {sp_sum}/{cnt} | Recall@{k}: {sp_rec:.4f} | NDCG@{k}: {sp_ndcg:.4f}")


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
