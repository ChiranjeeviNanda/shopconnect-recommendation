import numpy as np
import pandas as pd
from src.config import DATABASE_URL
from src.recommendation_engine import RecommendationEngine
from sqlalchemy import create_engine, text

def compute_metrics_at_k(recommended_codes, actual_code, k=5):
    rec_k = recommended_codes[:k]
    hit = 1 if actual_code in rec_k else 0
    
    # NDCG@5 calculation
    ndcg = 0.0
    if hit:
        idx = rec_k.index(actual_code)
        ndcg = 1.0 / np.log2(idx + 2) # idx 0 maps to log2(2) = 1.0
        
    return hit, ndcg

def run_offline_evaluation():
    engine = RecommendationEngine(db_url=DATABASE_URL)
    db_engine = create_engine(DATABASE_URL)
    
    # Define time split boundary (e.g., first 75 days train, last 15 days test)
    test_start_date = "2026-11-15"
    
    query = text("""
        SELECT i.household_id, i.date, i.recipe_code, mpi.slot, mpi.meal_role
        FROM interactions i
        JOIN meal_plan_items mpi ON i.plan_id = mpi.plan_id AND i.date = mpi.date AND i.recipe_code = mpi.recipe_code
        WHERE i.date >= :test_start AND i.action IN ('accepted', 'swapped', 'cooked')
        LIMIT 500
    """)
    
    with db_engine.connect() as conn:
        test_interactions = pd.read_sql(query, conn, params={"test_start": test_start_date})
        
    personalized_hits = []
    personalized_ndcgs = []
    
    for _, row in test_interactions.iterrows():
        suggestions = engine.suggest(
            household_id=row['household_id'],
            date=str(row['date']),
            slot=row['slot'],
            meal_role=row['meal_role']
        )
        rec_codes = [s['recipe_code'] for s in suggestions]
        
        hit, ndcg = compute_metrics_at_k(rec_codes, row['recipe_code'], k=5)
        personalized_hits.append(hit)
        personalized_ndcgs.append(ndcg)
        
    print(f"=== Offline Evaluation Results (N={len(test_interactions)}) ===")
    print(f"Personalized Engine Recall@5: {np.mean(personalized_hits):.4f}")
    print(f"Personalized Engine NDCG@5:   {np.mean(personalized_ndcgs):.4f}")

if __name__ == "__main__":
    run_offline_evaluation()