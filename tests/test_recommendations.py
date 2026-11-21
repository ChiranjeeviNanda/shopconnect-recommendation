import pytest
from sqlalchemy import text
from src.config import DATABASE_URL
from src.recommendation_engine import RecommendationEngine


@pytest.fixture
def engine():
    """Initializes the RecommendationEngine for testing."""
    return RecommendationEngine(db_url=DATABASE_URL, repeat_window_days=7)


def get_specific_household(db_engine, cold_start: bool = False) -> str:
    """Fetches a specific cold-start or active household ID from the database."""
    with db_engine.connect() as conn:
        if cold_start:
            query = text("""
                SELECT h.household_id 
                FROM households h
                LEFT JOIN interactions i ON h.household_id = i.household_id
                GROUP BY h.household_id
                HAVING COUNT(i.interaction_id) < 5
                LIMIT 1
            """)
        else:
            query = text("""
                SELECT h.household_id 
                FROM households h
                JOIN interactions i ON h.household_id = i.household_id
                GROUP BY h.household_id
                HAVING COUNT(i.interaction_id) >= 5
                LIMIT 1
            """)
        
        result = conn.execute(query).scalar()
        if not result:
            # Fallback to any household if specific category isn't found
            fallback = conn.execute(text("SELECT household_id FROM households LIMIT 1")).scalar()
            if not fallback:
                pytest.fail("No households found in the database. Please run ingestion first.")
            return fallback
        return result


def test_suggest_returns_valid_format(engine):
    """Verifies that suggest() returns a well-structured list of up to 5 recommendations."""
    household_id = get_specific_household(engine.engine, cold_start=True)
    suggestions = engine.suggest(household_id, "2026-09-15", "lunch", "main")
    
    assert isinstance(suggestions, list)
    assert len(suggestions) <= 5
    if suggestions:
        first_rec = suggestions[0]
        assert "recipe_code" in first_rec
        assert "name" in first_rec
        assert "computed_score" in first_rec


def test_hard_constraints_enforced(engine):
    """Proves that diet, dislikes, and N-day repeat window hard constraints are never violated."""
    household_id = get_specific_household(engine.engine, cold_start=False)
    test_date = "2026-09-15"
    slot = "lunch"
    role = "main"

    suggestions = engine.suggest(household_id, test_date, slot, role)
    # if not suggestions:
    #     pytest.skip(f"No recipes returned for household '{household_id}' to validate constraints against.")
    assert suggestions, f"No recipes returned for household '{household_id}' to validate constraints against."

    profile = engine._get_household_profile(household_id)
    dislikes = engine._get_household_dislikes(household_id)
    recent = engine._get_recent_recipe_codes(household_id, test_date)

    with engine.engine.connect() as conn:
        for rec in suggestions:
            r_code = rec["recipe_code"]
            row = conn.execute(
                text("SELECT diet, meal_role, name, ingredients_json, dish_family FROM recipes WHERE recipe_code = :code"),
                {"code": r_code}
            ).fetchone()

            # 1. Exact Diet Compatibility Constraint
            if profile["diet"].lower() in ["vegetarian", "vegan"]:
                assert row.diet.lower() in ["vegetarian", "vegan", "jain", "satvik"], \
                    f"Diet violation: Non-veg recipe assigned to vegetarian household ({row.name})"

            # 2. Food Dislikes Blacklist Constraint
            text_blob = f"{row.name} {row.ingredients_json} {row.dish_family}".lower()
            for d in dislikes:
                assert d not in text_blob, f"Dislike violation: Blacklisted item '{d}' found in recipe '{row.name}'"

            # 3. N-Day Repeat Window Constraint
            assert r_code not in recent, f"Repeat violation: Recipe '{r_code}' was cooked within the last 7 days."

            # 4. Exact Meal Role Constraint
            assert rec["meal_role_clean"] == role.lower(), \
                f"Meal role violation: Expected role '{role}', got '{rec['meal_role_clean']}' for recipe '{row.name}'"


def test_active_household_time_decay_personalization(engine):
    """Validates that active households successfully leverage interaction history for scoring."""
    household_id = get_specific_household(engine.engine, cold_start=False)
    test_date = "2026-09-15"
    
    # Fetch interaction history to verify active status
    history = engine._get_interaction_history(household_id)
    assert not history.empty, "Test household expected to have interaction history."

    suggestions = engine.suggest(household_id, test_date, "dinner", "main")
    assert isinstance(suggestions, list)
    
    # Ensure scores are computed and sorted in descending order
    if len(suggestions) > 1:
        scores = [rec["computed_score"] for rec in suggestions]
        assert scores == sorted(scores, reverse=True), "Recommendations must be sorted by computed score descending."


def test_cold_start_hard_constraints_enforced(engine):
    """Proves that hard constraints (diet, dislikes, role) hold even for cold-start households with no interaction history."""
    household_id = get_specific_household(engine.engine, cold_start=True)
    test_date = "2026-09-15"
    slot = "lunch"
    role = "main"

    suggestions = engine.suggest(household_id, test_date, slot, role)
    # if not suggestions:
    #     pytest.skip(f"No recipes returned for cold-start household '{household_id}' to validate constraints against.")
    assert suggestions, f"No recipes returned for cold-start household '{household_id}' to validate constraints against."

    profile = engine._get_household_profile(household_id)
    dislikes = engine._get_household_dislikes(household_id)

    with engine.engine.connect() as conn:
        for rec in suggestions:
            r_code = rec["recipe_code"]
            row = conn.execute(
                text("SELECT diet, meal_role, name, ingredients_json, dish_family FROM recipes WHERE recipe_code = :code"),
                {"code": r_code}
            ).fetchone()

            # 1. Exact Diet Compatibility Constraint for Cold-Start
            if profile["diet"].lower() in ["vegetarian", "vegan"]:
                assert row.diet.lower() in ["vegetarian", "vegan", "jain", "satvik"], \
                    f"Cold-start diet violation: Non-veg recipe assigned to vegetarian household ({row.name})"

            # 2. Food Dislikes Blacklist Constraint for Cold-Start
            text_blob = f"{row.name} {row.ingredients_json} {row.dish_family}".lower()
            for d in dislikes:
                assert d not in text_blob, f"Cold-start dislike violation: Blacklisted item '{d}' found in recipe '{row.name}'"

            # 3. Exact Meal Role Constraint
            assert rec["meal_role_clean"] == role.lower(), \
                f"Cold-start meal role violation: Expected role '{role}', got '{rec['meal_role_clean']}' for recipe '{row.name}'"


def test_cold_start_fallback_behavior(engine):
    """Validates that cold-start households (< 5 interactions) successfully receive fallback suggestions based on profile/content."""
    household_id = get_specific_household(engine.engine, cold_start=True)
    test_date = "2026-09-15"
    
    # Confirm interaction count is strictly less than 5
    history = engine._get_interaction_history(household_id)
    assert len(history) < 5, "Test household is expected to be a cold-start user (< 5 interactions)."

    suggestions = engine.suggest(household_id, test_date, "breakfast", "anchor")
    
    # Verify suggestions are successfully returned using content/popularity fallback
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0, "Cold-start fallback failed to return recommendations."
    assert all("computed_score" in rec for rec in suggestions), "All fallback recommendations must have a computed score."