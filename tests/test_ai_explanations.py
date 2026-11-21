import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from sqlalchemy import text

from src.ai_explanations import _PROJECT_ROOT, generate_grounded_explanations
from src.config import DATABASE_URL, LOG_DIR
from src.recommendation_engine import RecommendationEngine

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


@pytest.fixture
def live_gemini(request) -> bool:
    """Return whether --live-gemini was passed on the command line."""
    return bool(request.config.getoption("--live-gemini"))


@pytest.fixture
def engine():
    """Initialize RecommendationEngine against the configured database."""
    return RecommendationEngine(db_url=DATABASE_URL, repeat_window_days=7)


def get_sample_household_id(db_engine) -> str:
    """Fetch an existing household_id from the database."""
    with db_engine.connect() as conn:
        hid = conn.execute(text("SELECT household_id FROM households LIMIT 1")).scalar()
        if not hid:
            pytest.fail("No households found in database. Load data before running tests.")
        return hid


def _read_ai_explanations_log() -> str:
    """Read ai_explanations.log if it exists."""
    log_path = Path(LOG_DIR) / "ai_explanations.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8")


def _has_real_api_key() -> bool:
    """Check whether a real Gemini/Google API key is available in env or .env."""
    if load_dotenv is not None:
        load_dotenv(_PROJECT_ROOT / ".env", override=False)
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return bool(key) and not key.startswith("test-")


def test_explain_without_api_key_uses_fallback(engine, monkeypatch):
    """Verify fallback explanations work and log when no Gemini API key is set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("src.ai_explanations.load_dotenv", None)

    household_id = get_sample_household_id(engine.engine)
    test_date = "2026-09-15"

    plain_results = engine.suggest(
        household_id=household_id,
        date=test_date,
        slot="lunch",
        meal_role="main",
        k=5,
        explain=False,
    )
    explained_results = engine.suggest(
        household_id=household_id,
        date=test_date,
        slot="lunch",
        meal_role="main",
        k=5,
        explain=True,
    )

    assert len(explained_results) == len(plain_results)
    assert [r["recipe_code"] for r in explained_results] == [
        r["recipe_code"] for r in plain_results
    ]
    for item in explained_results:
        assert "explanation" in item
        assert isinstance(item["explanation"], str)
        assert len(item["explanation"].strip()) > 0
        assert "\n" not in item["explanation"].strip()

    if explained_results:
        log_text = _read_ai_explanations_log()
        assert "No API key found. Using fallback explanations." in log_text


def test_explain_with_mock_api_key_calls_genai_and_preserves_order(engine, monkeypatch):
    """Verify genai client is called, logged, and recipe order stays unchanged."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    household_id = get_sample_household_id(engine.engine)
    test_date = "2026-09-15"

    plain_results = engine.suggest(
        household_id=household_id,
        date=test_date,
        slot="lunch",
        meal_role="main",
        k=5,
        explain=False,
    )
    assert plain_results, "Expected non-empty suggestions for test household."

    captured_calls = {}

    class DummyModels:
        def generate_content(self, model, contents, config):
            captured_calls["model"] = model
            captured_calls["contents"] = contents
            captured_calls["config"] = config
            lines = "\n".join(
                f"{r['name']}: region {r['region_clean']} and {r['cook_time_min']} min cook time."
                for r in plain_results
            )
            return SimpleNamespace(text=lines)

    class DummyClient:
        def __init__(self, api_key):
            captured_calls["api_key"] = api_key
            self.models = DummyModels()

    monkeypatch.setattr("src.ai_explanations.genai.Client", DummyClient)

    explained_results = engine.suggest(
        household_id=household_id,
        date=test_date,
        slot="lunch",
        meal_role="main",
        k=5,
        explain=True,
    )

    assert captured_calls["api_key"] == "test-key-123"
    assert [r["recipe_code"] for r in explained_results] == [
        r["recipe_code"] for r in plain_results
    ]
    assert all("explanation" in r and r["explanation"] for r in explained_results)

    log_text = _read_ai_explanations_log()
    assert "Generated" in log_text and "with Gemini API" in log_text


def test_explain_falls_back_on_api_error(monkeypatch):
    """Verify retries run, test stays fast, and fallback logs when Gemini fails."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    attempts = {"count": 0}

    class FailingModels:
        def generate_content(self, model, contents, config):
            attempts["count"] += 1
            raise RuntimeError("API error")

    class FailingClient:
        def __init__(self, api_key):
            self.models = FailingModels()

    monkeypatch.setattr("src.ai_explanations.genai.Client", FailingClient)

    suggestions = [
        {
            "recipe_code": "R1",
            "name": "Moong Dal Chilla",
            "meal_role_clean": "main",
            "region_clean": "South",
            "cook_time_min": 15,
            "computed_score": 1.2,
        }
    ]
    context = {
        "home_region": "South",
        "health_goal": "high-protein",
        "spice_level": "medium",
        "slot": "breakfast",
        "is_weekday": True,
        "is_cold_start": True,
    }

    lines = generate_grounded_explanations(
        suggestions,
        context,
        max_retries=2,
        retry_delay_sec=0,
    )
    assert attempts["count"] == 2
    assert len(lines) == 1
    assert "Moong Dal Chilla" in lines[0]

    log_text = _read_ai_explanations_log()
    assert "Gemini API call failed" in log_text


def test_explain_live_gemini_api(engine, live_gemini):
    """Call the real Gemini API using the configured .env / environment key."""
    if not live_gemini:
        pytest.skip("Pass --live-gemini to run live Gemini API calls.")
    if not _has_real_api_key():
        pytest.skip("No real GEMINI_API_KEY or GOOGLE_API_KEY found in environment or .env.")

    household_id = get_sample_household_id(engine.engine)
    test_date = "2026-09-15"

    plain_results = engine.suggest(
        household_id=household_id,
        date=test_date,
        slot="lunch",
        meal_role="main",
        k=3,
        explain=False,
    )
    assert plain_results, "Expected non-empty suggestions for live test household."

    explained_results = engine.suggest(
        household_id=household_id,
        date=test_date,
        slot="lunch",
        meal_role="main",
        k=3,
        explain=True,
    )

    assert len(explained_results) == len(plain_results)
    assert [r["recipe_code"] for r in explained_results] == [
        r["recipe_code"] for r in plain_results
    ]
    for item in explained_results:
        exp = (item.get("explanation") or "").strip()
        assert exp, f"Missing explanation for {item.get('recipe_code')}"
        assert "\n" not in exp

    log_text = _read_ai_explanations_log()
    assert "with Gemini API" in log_text