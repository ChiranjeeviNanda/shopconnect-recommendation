import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .logging_config import configure_logging

# Load .env from the project root if available
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(_PROJECT_ROOT / ".env")

logger = configure_logging("ai_explanations.log", "ai_explanations")

SYSTEM_INSTRUCTION = """You write short meal-plan explanations.
Rules:
- Return one short line (under 18 words) per recipe.
- Use only facts from the provided data.
- Do not invent health claims, ingredient lists, or ranking claims.
- Do not change recipe order or select recipes."""


def _format_exception_details(exc: BaseException) -> str:
    """Extract extra diagnostic fields from an exception when available."""
    parts = [f"type={type(exc).__name__}", f"repr={exc!r}"]
    # Inspect common SDK/HTTP error attributes without failing if absent
    for attr in ("code", "status_code", "status", "details", "response"):
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(f"{attr}={val!r}")
    return ", ".join(parts)


def _fallback_explanation(item: dict[str, Any], context: dict[str, Any]) -> str:
    """Return a rule-based line when no API key exists or the API call fails."""
    reasons: list[str] = []
    region = str(item.get("region_clean") or "").strip()
    home_region = str(context.get("home_region") or "").strip()
    cook_time = item.get("cook_time_min")
    goal = str(context.get("health_goal") or "").strip()

    if region and home_region and region.lower() == home_region.lower():
        reasons.append(f"matches {region} home region")

    # Prefer weekday-breakfast phrasing when both slot and weekday flags match
    if context.get("slot") == "breakfast" and context.get("is_weekday") and cook_time is not None:
        reasons.append(f"{cook_time} min weekday breakfast")
    elif cook_time is not None:
        reasons.append(f"{cook_time} min cook time")

    if goal:
        reasons.append(f"fits {goal} goal")

    # Cap at two reasons to keep the fallback explanation concise
    detail = ", ".join(reasons[:2]) if reasons else "passes filters and scoring"
    return f"{item.get('name', 'Dish')}: {detail}."


def generate_grounded_explanations(
    suggestions: list[dict[str, Any]],
    context: dict[str, Any],
    model: str = "gemini-3.8-flash",
    max_retries: int = 2,
    retry_delay_sec: float = 1.5,
) -> list[str]:
    """Generate one grounded line per suggestion using google-genai."""
    if not suggestions:
        logger.info("No suggestions provided. Returning empty explanation list.")
        return []

    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        logger.info("No API key found. Using fallback explanations.")
        return [_fallback_explanation(s, context) for s in suggestions]

    payload = {
        "context": context,
        "suggestions": suggestions,
    }

    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, default=str),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=250,
                ),
            )
            # Strip bullet markers, list numbering, and blank lines so each
            # non-empty line maps cleanly to a single recipe suggestion.
            lines = [
                line.strip().lstrip("-*•0123456789. ").strip()
                for line in (response.text or "").splitlines()
                if line.strip()
            ]
            if len(lines) == len(suggestions):
                logger.info(
                    "Generated %d explanations with Gemini API (model=%s).",
                    len(lines),
                    model,
                )
                return lines

            logger.warning(
                "Gemini returned %d lines for %d suggestions. Using fallback explanations.",
                len(lines),
                len(suggestions),
            )
            return [_fallback_explanation(s, context) for s in suggestions]
        except Exception as exc:
            last_exc = exc
            details = _format_exception_details(exc)
            logger.warning(
                "Gemini API attempt %d of %d failed for model=%s (%s): %s | %s",
                attempt + 1,
                max_retries,
                model,
                len(suggestions),
                exc,
                details,
            )
            if attempt + 1 < max_retries:
                time.sleep(retry_delay_sec)

    if last_exc is not None:
        logger.error(
            "Gemini API call failed after %d attempt(s) for model=%s (%s). Using fallback explanations.",
            max_retries,
            model,
            _format_exception_details(last_exc),
            exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
        )

    return [_fallback_explanation(s, context) for s in suggestions]