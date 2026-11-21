# tests/conftest.py
def pytest_addoption(parser):
    """Add --live-gemini option for pytest."""
    parser.addoption(
        "--live-gemini",
        action="store_true",
        default=False,
        help="Run live Gemini API tests using GEMINI_API_KEY or GOOGLE_API_KEY (off by default).",
    )