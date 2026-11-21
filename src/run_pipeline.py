import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_STEPS = (
    ("skip_analysis", "analyze_recipes.py", "Recipe analysis"),
    ("skip_generation", "generate_synthetic_data.py", "Synthetic data generation"),
    ("skip_ingestion", "ingest_data.py", "Database ingestion"),
    ("skip_views", "setup_views.py", "Database views setup"),
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the ShopConnect data pipeline in sequence."
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip recipe catalog analysis.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip synthetic data generation.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip database ingestion.",
    )
    parser.add_argument(
        "--skip-views",
        action="store_true",
        help="Skip database views setup.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    for skip_option, script_name, label in PIPELINE_STEPS:
        if getattr(args, skip_option):
            print(f"Skipping {label.lower()}.")
            continue

        script_path = PROJECT_ROOT / "src" / script_name
        print(f"Running {label.lower()}...", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
