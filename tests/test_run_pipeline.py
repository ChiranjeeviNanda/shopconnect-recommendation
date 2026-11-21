from pathlib import Path
from unittest.mock import patch

from src import run_pipeline


def test_main_runs_steps_in_order():
    with patch.object(run_pipeline.subprocess, "run") as run:
        result = run_pipeline.main([])

    assert result == 0
    assert [Path(call_args.args[0][1]).name for call_args in run.call_args_list] == [
        "analyze_recipes.py",
        "generate_synthetic_data.py",
        "ingest_data.py",
        "setup_views.py",
    ]


def test_main_skips_requested_steps():
    with patch.object(run_pipeline.subprocess, "run") as run:
        result = run_pipeline.main(
            ["--skip-analysis", "--skip-generation", "--skip-views"]
        )

    assert result == 0
    run.assert_called_once_with(
        [
            run_pipeline.sys.executable,
            str(run_pipeline.PROJECT_ROOT / "src" / "ingest_data.py"),
        ],
        cwd=run_pipeline.PROJECT_ROOT,
        check=True,
    )
