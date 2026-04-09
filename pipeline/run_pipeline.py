"""
Main pipeline orchestrator — run this once before deploying.

Step 1: compute_aggregates  → reads CSV → stores stats in Supabase
Step 2: generate_insights   → reads stats → calls Ollama → stores insights

That's it. No more per-combination generation.
10 insights cover every user profile.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.compute_aggregates import run as run_aggregates
from pipeline.generate_insights  import run as run_insights


def main() -> None:
    print("\n" + "=" * 55)
    print("  SALARY PREDICTOR — PIPELINE")
    print("=" * 55)

    # Step 1
    run_aggregates()

    # Step 2
    run_insights()

    print("\n" + "=" * 55)
    print("  Pipeline complete. Dashboard is ready.")
    print("=" * 55)


if __name__ == "__main__":
    main()