"""B0 smoke test — 5 OSWorld tasks × 5 GEPA iterations, end-to-end plumbing check.

Success criterion (per refine-logs/EXPERIMENT_PLAN.md):
  - All 5 episodes complete (success or fail OK)
  - ≥ 1 schema-valid FCVRPatch lands in BEHAVIORAL_PATCHES
  - CLIP embedding cache initialized
  - No crashes
  - Log saved to results/B0_smoke.json

Usage:
  python scripts/B0_smoke.py \\
    --tasks configs/osworld_smoke_5.json \\
    --backbone-endpoint http://localhost:8000/v1 \\
    --reflection-model claude-opus-4-7 \\
    --iterations 5 \\
    --output results/B0_smoke.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Visual-GEPA B0 smoke test")
    ap.add_argument("--tasks", required=True, help="path to task list JSON")
    ap.add_argument(
        "--backbone-endpoint",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible vLLM endpoint",
    )
    ap.add_argument(
        "--reflection-model",
        default="claude-opus-4-7",
        help="Anthropic model id for FCVR reflection",
    )
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--output", default="results/B0_smoke.json")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # TODO Day-1: instantiate FCVROperator + OSWorldAdapter + GEPA loop
    raise NotImplementedError("Implement during Day-1 / B0.")


if __name__ == "__main__":
    raise SystemExit(main())
