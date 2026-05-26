"""B1 baseline — vanilla Qwen3.5-9B across the 5-task hand-picked OSWorld set.

Spans the same apps as the mock B0 + real B1_smoke set, so cross-block
comparisons stay apples-to-apples. Vanilla means NO GEPA, NO FCVR
reflection — just the raw backbone in an agent loop.

Pass criterion for this script (which is plumbing — not the paper baseline):
  - All 5 tasks completed without crash
  - Aggregate success_rate reported (numeric, 0.0 OK)

Out of scope for this script:
  - The 60-task split (R002-R004 in EXPERIMENT_TRACKER.md) — needs a fixed
    split file released first.
  - GEPA iterations / FCVR reflection (B2).
  - Three-seed runs (B1 proper).

Usage (on the server):
  python scripts/B1_baseline.py \\
    --tasks configs/osworld_b1_5.json \\
    --backbone-endpoint http://127.0.0.1:8000/v1 \\
    --backbone-model /root/models/Qwen3.5-9B \\
    --cache-dir /root/visual-gepa/osworld_cache \\
    --max-steps 15 \\
    --rng-seed 42 \\
    --output results/B1_baseline_5task_seed42.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_gepa.osworld_adapter import OSWorldAdapter, load_osworld_task_config
from visual_gepa.structured_prompt import StructuredPrompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("B1_baseline")


# Same SEED structured prompt as B0_smoke / B1_smoke for apples-to-apples.
SEED_PERSONA = (
    "You are an autonomous computer-use agent. You see desktop screenshots and "
    "produce pyautogui-style actions to satisfy the user's task."
)
SEED_GLOBAL_RULES = (
    "1. Read the entire visible screen before acting.\n"
    "2. Prefer keyboard shortcuts when reliable (Ctrl+S, Alt+Tab, etc.).\n"
    "3. After every click, verify the expected UI state change in the next "
    "screenshot before continuing.\n"
    "4. If a step has no effect, do not retry the same coordinate more than "
    "twice; replan instead.\n"
    "5. Emit ONE pyautogui action per turn, fenced in a ```python``` block. "
    "Use WAIT / DONE / FAIL as bare tokens when appropriate."
)
SEED_TASK_SCAFFOLD = (
    "TASK: {instruction}\nProduce one pyautogui action per step. When complete, "
    "emit `DONE`. If the task is infeasible, emit `FAIL`."
)


def build_seed_prompt() -> StructuredPrompt:
    return StructuredPrompt(
        persona=SEED_PERSONA,
        global_rules=SEED_GLOBAL_RULES,
        behavioral_patches=[],
        task_scaffold=SEED_TASK_SCAFFOLD,
    )


def run_one_task(
    task_entry: dict,
    args,
    prompt: StructuredPrompt,
) -> dict:
    """Run a single OSWorld task via OSWorldAdapter; return a result dict."""
    task_id = task_entry.get("id", "<unknown>")
    cfg_path = task_entry.get("task_config_path") or task_entry.get("id")
    logger.info("=== task %s ===", task_id)

    task_cfg = load_osworld_task_config(cfg_path)
    adapter = OSWorldAdapter(
        task_dict=task_cfg,
        vllm_endpoint=args.backbone_endpoint,
        vllm_model=args.backbone_model,
        provider_name=args.provider,
        os_type=args.os_type,
        max_steps=args.max_steps,
        headless=True,
        cache_dir=args.cache_dir,
    )

    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    t0 = time.perf_counter()
    crashed_with: str | None = None
    try:
        traj = adapter.run(prompt)
    except Exception as e:  # noqa: BLE001 — record failure, keep aggregate going
        logger.exception("task %s crashed", task_id)
        crashed_with = f"{type(e).__name__}: {e}"
        traj = None
    elapsed_s = round(time.perf_counter() - t0, 3)
    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    rec = {
        "task_id": task_id,
        "task_config_path": str(cfg_path),
        "instruction": task_cfg.get("instruction"),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": elapsed_s,
        "crashed_with": crashed_with,
    }
    if traj is not None:
        score, feedback = adapter.metric(traj)
        rec.update(
            {
                "n_steps": traj.n_steps,
                "final_reward": float(traj.final_reward),
                "succeeded": traj.succeeded,
                "score": score,
                "feedback": feedback,
                "actions": [(s.action or "")[:200] for s in traj.steps],
            }
        )
    else:
        rec.update({
            "n_steps": 0,
            "final_reward": None,
            "succeeded": False,
            "score": None,
            "feedback": "(crashed)",
            "actions": [],
        })
    return rec


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Visual-GEPA B1 baseline (vanilla Qwen, multi-task)")
    ap.add_argument("--tasks", required=True, help="JSON config: configs/osworld_b1_5.json")
    ap.add_argument(
        "--backbone-endpoint",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    ap.add_argument(
        "--backbone-model",
        default=os.environ.get("VLLM_MODEL_NAME", "/root/models/Qwen3.5-9B"),
    )
    ap.add_argument(
        "--provider",
        default="docker",
        choices=["docker", "vmware", "aws", "aliyun", "azure", "gcp"],
    )
    ap.add_argument("--os-type", default="Ubuntu", choices=["Ubuntu", "Windows"])
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="OSWorld cache dir (Ubuntu.qcow2.zip + per-task files).",
    )
    ap.add_argument("--output", default="results/B1_baseline_5task_seed42.json")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks_cfg = json.loads(Path(args.tasks).read_text())
    tasks = tasks_cfg.get("tasks", [])
    logger.info("loaded %d tasks from %s", len(tasks), args.tasks)

    prompt = build_seed_prompt()
    overall: dict = {
        "B1_baseline": True,
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "tasks_config": args.tasks,
            "backbone_endpoint": args.backbone_endpoint,
            "backbone_model": args.backbone_model,
            "provider": args.provider,
            "os_type": args.os_type,
            "max_steps": args.max_steps,
            "rng_seed": args.rng_seed,
            "n_tasks": len(tasks),
        },
        "tasks": [],
    }
    t_total = time.perf_counter()
    for i, task_entry in enumerate(tasks):
        logger.info("--- task %d/%d : %s ---", i + 1, len(tasks), task_entry.get("id"))
        rec = run_one_task(task_entry, args, prompt)
        overall["tasks"].append(rec)
        # Stream-write after each task so a mid-run crash doesn't lose data
        out_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False, default=str))
        logger.info(
            "  task %s done elapsed=%ss reward=%s",
            rec["task_id"],
            rec["elapsed_s"],
            rec.get("final_reward"),
        )

    overall["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    overall["elapsed_s_total"] = round(time.perf_counter() - t_total, 3)

    # Aggregate
    n_completed = sum(1 for r in overall["tasks"] if r["crashed_with"] is None)
    n_succeeded = sum(1 for r in overall["tasks"] if r.get("succeeded"))
    rewards = [r.get("final_reward") for r in overall["tasks"] if r.get("final_reward") is not None]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    overall["summary"] = {
        "n_tasks": len(tasks),
        "n_completed": n_completed,
        "n_succeeded": n_succeeded,
        "success_rate": (n_succeeded / len(tasks)) if tasks else 0.0,
        "mean_reward": mean_reward,
        "pass_criteria_met": {
            "all_tasks_completed_without_crash": n_completed == len(tasks),
            "got_numeric_aggregate": True,
            "overall": n_completed == len(tasks),
        },
    }

    out_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False, default=str))
    logger.info("wrote %s", out_path)
    logger.info("summary: %s", json.dumps(overall["summary"], indent=2))
    return 0 if overall["summary"]["pass_criteria_met"]["overall"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
