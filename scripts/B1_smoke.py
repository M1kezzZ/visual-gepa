"""B1 smoke — single real OSWorld task end-to-end (vanilla Qwen3.5-9B, no FCVR).

What this proves:
  - The real `OSWorldAdapter` (Docker provider) boots a desktop_env container
    on this host.
  - vLLM serving Qwen3.5-9B answers vision chat completions over OpenAI API.
  - The agent loop (parse_action → env.step → screenshot) round-trips.
  - `env.evaluate()` returns a numeric reward.

Pass criterion: the script exits 0, the result JSON is written, and at least one
step landed. The actual reward is not the gate here — that's B1 proper. This
is purely the "real OSWorld plumbing works on this host" gate (analog to B0
but with a real env in place of the mock).

Usage (on the 4090 KVM host):
  python scripts/B1_smoke.py \\
      --task third_party/OSWorld/evaluation_examples/examples/libreoffice_calc/<uuid>.json \\
      --backbone-endpoint http://localhost:8000/v1 \\
      --backbone-model Qwen/Qwen3.5-9B \\
      --max-steps 15 \\
      --output results/B1_smoke.json
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
logger = logging.getLogger("B1_smoke")


# --- Seed structured prompt (identical to B0 — single-source-of-truth) -----
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


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Visual-GEPA B1 smoke (real OSWorld, 1 task)")
    ap.add_argument(
        "--task",
        required=True,
        help="Path to OSWorld task config JSON, OR `<domain>/<uuid>` shorthand.",
    )
    ap.add_argument(
        "--backbone-endpoint",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    ap.add_argument(
        "--backbone-model",
        default=os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.5-9B"),
    )
    ap.add_argument("--provider", default="docker", choices=["docker", "vmware", "aws", "aliyun", "azure", "gcp"])
    ap.add_argument("--os-type", default="Ubuntu", choices=["Ubuntu", "Windows"])
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--output", default="results/B1_smoke.json")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve task config
    task_cfg = load_osworld_task_config(args.task)
    logger.info(
        "loaded OSWorld task id=%s domain=%s instruction=%r",
        task_cfg.get("id"),
        (task_cfg.get("related_apps") or ["?"])[0],
        task_cfg.get("instruction", "")[:120],
    )

    adapter = OSWorldAdapter(
        task_dict=task_cfg,
        vllm_endpoint=args.backbone_endpoint,
        vllm_model=args.backbone_model,
        provider_name=args.provider,
        os_type=args.os_type,
        max_steps=args.max_steps,
        headless=True,
    )

    prompt = build_seed_prompt()
    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    t0 = time.perf_counter()

    crashed_with: str | None = None
    try:
        traj = adapter.run(prompt)
    except Exception as e:  # noqa: BLE001 — record + write the failure
        logger.exception("adapter.run crashed")
        crashed_with = f"{type(e).__name__}: {e}"
        traj = None

    elapsed_s = round(time.perf_counter() - t0, 3)
    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    summary = {
        "B1_smoke": True,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": elapsed_s,
        "config": {
            "task_id": task_cfg.get("id"),
            "task_instruction": task_cfg.get("instruction"),
            "backbone_endpoint": args.backbone_endpoint,
            "backbone_model": args.backbone_model,
            "provider": args.provider,
            "os_type": args.os_type,
            "max_steps": args.max_steps,
        },
        "crashed_with": crashed_with,
    }

    if traj is not None:
        score, feedback_str = adapter.metric(traj)
        summary["trajectory"] = {
            "n_steps": traj.n_steps,
            "final_reward": float(traj.final_reward),
            "succeeded": traj.succeeded,
            "score": score,
            "feedback": feedback_str,
            "steps": [
                {
                    "i": i,
                    "action": (s.action or "")[:512],
                    "reward": s.reward,
                    "feedback": (s.feedback or "")[:512],
                    "axtree_len": len(s.accessibility_tree or ""),
                }
                for i, s in enumerate(traj.steps)
            ],
        }
        summary["pass_criteria_met"] = {
            "ran_at_least_one_step": traj.n_steps >= 1,
            "no_crash": True,
            "got_numeric_reward": isinstance(traj.final_reward, (int, float)),
        }
    else:
        summary["trajectory"] = None
        summary["pass_criteria_met"] = {
            "ran_at_least_one_step": False,
            "no_crash": False,
            "got_numeric_reward": False,
        }

    summary["pass_criteria_met"]["overall"] = all(summary["pass_criteria_met"].values())

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out_path)
    logger.info("summary: %s", json.dumps(summary["pass_criteria_met"], indent=2))
    return 0 if summary["pass_criteria_met"]["overall"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
