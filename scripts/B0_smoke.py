"""B0 smoke test — 5 mock-OSWorld tasks × N GEPA iterations, end-to-end plumbing.

Success criterion (per refine-logs/EXPERIMENT_PLAN.md §3 B0):
  - All 5 episodes complete (success or fail OK)
  - ≥ 1 schema-valid FCVRPatch lands in BEHAVIORAL_PATCHES
  - CLIP embedder cache initialized
  - No crashes
  - Budget tracker within 20% of expected
  - Log saved to results/B0_smoke.json

NOTES
  - For B0 only, the env loop is mocked (`MockOSWorldAdapter`) because the
    AutoDL host has no KVM/Docker. Every other component (vLLM backbone, CLIP
    embedder, KMeans, MMR, Claude vision reflection, FCVRPatch schema,
    structured-prompt merge) runs for real. See `mock_osworld_adapter.py`.
  - The agent rollout is a *scripted* sequence (not a vLLM-driven agent loop)
    because B0's purpose is FCVR plumbing, not policy quality. vLLM is still
    optionally probed via the `--probe-vllm` flag to confirm the backbone
    serves vision input correctly.

Usage:
  python scripts/B0_smoke.py \\
    --tasks configs/osworld_smoke_5.json \\
    --iterations 5 \\
    --output results/B0_smoke.json
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
from typing import Any

from dotenv import load_dotenv

# Make `visual_gepa` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_gepa.clip_embedder import CLIPImageEmbedder
from visual_gepa.fcvr import DEFAULT_BUDGET, FCVRBudget, FCVROperator
from visual_gepa.mock_osworld_adapter import MockOSWorldAdapter
from visual_gepa.reflection import ClaudeReflectionClient, DEFAULT_REFLECTION_MODEL
from visual_gepa.structured_prompt import StructuredPrompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("B0_smoke")


# --- Default structured prompt (seed P_0) ------------------------------------
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
    "twice; replan instead."
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


# --- vLLM probe (optional sanity check) --------------------------------------
def probe_vllm(endpoint: str, model: str, timeout: float = 30.0) -> dict[str, Any]:
    """Send a tiny chat-completions request to vLLM. Returns a result dict."""
    import urllib.error
    import urllib.request

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with the single word 'OK'."}
            ],
            "max_tokens": 4,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
        dt = time.perf_counter() - t0
        data = json.loads(raw)
        text = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")[:64]
        )
        return {"ok": True, "latency_s": round(dt, 3), "reply_preview": text}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --- One GEPA-style iteration ------------------------------------------------
def run_one_iteration(
    iteration: int,
    adapters: list[MockOSWorldAdapter],
    prompt: StructuredPrompt,
    fcvr: FCVROperator,
) -> dict[str, Any]:
    """One pass: roll out → collect failures → FCVR reflect → emit patches.

    Returns a JSON-serializable iter record.
    """
    rec: dict[str, Any] = {
        "iteration": iteration,
        "tasks": [],
        "n_succeeded": 0,
        "n_failed": 0,
    }
    failed_trajs = []
    for adapter in adapters:
        traj = adapter.run(prompt)
        # metric() return value is logged via traj.final_reward + terminal feedback below;
        # the call here is kept for the side-effect of normalizing trajectories.
        _score, _feedback = adapter.metric(traj)
        rec["tasks"].append(
            {
                "task_id": traj.task_id,
                "n_steps": traj.n_steps,
                "final_reward": float(traj.final_reward),
                "succeeded": traj.succeeded,
                "terminal_feedback": (
                    traj.steps[-1].feedback if traj.steps else "(no steps)"
                ),
            }
        )
        if traj.succeeded:
            rec["n_succeeded"] += 1
        else:
            rec["n_failed"] += 1
            failed_trajs.append(traj)

    # FCVR reflection over failed trajectories.
    patches, fcvr_record = fcvr.run(failed_trajs, parent_prompt=prompt)
    rec["fcvr"] = {
        "n_failed_input": fcvr_record.n_failed_input,
        "n_clusters_used": fcvr_record.n_clusters_used,
        "cluster_sizes": fcvr_record.cluster_sizes,
        "patches": fcvr_record.patches,
        "reflection_stats": fcvr_record.reflection_stats,
        "total_input_tokens": fcvr_record.total_input_tokens,
        "total_output_tokens": fcvr_record.total_output_tokens,
        "total_latency_s": round(fcvr_record.total_latency_s, 3),
        "schema_violations_total": fcvr_record.schema_violations_total,
        "elapsed_s": round(fcvr_record.elapsed_s, 3),
    }

    # Merge new patches into BEHAVIORAL_PATCHES.
    for p in patches:
        prompt.append_patch(scope_guard=p.scope_guard, prompt_diff=p.prompt_diff)
    rec["len_prompt_tokens_after"] = prompt.token_length()
    rec["n_behavioral_patches_after"] = len(prompt.behavioral_patches)
    return rec


# --- Main --------------------------------------------------------------------
def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Visual-GEPA B0 smoke test")
    ap.add_argument("--tasks", required=True, help="path to task list JSON")
    ap.add_argument(
        "--backbone-endpoint",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible vLLM endpoint",
    )
    ap.add_argument(
        "--backbone-model",
        default=os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.5-9B"),
        help="Backbone model name registered with vLLM",
    )
    ap.add_argument(
        "--reflection-model",
        default=os.environ.get("REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL),
        help="Anthropic model id for FCVR reflection",
    )
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=DEFAULT_BUDGET.K)
    ap.add_argument("--J", type=int, default=DEFAULT_BUDGET.J)
    ap.add_argument("--M", type=int, default=DEFAULT_BUDGET.M)
    ap.add_argument("--T_patch", type=int, default=DEFAULT_BUDGET.T_patch)
    ap.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP image encoder for trajectory embedding",
    )
    ap.add_argument("--clip-device", default=None)
    ap.add_argument(
        "--probe-vllm",
        action="store_true",
        help="Send one sanity chat completion to the backbone before running.",
    )
    ap.add_argument("--output", default="results/B0_smoke.json")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load task list -----------------------------------------------------
    tasks_cfg = json.loads(Path(args.tasks).read_text())
    task_ids = [t["id"] for t in tasks_cfg["tasks"]]
    logger.info("loaded %d tasks from %s", len(task_ids), args.tasks)

    # --- Initialize components ---------------------------------------------
    clip = CLIPImageEmbedder(model_name=args.clip_model, device=args.clip_device)
    reflection = ClaudeReflectionClient(
        model=args.reflection_model,
        max_output_tokens=args.T_patch,
    )
    budget = FCVRBudget(K=args.K, J=args.J, M=args.M, T_patch=args.T_patch)
    fcvr = FCVROperator(
        budget=budget,
        clip_embedder=clip,
        reflection_client=reflection,
        rng_seed=args.rng_seed,
    )
    adapters = [
        MockOSWorldAdapter(
            task_id=tid,
            vllm_endpoint=args.backbone_endpoint,
            rng_seed=args.rng_seed,
        )
        for tid in task_ids
    ]
    prompt = build_seed_prompt()

    # --- Optional vLLM probe -----------------------------------------------
    vllm_probe: dict[str, Any] | None = None
    if args.probe_vllm:
        logger.info("probing vLLM endpoint=%s model=%s ...", args.backbone_endpoint, args.backbone_model)
        vllm_probe = probe_vllm(args.backbone_endpoint, args.backbone_model)
        logger.info("vLLM probe: %s", vllm_probe)

    # --- GEPA-style iterations ---------------------------------------------
    overall: dict[str, Any] = {
        "B0_smoke": True,
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "iterations": args.iterations,
            "K": args.K,
            "J": args.J,
            "M": args.M,
            "T_patch": args.T_patch,
            "rng_seed": args.rng_seed,
            "clip_model": args.clip_model,
            "reflection_model": args.reflection_model,
            "backbone_endpoint": args.backbone_endpoint,
            "backbone_model": args.backbone_model,
            "tasks": task_ids,
        },
        "vllm_probe": vllm_probe,
        "iterations": [],
    }
    t_total = time.perf_counter()
    for i in range(args.iterations):
        logger.info("=== iter %d / %d ===", i + 1, args.iterations)
        rec = run_one_iteration(
            iteration=i + 1,
            adapters=adapters,
            prompt=prompt,
            fcvr=fcvr,
        )
        overall["iterations"].append(rec)
        logger.info(
            "iter %d done: n_succeeded=%d/%d patches_total=%d len_tokens=%d",
            i + 1,
            rec["n_succeeded"],
            len(adapters),
            rec["n_behavioral_patches_after"],
            rec["len_prompt_tokens_after"],
        )
    overall["elapsed_s_total"] = round(time.perf_counter() - t_total, 3)
    overall["final_prompt"] = {
        "persona": prompt.persona,
        "global_rules": prompt.global_rules,
        "behavioral_patches": prompt.behavioral_patches,
        "task_scaffold": prompt.task_scaffold,
        "rendered": prompt.render(),
        "n_patches": len(prompt.behavioral_patches),
        "approx_token_length": prompt.token_length(),
    }
    overall["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    # --- Success criterion --------------------------------------------------
    n_iters = len(overall["iterations"])
    total_patches_landed = sum(
        len(it["fcvr"]["patches"]) for it in overall["iterations"]
    )
    final_succ = overall["iterations"][-1]["n_succeeded"] if n_iters else 0
    total_in_tokens = sum(
        it["fcvr"]["total_input_tokens"] for it in overall["iterations"]
    )
    total_out_tokens = sum(
        it["fcvr"]["total_output_tokens"] for it in overall["iterations"]
    )
    # Claude Opus 4.7 vision pricing (2026): $5 / Mtok in, $25 / Mtok out.
    OPUS_PRICE_USD_PER_MTOK_IN = 5.0
    OPUS_PRICE_USD_PER_MTOK_OUT = 25.0
    api_cost_usd = round(
        (total_in_tokens / 1_000_000) * OPUS_PRICE_USD_PER_MTOK_IN
        + (total_out_tokens / 1_000_000) * OPUS_PRICE_USD_PER_MTOK_OUT,
        4,
    )

    # Budget envelope check (B0 spec: "within 20% of expected"). Expected B0
    # budget per EXPERIMENT_PLAN.md §4 M0 ≈ $20 ⇒ ceiling at 1.2 × = $24.
    EXPECTED_USD = 20.0
    BUDGET_CEILING_USD = EXPECTED_USD * 1.2
    budget_within_20pct = api_cost_usd <= BUDGET_CEILING_USD

    # Cache initialization check — CLIP embedder was lazy-loaded if any
    # encode_* call ran; we proxy that by "CLIP produced at least one
    # non-zero embedding". Any FCVR call that completed implies this.
    cache_initialized = total_patches_landed > 0 or any(
        it["fcvr"]["n_clusters_used"] > 0 for it in overall["iterations"]
    )

    all_episodes_completed = all(
        len(it["tasks"]) == len(adapters) for it in overall["iterations"]
    )
    at_least_one_valid_patch_landed = total_patches_landed >= 1

    overall["summary"] = {
        "n_iters_completed": n_iters,
        "total_patches_landed": total_patches_landed,
        "final_n_succeeded": final_succ,
        "final_n_tasks": len(adapters),
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "approx_total_api_cost_usd_at_opus_pricing": api_cost_usd,
        "expected_cost_usd": EXPECTED_USD,
        "budget_ceiling_usd": BUDGET_CEILING_USD,
        "pass_criteria_met": {
            "all_episodes_completed": all_episodes_completed,
            "at_least_one_valid_patch_landed": at_least_one_valid_patch_landed,
            "no_iter_crashed": True,  # any crash would have raised before here
            "cache_initialized": cache_initialized,
            "budget_within_20pct": budget_within_20pct,
        },
    }
    overall["summary"]["pass_criteria_met"]["overall"] = all(
        overall["summary"]["pass_criteria_met"][k]
        for k in (
            "all_episodes_completed",
            "at_least_one_valid_patch_landed",
            "no_iter_crashed",
            "cache_initialized",
            "budget_within_20pct",
        )
    )

    out_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out_path)
    logger.info("summary: %s", json.dumps(overall["summary"], indent=2))
    return 0 if overall["summary"]["pass_criteria_met"]["overall"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
