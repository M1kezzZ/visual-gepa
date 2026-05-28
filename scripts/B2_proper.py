"""B2 proper — multi-iter GEPA-lite loop with FCVR proposer.

Pipeline per seed:
  Phase A (vanilla):      SEED prompt on N_tasks → baseline trajectories
  Phase G (GEPA loop, max_iters):
    iter k:
      1. failed_k = failures from current_parent's most-recent full eval
      2. If |failed_k| < min_failures_for_fcvr: skip iter (no patches to propose)
      3. patches_k = FCVR(failed_k, parent_prompt)
      4. For each patch p_i in patches_k:
           child_i = parent_prompt + p_i (appended to BEHAVIORAL_PATCHES)
           score_i = evaluate(child_i, minibatch_k)  # minibatch random subset
      5. best_child = argmax_i score_i ; if better than parent on minibatch:
           promote best_child → new parent
           re-eval new parent on full N_tasks (for next iter's failures)
      6. Else: keep parent (this iter didn't help)
  Phase F (final):        evaluate current parent on full N_tasks → headline

Multi-seed orchestration: run this script 3 times with different --rng-seed
(42, 43, 44). Each produces a self-contained result JSON + manifest. An
external aggregator (`analyze_b2.py results/B2_proper_seed*.json`)
combines them with bootstrap CIs.

Why GEPA-lite instead of full Pareto frontier (as in the original GEPA
paper): for B2 proper v1, we just want to test whether iterated FCVR
reflection IMPROVES a single candidate at all. Full Pareto frontier
(multiple co-evolving candidates, per-instance selection) is a B3 thing.

Headline paper metrics computed in `paper_metrics`:
  - phase A success rate / mean reward / early-stop rate    (= vanilla)
  - phase F success rate / mean reward / early-stop rate    (= post-GEPA)
  - loop_escape_count   (tasks where A early-stopped but F did not)
  - per-iter trajectory of improvements (catches "FCVR plateaus at iter k")

Cost model (default args: 25 tasks × 8 iters × minibatch=5 × K=4 patches):
  - Phase A: 25 rollouts
  - Each iter: 4 children × 5 minibatch = 20 rollouts + 1 FCVR Claude call
    (+ optional re-eval of new parent on 25 = 25 rollouts on accept)
  - Phase F: 25 rollouts
  Worst-case (every iter accepts): 25 + 8×(20+25) + 25 = 410 rollouts/seed
  Best-case (no iter accepts):     25 + 8×20      + 25 = 210 rollouts/seed
  At ~2 min/rollout (with early-stop on 5090): 7-14 hours per seed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_gepa.clip_embedder import CLIPImageEmbedder
from visual_gepa.fcvr import DEFAULT_BUDGET, FCVRBudget, FCVROperator
from visual_gepa.manifest import Manifest
from visual_gepa.osworld_adapter import OSWorldAdapter, load_osworld_task_config
from visual_gepa.reflection import ClaudeReflectionClient, DEFAULT_REFLECTION_MODEL
from visual_gepa.structured_prompt import StructuredPrompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("B2_proper")


# Shared seed prompt with B2_mini for direct comparability.
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


def n_distinct(seq: list[str]) -> int:
    return len(set(seq))


def early_stop_fraction(records: list) -> float:
    if not records:
        return 0.0
    n = sum(
        1 for r in records
        if (r.get("early_stop_reason") or "").startswith("repeated_actions_")
    )
    return n / len(records)


def evaluate_candidate(
    label: str,
    prompt: StructuredPrompt,
    tasks: list[dict],
    args,
    stream_out_path: Path | None = None,
    halt_on_task1_crash: bool = False,
) -> list[dict]:
    """Roll out the candidate on every task in `tasks`. Stream-write partial
    results to `stream_out_path` for resumability. Returns a list of records
    (same shape as B2_mini's run_phase records).
    """
    records: list[dict] = []
    for i, task_entry in enumerate(tasks):
        task_id = task_entry.get("id", "<unknown>")
        cfg_path = task_entry.get("task_config_path") or task_entry.get("id")
        try:
            task_cfg = load_osworld_task_config(cfg_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not load %s: %s", cfg_path, e)
            records.append({
                "task_id": task_id, "task_config_path": str(cfg_path),
                "phase": label, "elapsed_s": 0.0,
                "crashed_with": f"load_config: {e}",
                "n_steps": 0, "final_reward": None, "succeeded": False,
                "score": None, "feedback": "(config load failed)",
                "actions": [], "raw_model_texts": [], "n_distinct_actions": 0,
                "early_stop_reason": None, "reward_source": "config_load_failed",
                "_traj": None,
            })
            continue
        logger.info("--- %s task %d/%d : %s ---", label, i + 1, len(tasks), task_id)
        adapter = OSWorldAdapter(
            task_dict=task_cfg,
            vllm_endpoint=args.backbone_endpoint,
            vllm_model=args.backbone_model,
            provider_name=args.provider,
            os_type=args.os_type,
            max_steps=args.max_steps,
            headless=True,
            cache_dir=args.cache_dir,
            early_stop_on_repeated_actions=args.early_stop_on_repeated,
            step_watchdog_seconds=args.step_watchdog,
            reset_watchdog_seconds=args.reset_watchdog,
            evaluate_watchdog_seconds=args.evaluate_watchdog,
            # Backbone routing
            backbone_kind=args.backbone_kind,
            openai_endpoint=args.openai_endpoint,
            openai_api_key=args.openai_api_key,
            openai_model=args.openai_model,
            agent_max_trajectory_length=args.agent_max_trajectory_length,
            agent_temperature=args.agent_temperature,
            agent_top_p=args.agent_top_p,
            agent_max_tokens=args.agent_max_tokens,
            agent_image_detail=args.agent_image_detail or None,
        )
        t0 = time.perf_counter()
        crashed = None
        traj = None
        try:
            traj = adapter.run(prompt)
        except Exception as e:  # noqa: BLE001
            logger.exception("%s task %s crashed", label, task_id)
            crashed = f"{type(e).__name__}: {e}"
        rec: dict = {
            "task_id": task_id, "task_config_path": str(cfg_path),
            "instruction": task_cfg.get("instruction"),
            "phase": label,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "crashed_with": crashed,
        }
        if traj is not None:
            score, feedback = adapter.metric(traj)
            actions = [(s.action or "")[:200] for s in traj.steps]
            raws = [(s.raw_model_text or "")[:2000] for s in traj.steps]
            rec.update({
                "n_steps": traj.n_steps,
                "final_reward": float(traj.final_reward),
                "succeeded": traj.succeeded,
                "score": score, "feedback": feedback,
                "actions": actions, "raw_model_texts": raws,
                "n_distinct_actions": n_distinct(actions),
                "early_stop_reason": getattr(traj, "early_stop_reason", None),
                "reward_source": getattr(traj, "reward_source", "unset"),
                # Backbone token usage (codex stop-time review 2026-05-28 fix)
                "backbone_prompt_tokens": getattr(traj, "backbone_prompt_tokens", 0),
                "backbone_completion_tokens": getattr(traj, "backbone_completion_tokens", 0),
                "backbone_reasoning_tokens": getattr(traj, "backbone_reasoning_tokens", 0),
                "_traj": traj,
            })
        else:
            rec.update({
                "n_steps": 0, "final_reward": None, "succeeded": False,
                "score": None, "feedback": "(crashed)", "actions": [],
                "raw_model_texts": [], "n_distinct_actions": 0,
                "early_stop_reason": None, "reward_source": "crashed_before_eval",
                "backbone_prompt_tokens": 0,
                "backbone_completion_tokens": 0,
                "backbone_reasoning_tokens": 0,
                "_traj": None,
            })
        records.append(rec)
        if stream_out_path is not None:
            writable = [{k: v for k, v in r.items() if k != "_traj"} for r in records]
            stream_out_path.write_text(json.dumps({"phase": label, "tasks": writable}, indent=2, default=str))
        logger.info(
            "  %s task %s done elapsed=%ss reward=%s distinct=%d estop=%s",
            label, task_id, rec["elapsed_s"], rec.get("final_reward"),
            rec.get("n_distinct_actions"), rec.get("early_stop_reason"),
        )
        if i == 0 and halt_on_task1_crash and crashed:
            logger.error("🚨 %s task 1 crashed: %s — HALTING per fail-fast", label, crashed)
            raise SystemExit(2)
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n_tasks": 0, "n_completed": 0, "n_succeeded": 0, "success_rate": 0.0,
                "mean_reward": 0.0, "mean_distinct_actions": 0.0, "early_stop_rate": 0.0}
    n_comp = sum(1 for r in records if r["crashed_with"] is None)
    n_succ = sum(1 for r in records if r.get("succeeded"))
    rewards = [r["final_reward"] for r in records if r.get("final_reward") is not None]
    distincts = [r["n_distinct_actions"] for r in records]
    return {
        "n_tasks": len(records),
        "n_completed": n_comp,
        "n_succeeded": n_succ,
        "success_rate": n_succ / len(records),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "mean_distinct_actions": sum(distincts) / len(distincts) if distincts else 0.0,
        "early_stop_rate": early_stop_fraction(records),
    }


def score_candidate(records: list[dict]) -> tuple[float, float, float]:
    """Sortable tuple: (mean_reward, success_rate, 1 - early_stop_rate).
    Higher is better; ties broken first by reward, then success, then
    by lower early-stop rate (proxy for "agent at least tried different
    things instead of click-looping").

    NOTE: this is used to RANK children for picking the best, NOT for the
    promotion decision. The promotion guard (`should_promote`) is stricter
    — see codex review (2026-05-27) Q1: tuple-cmp alone degenerates to
    early-stop-rate when rewards are mostly 0, producing noisy promotions.
    """
    s = summarize(records)
    return (s["mean_reward"], s["success_rate"], 1.0 - s["early_stop_rate"])


def should_promote(
    parent_records: list[dict],
    child_records: list[dict],
) -> tuple[bool, str]:
    """Promotion guard.

    Accept the child IF any of:
      1. Strict mean reward improvement
      2. Strict success-count improvement
      3. Early-stop reduction by >= max(2, ceil(0.15 * n)) tasks on minibatch
         AND no reward/success regression

    The ES threshold scales with minibatch size (codex dry-smoke audit
    2026-05-27 Q1): "≥2 fewer ES" was right at n=5 (40pp threshold) but at
    n=25 that's only 8pp and would promote noise.

    Returns (accept_bool, reason_string).
    """
    import math
    p_sum = summarize(parent_records)
    c_sum = summarize(child_records)
    n = max(len(parent_records), len(child_records))
    es_threshold = max(2, math.ceil(0.15 * n))
    if c_sum["mean_reward"] > p_sum["mean_reward"]:
        return True, f"reward_better ({p_sum['mean_reward']:.3f} → {c_sum['mean_reward']:.3f})"
    if c_sum["n_succeeded"] > p_sum["n_succeeded"]:
        return True, f"success_better ({p_sum['n_succeeded']} → {c_sum['n_succeeded']})"
    p_es = int(round(p_sum["early_stop_rate"] * len(parent_records)))
    c_es = int(round(c_sum["early_stop_rate"] * len(child_records)))
    if (
        (p_es - c_es) >= es_threshold
        and c_sum["mean_reward"] >= p_sum["mean_reward"]
        and c_sum["n_succeeded"] >= p_sum["n_succeeded"]
    ):
        return True, (
            f"early_stop_better (ΔES {p_es}→{c_es}={p_es - c_es}≥{es_threshold}, "
            f"reward {p_sum['mean_reward']:.3f}≥{c_sum['mean_reward']:.3f}, "
            f"succ {p_sum['n_succeeded']}≥{c_sum['n_succeeded']})"
        )
    return False, (
        f"reject (n={n}, ES_threshold={es_threshold}; parent: r={p_sum['mean_reward']:.3f}, "
        f"succ={p_sum['n_succeeded']}, ES={p_es}; child: r={c_sum['mean_reward']:.3f}, "
        f"succ={c_sum['n_succeeded']}, ES={c_es})"
    )


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="B2 proper — multi-iter GEPA-lite with FCVR")
    ap.add_argument("--tasks", required=True, help="Task set JSON (e.g. configs/osworld_b2_25.json)")
    # Backbone: choose vllm (self-hosted) OR openai_api (anyaigc / direct)
    ap.add_argument("--backbone-kind", default="openai_api", choices=["vllm", "openai_api"],
                    help="vllm = self-hosted Qwen via vLLM (legacy); "
                         "openai_api = anyaigc / OpenAI / Anthropic OpenAI-compat (new default 2026-05-28)")
    # vLLM args (legacy path)
    ap.add_argument("--backbone-endpoint", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--backbone-model", default=os.environ.get("VLLM_MODEL_NAME", "/root/models/Qwen3.5-9B"))
    # openai_api args (new path)
    ap.add_argument("--openai-endpoint", default=os.environ.get("OPENAI_BASE_URL", "https://anyaigc.com/v1"),
                    help="OpenAI-compatible base URL (anyaigc.com/v1 default)")
    ap.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"),
                    help="API key for the openai_api backbone (env OPENAI_API_KEY by default)")
    ap.add_argument("--openai-model", default="gpt-5.5",
                    help="Backbone model id (gpt-5.5 default, holds 78.7pct on OSWorld-Verified May 2026)")
    ap.add_argument("--agent-max-trajectory-length", type=int, default=3,
                    help="OSWorld official default 3 (max history pairs kept)")
    ap.add_argument("--agent-temperature", type=float, default=1.0,
                    help="OSWorld official default 1.0")
    ap.add_argument("--agent-top-p", type=float, default=0.9,
                    help="OSWorld official default 0.9")
    ap.add_argument("--agent-max-tokens", type=int, default=1500,
                    help="OSWorld official default 1500 (completion budget)")
    ap.add_argument("--agent-image-detail", default="", choices=["", "low", "high"],
                    help="OpenAI vision detail; empty lets provider decide")
    ap.add_argument("--provider", default="docker")
    ap.add_argument("--os-type", default="Ubuntu")
    # max_steps: changed default 15 → 50 on 2026-05-28. OSWorld original
    # benchmark used 15 but OSWorld-Verified leaderboard models routinely run
    # at 50 (o3 saw 9.1% → 23% jump with bigger budget). early_stop=5 keeps
    # click-loop tasks bounded so 50 is just the ceiling, not the typical run.
    ap.add_argument("--max-steps", type=int, default=50,
                    help="Max steps per task; 50 is the modern OSWorld-Verified convention")
    ap.add_argument("--rng-seed", type=int, required=True)
    ap.add_argument("--cache-dir", default=None)
    # early_stop_on_repeated_actions: raised 3 → 5 on 2026-05-28. N=3 was too
    # aggressive and cut effective trajectory to ~5 steps; N=5 keeps the
    # click-loop guard while giving the agent room to recover.
    ap.add_argument("--early-stop-on-repeated", type=int, default=5)
    ap.add_argument("--step-watchdog", type=int, default=180)
    ap.add_argument("--reset-watchdog", type=int, default=600)
    ap.add_argument("--evaluate-watchdog", type=int, default=60)
    ap.add_argument("--reflection-model", default=os.environ.get("REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL))
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip-device", default=None)
    # K: lowered 4 → 2 on 2026-05-28. B2 proper full found cluster_interpretable
    # = False across all 24 calls (silhouette ≤ 0.29); failures on OSWorld are
    # too homogeneous in CLIP space for K=4. K=2 makes it explicit.
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--J", type=int, default=DEFAULT_BUDGET.J)
    ap.add_argument("--M", type=int, default=DEFAULT_BUDGET.M)
    # T_patch: raised 512 → 1024 on 2026-05-28. Generic 100-tok-per-field patches
    # weren't specific enough; OSWorld backbone uses 1500 max_tokens.
    ap.add_argument("--T_patch", type=int, default=1024)
    ap.add_argument("--max-iters", type=int, default=8)
    # minibatch: raised 5 → 10 on 2026-05-28. B2 proper full's only accept
    # (seed=42 iter 1) was minibatch selection bias (88% → 96% ES regression
    # on full 25). minibatch=10 halves variance; minibatch=25 (full) is the
    # ideal but costs 5× more.
    ap.add_argument("--minibatch-size", type=int, default=10)
    ap.add_argument("--min-failures-for-fcvr", type=int, default=3)
    ap.add_argument("--accept-epsilon", type=float, default=0.0,
                    help="child must beat parent on minibatch by > epsilon (default 0.0 = strict gt)")
    # Backbone cost rates (codex stop-time review 2026-05-28 fix). Estimate
    # only — anyaigc doesn't publish per-model rates, so we use OpenAI list
    # prices as a reference. Override if you have better numbers.
    ap.add_argument("--backbone-input-cost-per-mtok", type=float, default=1.25,
                    help="USD per 1M backbone INPUT tokens (gpt-5.5 OpenAI list est. 1.25)")
    ap.add_argument("--backbone-output-cost-per-mtok", type=float, default=10.0,
                    help="USD per 1M backbone OUTPUT+reasoning tokens (gpt-5.5 OpenAI list est. 10)")
    ap.add_argument("--reflection-input-cost-per-mtok", type=float, default=5.0,
                    help="USD per 1M reflection INPUT tokens (claude-opus-4-7 list 5)")
    ap.add_argument("--reflection-output-cost-per-mtok", type=float, default=25.0,
                    help="USD per 1M reflection OUTPUT tokens (claude-opus-4-7 list 25)")
    # SwanLab experiment tracking (added 2026-05-29). Auto-logs scalar metrics +
    # GPU/CPU/RAM (Nvidia / MetaX / Ascend / etc.). Opt-out via --swanlab-disabled.
    ap.add_argument("--swanlab-project", default="visual-gepa",
                    help="SwanLab project name (default: visual-gepa)")
    ap.add_argument("--swanlab-experiment", default=None,
                    help="SwanLab experiment name (default: derived from --output filename)")
    ap.add_argument("--swanlab-mode", default="cloud", choices=["cloud", "offline", "local", "disabled"],
                    help="SwanLab logging mode. 'cloud'=sync to swanlab.cn; 'disabled'=off")
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    if args.backbone_kind == "openai_api" and not args.openai_api_key:
        raise SystemExit(
            "ERROR: --openai-api-key not set (or env OPENAI_API_KEY missing). "
            "Required for backbone_kind=openai_api."
        )

    # --- SwanLab init (codex stop-time hardening 2026-05-29) ----------------
    # Wrap import + init in try/except so an absent swanlab pkg or bad key
    # doesn't kill the experiment. We log via global swanlab.log() throughout.
    #
    # SECURITY: vars(args) contains literal API keys (--openai-api-key parses
    # the secret string into args.openai_api_key). swanlab uploads config to
    # cloud, so the raw dict would leak credentials. Redact before passing.
    # Codex stop-time review (2026-05-29) caught this leak. The redaction set
    # covers the args we KNOW are sensitive; future args matching the
    # SENSITIVE_KEY_PATTERNS regex (`*_key`, `*_token`, `*_secret`, `*password*`)
    # are also redacted defensively.
    SENSITIVE_ARG_NAMES = {"openai_api_key", "anthropic_api_key", "swanlab_api_key"}
    SENSITIVE_KEY_PATTERNS = re.compile(r"(_key|_token|_secret|password)$", re.IGNORECASE)

    def _redact_config(args_dict: dict) -> dict:
        out = {}
        for k, v in args_dict.items():
            if k in SENSITIVE_ARG_NAMES or SENSITIVE_KEY_PATTERNS.search(k):
                # Replace with a placeholder that records (a) the field exists,
                # (b) whether it had a value, (c) a fingerprint hint (length).
                if v:
                    s = str(v)
                    out[k] = f"<redacted-{len(s)}c>"
                else:
                    out[k] = None
            else:
                out[k] = v
        return out

    swanlab_active = False
    if args.swanlab_mode != "disabled":
        try:
            import swanlab
            exp_name = args.swanlab_experiment or Path(args.output).stem
            sl_key = os.environ.get("SWANLAB_API_KEY")
            if args.swanlab_mode == "cloud" and not sl_key:
                logger.warning("SWANLAB_API_KEY not set; falling back to swanlab-mode=offline")
                args.swanlab_mode = "offline"
            safe_config = _redact_config(vars(args))
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=exp_name,
                mode=args.swanlab_mode,
                config=safe_config,
                description=(
                    f"B2 proper {exp_name} | backbone={args.backbone_kind}:"
                    f"{args.openai_model if args.backbone_kind == 'openai_api' else args.backbone_model} | "
                    f"seed={args.rng_seed} | max_iters={args.max_iters} K={args.K} "
                    f"max_steps={args.max_steps} minibatch={args.minibatch_size}"
                ),
            )
            swanlab_active = True
            logger.info("swanlab initialized: project=%s mode=%s exp=%s (config redacted: %s)",
                        args.swanlab_project, args.swanlab_mode, exp_name,
                        sorted(k for k, v in safe_config.items() if isinstance(v, str) and v.startswith("<redacted")))
        except ImportError:
            logger.warning("swanlab package not installed — skipping experiment tracking")
        except Exception as e:  # noqa: BLE001
            logger.warning("swanlab init failed (%s) — continuing without tracking", e)

    def sl_log(d: dict, step: int | None = None) -> None:
        """Log dict to swanlab if active. No-op otherwise."""
        if not swanlab_active:
            return
        try:
            if step is not None:
                swanlab.log(d, step=step)
            else:
                swanlab.log(d)
        except Exception as e:  # noqa: BLE001
            logger.warning("swanlab.log failed: %s", e)

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_name(out_path.stem + "_manifest.json")
    rng = random.Random(args.rng_seed)

    # Provenance manifest start
    m = Manifest(
        experiment_id=f"B2_proper_seed{args.rng_seed}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        block="B2",
    )
    if args.backbone_kind == "openai_api":
        backbone_cmd = f"openai_api endpoint={args.openai_endpoint} model={args.openai_model}"
        model_path_for_manifest = args.openai_model
    else:
        backbone_cmd = f"endpoint={args.backbone_endpoint} model={args.backbone_model}"
        model_path_for_manifest = args.backbone_model
    m.start(
        vllm_cmd=backbone_cmd,
        model_path=model_path_for_manifest,
        qcow2_path=(str(Path(args.cache_dir) / "docker_vm_data" / "Ubuntu.qcow2") if args.cache_dir else ""),
        config_path=args.tasks,
        seed=args.rng_seed,
        compute_model_md5=False,
        compute_qcow2_md5=False,
    )

    tasks_cfg = json.loads(Path(args.tasks).read_text())
    tasks = tasks_cfg.get("tasks", [])
    logger.info(
        "loaded %d tasks from %s | seed=%d | max_iters=%d minibatch=%d K=%d",
        len(tasks), args.tasks, args.rng_seed, args.max_iters, args.minibatch_size, args.K,
    )

    clip = CLIPImageEmbedder(model_name=args.clip_model, device=args.clip_device)
    reflection = ClaudeReflectionClient(model=args.reflection_model, max_output_tokens=args.T_patch)
    budget = FCVRBudget(K=args.K, J=args.J, M=args.M, T_patch=args.T_patch)
    # FCVR rng_seed is reset per-iter inside the loop (codex Q3 hygiene fix).
    fcvr = FCVROperator(budget=budget, clip_embedder=clip, reflection_client=reflection, rng_seed=args.rng_seed)

    started = datetime.datetime.utcnow().isoformat() + "Z"
    t_total = time.perf_counter()

    # Cost-accounting accumulator (codex stop-time review 2026-05-28 fix).
    # Every evaluate_candidate() output gets appended; backbone tokens
    # summed at the end.
    all_eval_records_for_cost: list[list[dict]] = []

    # --- PHASE A (vanilla baseline) -----------------------------------------
    logger.info("=== PHASE A: vanilla baseline on %d tasks ===", len(tasks))
    parent_prompt = build_seed_prompt()
    phaseA_path = out_path.with_name(out_path.stem + "_phaseA.json")
    A_records = evaluate_candidate("A", parent_prompt, tasks, args, phaseA_path, halt_on_task1_crash=True)
    all_eval_records_for_cost.append(A_records)
    parent_full_records = A_records
    parent_score = score_candidate(parent_full_records)
    logger.info(
        "Phase A baseline: succ=%d/%d reward=%.3f early_stop_rate=%.0f%%",
        sum(r.get("succeeded", False) for r in A_records), len(A_records),
        parent_score[0], 100 * (1 - parent_score[2]),
    )
    # Log phase A summary to swanlab (codex stop-time fix 2026-05-29)
    _a_sum = summarize(A_records)
    sl_log({
        "phaseA/success_rate": _a_sum["success_rate"],
        "phaseA/mean_reward": _a_sum["mean_reward"],
        "phaseA/early_stop_rate": _a_sum["early_stop_rate"],
        "phaseA/mean_distinct_actions": _a_sum["mean_distinct_actions"],
        "phaseA/n_completed": _a_sum["n_completed"],
        "phaseA/n_succeeded": _a_sum["n_succeeded"],
    }, step=0)

    # --- PHASE G (GEPA-lite loop) ------------------------------------------
    iter_history: list[dict] = []
    accepted_children = 0
    total_fcvr_input_tokens = 0
    total_fcvr_output_tokens = 0
    iter_total_cost = 0.0
    # Codex Q2 fix (dry-smoke 2026-05-27): rejected-cluster memory. When
    # parent doesn't change after a reject, iter k+1 sees the same failures
    # → KMeans picks the same clusters → reflection makes the same patches
    # → same reject. Track which cluster member-sets have been tried and
    # skip subsequent iters' patches whose cluster member-set is contained
    # in any previously-rejected member-set.
    rejected_member_sets: list[frozenset[str]] = []
    # Codex Q4 fix: best_child_loop_escape — track the union across iters
    # of (task_id where vanilla A early-stopped) AND (any child this iter
    # did NOT early-stop). This captures FCVR's reachability even when no
    # iter promotes the child to parent.
    a_loop_task_ids = {
        r["task_id"] for r in A_records
        if (r.get("early_stop_reason") or "").startswith("repeated_actions_")
    }
    best_child_loop_escape_set: set[str] = set()

    for iter_k in range(1, args.max_iters + 1):
        logger.info("=== PHASE G iter %d/%d ===", iter_k, args.max_iters)
        failed_trajs = [r["_traj"] for r in parent_full_records if r.get("_traj") and not r["_traj"].succeeded]
        iter_rec: dict = {
            "iter": iter_k, "n_failures_input": len(failed_trajs),
            "parent_score_before": list(parent_score),
            "decision": "pending",
        }
        if len(failed_trajs) < args.min_failures_for_fcvr:
            iter_rec["decision"] = f"skip_too_few_failures (need>={args.min_failures_for_fcvr})"
            iter_history.append(iter_rec)
            logger.info("  iter %d: too few failures (%d) — skip", iter_k, len(failed_trajs))
            continue
        # Codex Q3 hygiene fix: per-iter FCVR rng so KMeans differs across iters
        # even when the failure set is unchanged. Without this, iter k+1 with
        # same failures + same seed produces identical clusters.
        fcvr.rng_seed = args.rng_seed + iter_k
        patches, fcvr_record = fcvr.run(failed_trajs, parent_prompt=parent_prompt)
        # Codex Q2 fix: filter out patches whose cluster member-set is contained
        # in any previously-rejected cluster (these would just retry a failed
        # mutation). Keep patches whose cluster has at least one NEW member.
        filtered_patches: list = []
        filtered_member_sets: list[frozenset[str]] = []
        cluster_member_sets = [frozenset(s) for s in (fcvr_record.cluster_member_task_ids or [])]
        skipped_count = 0
        for i, p in enumerate(patches):
            mset = cluster_member_sets[i] if i < len(cluster_member_sets) else frozenset()
            if any(mset and mset <= rej for rej in rejected_member_sets):
                skipped_count += 1
                continue
            filtered_patches.append(p)
            filtered_member_sets.append(mset)
        iter_rec.update({
            "n_patches_proposed": len(patches),
            "n_patches_after_dedup": len(filtered_patches),
            "n_patches_skipped_as_rejected_dup": skipped_count,
            "fcvr_silhouette": fcvr_record.silhouette_score if fcvr_record else None,
            "cluster_sizes": fcvr_record.cluster_sizes if fcvr_record else [],
            "cluster_interpretable": fcvr_record.cluster_interpretable if fcvr_record else False,
            "fcvr_input_tokens": fcvr_record.total_input_tokens if fcvr_record else 0,
            "fcvr_output_tokens": fcvr_record.total_output_tokens if fcvr_record else 0,
            "fcvr_rng_seed_used": fcvr.rng_seed,
        })
        # Backward-compat key (older analyzers read "n_patches"):
        iter_rec["n_patches"] = len(filtered_patches)
        if fcvr_record:
            total_fcvr_input_tokens += fcvr_record.total_input_tokens
            total_fcvr_output_tokens += fcvr_record.total_output_tokens
        if not filtered_patches:
            iter_rec["decision"] = (
                "skip_no_new_patches"
                if skipped_count > 0 else "skip_no_patches"
            )
            iter_history.append(iter_rec)
            logger.info(
                "  iter %d: no usable patches (proposed=%d, dedup-skipped=%d) — skip",
                iter_k, len(patches), skipped_count,
            )
            continue
        # Reassign for the rest of the loop
        patches = filtered_patches
        # Spawn K children = parent + each patch, eval on a random minibatch
        minibatch_idx = sorted(rng.sample(range(len(tasks)), min(args.minibatch_size, len(tasks))))
        minibatch_tasks = [tasks[i] for i in minibatch_idx]
        # Codex Q2 fix (2026-05-27): re-eval parent FRESH on this minibatch.
        # Reusing cached records from prior full-eval suffers from optimizer's
        # curse: K freshly-sampled children max-selected against a stale
        # parent estimate produces upward-biased acceptance. Cost: +5 rollouts/
        # iter/seed. Cheaper than false promotions.
        parent_stream = out_path.with_name(out_path.stem + f"_iter{iter_k}_parent.json")
        parent_minibatch_records = evaluate_candidate(
            f"G{iter_k}.P", parent_prompt, minibatch_tasks, args, parent_stream,
        )
        all_eval_records_for_cost.append(parent_minibatch_records)
        parent_minibatch_score = score_candidate(parent_minibatch_records)
        iter_rec["parent_minibatch_score"] = list(parent_minibatch_score)
        iter_rec["parent_minibatch_summary"] = summarize(parent_minibatch_records)
        iter_rec["minibatch_task_ids"] = [t["id"] for t in minibatch_tasks]
        children_records: list[dict] = []
        children_full_records: list[list[dict]] = []  # raw lists for should_promote
        for i, p in enumerate(patches):
            child_prompt = StructuredPrompt(
                persona=parent_prompt.persona,
                global_rules=parent_prompt.global_rules,
                behavioral_patches=list(parent_prompt.behavioral_patches),
                task_scaffold=parent_prompt.task_scaffold,
            )
            child_prompt.append_patch(scope_guard=p.scope_guard, prompt_diff=p.prompt_diff)
            child_label = f"G{iter_k}.C{i}"
            child_stream = out_path.with_name(out_path.stem + f"_iter{iter_k}_child{i}.json")
            child_recs = evaluate_candidate(child_label, child_prompt, minibatch_tasks, args, child_stream)
            all_eval_records_for_cost.append(child_recs)
            children_full_records.append(child_recs)
            child_score = score_candidate(child_recs)
            # Codex Q4: best_child_loop_escape — for each minibatch task where
            # vanilla A early-stopped, did THIS child fail to early-stop?
            for cr in child_recs:
                tid = cr["task_id"]
                if tid in a_loop_task_ids and not (cr.get("early_stop_reason") or "").startswith("repeated_actions_"):
                    best_child_loop_escape_set.add(tid)
            children_records.append({
                "i": i, "patch_failure_pattern": p.failure_pattern[:200],
                "score": list(child_score),
                "summary": summarize(child_recs),
            })
        iter_rec["children"] = children_records
        # Pick best child by (mean_reward, success_rate, 1-early_stop_rate)
        best_idx = max(range(len(children_records)), key=lambda i: tuple(children_records[i]["score"]))
        best_child_score = tuple(children_records[best_idx]["score"])
        iter_rec["best_child_idx"] = best_idx
        iter_rec["best_child_score"] = list(best_child_score)
        # Codex Q1 fix (2026-05-27): promotion guard. Tuple-cmp alone is too
        # lax when reward~0; require strict reward/success improvement OR
        # early-stop drop by ≥2 tasks WITH no reward regression.
        accept, reason = should_promote(parent_minibatch_records, children_full_records[best_idx])
        iter_rec["promotion_decision_reason"] = reason
        if accept:
            accepted_children += 1
            best_patch = patches[best_idx]
            parent_prompt = StructuredPrompt(
                persona=parent_prompt.persona,
                global_rules=parent_prompt.global_rules,
                behavioral_patches=list(parent_prompt.behavioral_patches),
                task_scaffold=parent_prompt.task_scaffold,
            )
            parent_prompt.append_patch(scope_guard=best_patch.scope_guard, prompt_diff=best_patch.prompt_diff)
            iter_rec["decision"] = "accept_child"
            iter_rec["accepted_patch_failure_pattern"] = best_patch.failure_pattern[:200]
            logger.info("  iter %d: ACCEPT child %d (%s) — re-evaluating new parent on full %d tasks",
                        iter_k, best_idx, reason, len(tasks))
            full_stream = out_path.with_name(out_path.stem + f"_iter{iter_k}_full.json")
            parent_full_records = evaluate_candidate(f"G{iter_k}.FULL", parent_prompt, tasks, args, full_stream)
            all_eval_records_for_cost.append(parent_full_records)
            parent_score = score_candidate(parent_full_records)
            iter_rec["new_parent_full_score"] = list(parent_score)
            iter_rec["new_parent_n_patches"] = len(parent_prompt.behavioral_patches)
        else:
            iter_rec["decision"] = "reject_no_child_better"
            logger.info("  iter %d: REJECT — %s", iter_k, reason)
            # Codex Q2: remember which cluster member-sets we just rejected,
            # so iter k+1 doesn't propose patches over the same members again.
            for mset in filtered_member_sets:
                if mset:
                    rejected_member_sets.append(mset)
        iter_history.append(iter_rec)
        # Log iter summary to swanlab (codex stop-time fix 2026-05-29).
        # step=iter_k so plots show per-iter trajectory cleanly.
        _decision = iter_rec.get("decision", "")
        sl_log({
            f"iter/n_failures_input": iter_rec.get("n_failures_input", 0),
            f"iter/n_patches_proposed": iter_rec.get("n_patches_proposed", 0),
            f"iter/n_patches_after_dedup": iter_rec.get("n_patches_after_dedup", 0),
            f"iter/n_patches_skipped_as_rejected_dup": iter_rec.get("n_patches_skipped_as_rejected_dup", 0),
            f"iter/fcvr_silhouette": iter_rec.get("fcvr_silhouette") or 0.0,
            f"iter/cluster_interpretable": int(bool(iter_rec.get("cluster_interpretable"))),
            f"iter/decision_accept": int(_decision == "accept_child"),
            f"iter/decision_reject": int(_decision == "reject_no_child_better"),
            f"iter/decision_skip": int("skip" in _decision),
            f"iter/parent_minibatch_reward": (iter_rec.get("parent_minibatch_summary") or {}).get("mean_reward", 0.0),
            f"iter/parent_minibatch_succ_rate": (iter_rec.get("parent_minibatch_summary") or {}).get("success_rate", 0.0),
            f"iter/best_child_score_reward": (iter_rec.get("best_child_score") or [0])[0] if iter_rec.get("best_child_score") else 0,
            f"iter/fcvr_input_tokens_this_iter": iter_rec.get("fcvr_input_tokens", 0),
            f"iter/fcvr_output_tokens_this_iter": iter_rec.get("fcvr_output_tokens", 0),
            f"iter/cum_accepts": accepted_children,
        }, step=iter_k)

    # --- PHASE F (final eval) ----------------------------------------------
    logger.info("=== PHASE F: final eval of best parent on %d tasks ===", len(tasks))
    # If the last iter accepted, parent_full_records is already on the new parent.
    # If no iter ever accepted, parent_full_records is still phase A.
    # Either way, we record this as Phase F.
    F_records = parent_full_records
    F_summary = summarize(F_records)
    A_summary = summarize(A_records)

    # --- Headline paper metrics --------------------------------------------
    a_by_id = {r["task_id"]: r for r in A_records}
    loop_escape = []
    for r in F_records:
        ra = a_by_id.get(r["task_id"])
        if ra is None:
            continue
        a_loop = (ra.get("early_stop_reason") or "").startswith("repeated_actions_")
        f_loop = (r.get("early_stop_reason") or "").startswith("repeated_actions_")
        if a_loop and not f_loop:
            loop_escape.append(r["task_id"])

    elapsed_total = round(time.perf_counter() - t_total, 3)
    finished = datetime.datetime.utcnow().isoformat() + "Z"

    # --- Cost accounting (codex stop-time review 2026-05-28 fix) -----------
    # Sum backbone tokens across EVERY evaluate_candidate() call collected
    # in `all_eval_records_for_cost` (Phase A + per-iter parent minibatch +
    # per-iter children + per-iter full re-eval on accept + Phase F implicit
    # via parent_full_records aliasing).
    bb_in = 0
    bb_out = 0
    bb_reason = 0
    for recs in all_eval_records_for_cost:
        for r in recs or []:
            bb_in += r.get("backbone_prompt_tokens", 0) or 0
            bb_out += r.get("backbone_completion_tokens", 0) or 0
            bb_reason += r.get("backbone_reasoning_tokens", 0) or 0

    # Backbone cost: input + (output + reasoning) at the configured rates.
    # Reasoning tokens are typically billed at output rate.
    backbone_cost_usd = (
        bb_in * args.backbone_input_cost_per_mtok / 1_000_000
        + (bb_out + bb_reason) * args.backbone_output_cost_per_mtok / 1_000_000
    )
    reflection_cost_usd = (
        total_fcvr_input_tokens * args.reflection_input_cost_per_mtok / 1_000_000
        + total_fcvr_output_tokens * args.reflection_output_cost_per_mtok / 1_000_000
    )
    cost_usd = backbone_cost_usd + reflection_cost_usd

    overall = {
        "B2_proper": True,
        "seed": args.rng_seed,
        "started_at": started,
        "finished_at": finished,
        "elapsed_s_total": elapsed_total,
        "config": {
            "tasks_config": args.tasks,
            "n_tasks": len(tasks),
            "backbone_kind": args.backbone_kind,
            "backbone_model": (
                args.openai_model if args.backbone_kind == "openai_api"
                else args.backbone_model
            ),
            "backbone_endpoint": (
                args.openai_endpoint if args.backbone_kind == "openai_api"
                else args.backbone_endpoint
            ),
            "agent_max_trajectory_length": args.agent_max_trajectory_length,
            "agent_temperature": args.agent_temperature,
            "agent_top_p": args.agent_top_p,
            "agent_max_tokens": args.agent_max_tokens,
            "agent_image_detail": args.agent_image_detail,
            "reflection_model": args.reflection_model,
            "K": args.K, "J": args.J, "M": args.M, "T_patch": args.T_patch,
            "max_iters": args.max_iters,
            "minibatch_size": args.minibatch_size,
            "min_failures_for_fcvr": args.min_failures_for_fcvr,
            "max_steps": args.max_steps,
            "early_stop_on_repeated": args.early_stop_on_repeated,
            "step_watchdog": args.step_watchdog,
            "rng_seed": args.rng_seed,
        },
        "phaseA_vanilla": {
            "summary": A_summary,
            "tasks": [{k: v for k, v in r.items() if k != "_traj"} for r in A_records],
        },
        "phaseG_iters": iter_history,
        "phaseF_final": {
            "summary": F_summary,
            "tasks": [{k: v for k, v in r.items() if k != "_traj"} for r in F_records],
            "n_patches_in_final_prompt": len(parent_prompt.behavioral_patches),
        },
        "paper_metrics": {
            "phase_A_success_rate": A_summary["success_rate"],
            "phase_A_mean_reward": A_summary["mean_reward"],
            "phase_A_early_stop_rate": A_summary["early_stop_rate"],
            "phase_F_success_rate": F_summary["success_rate"],
            "phase_F_mean_reward": F_summary["mean_reward"],
            "phase_F_early_stop_rate": F_summary["early_stop_rate"],
            # promoted_loop_escape = tasks where Phase A early-stopped AND
            # Phase F (post-GEPA, accepted-only) did not. Final-policy effect.
            "promoted_loop_escape_count": len(loop_escape),
            "promoted_loop_escape_task_ids": loop_escape,
            # best_child_loop_escape = tasks where Phase A early-stopped AND
            # at least one child (ANY iter, accepted or rejected) did not
            # early-stop on that task. Codex Q4 (2026-05-27): captures FCVR's
            # reachability even when no child was promoted. Lossy: ignores
            # reward / partial credit; use alongside reward & success_rate.
            "best_child_loop_escape_count": len(best_child_loop_escape_set),
            "best_child_loop_escape_task_ids": sorted(best_child_loop_escape_set),
            # Backward-compat alias (older analyzers read "loop_escape_count"):
            "loop_escape_count": len(loop_escape),
            "loop_escape_task_ids": loop_escape,
            "gepa_iters_accepted": accepted_children,
            "gepa_iters_attempted": args.max_iters,
            "rejected_cluster_member_sets": [sorted(s) for s in rejected_member_sets],
        },
        "approx_claude_cost_usd": round(reflection_cost_usd, 4),
        "cost_breakdown_usd": {
            "backbone_input_tokens": bb_in,
            "backbone_output_tokens": bb_out,
            "backbone_reasoning_tokens": bb_reason,
            "backbone_cost_usd": round(backbone_cost_usd, 4),
            "reflection_input_tokens": total_fcvr_input_tokens,
            "reflection_output_tokens": total_fcvr_output_tokens,
            "reflection_cost_usd": round(reflection_cost_usd, 4),
            "total_cost_usd": round(cost_usd, 4),
            "rates_usd_per_mtok": {
                "backbone_input": args.backbone_input_cost_per_mtok,
                "backbone_output": args.backbone_output_cost_per_mtok,
                "reflection_input": args.reflection_input_cost_per_mtok,
                "reflection_output": args.reflection_output_cost_per_mtok,
            },
            "n_eval_record_lists_accumulated": len(all_eval_records_for_cost),
        },
        "pass_criteria_met": {
            "phase_A_ran": A_summary["n_completed"] > 0,
            "phase_F_ran": F_summary["n_completed"] > 0,
            "at_least_one_iter_recorded": len(iter_history) > 0,
            "overall": A_summary["n_completed"] > 0 and F_summary["n_completed"] > 0,
        },
    }
    out_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False, default=str))
    logger.info("wrote %s", out_path)
    logger.info("summary: %s", json.dumps(overall["pass_criteria_met"], indent=2))

    m.finish(
        result_path=str(out_path),
        notes=(
            f"seed={args.rng_seed} A={A_summary['n_succeeded']}/{A_summary['n_tasks']} "
            f"F={F_summary['n_succeeded']}/{F_summary['n_tasks']} "
            f"accepted={accepted_children}/{args.max_iters} "
            f"cost=${cost_usd:.3f} (bb=${backbone_cost_usd:.3f} refl=${reflection_cost_usd:.3f})"
        ),
    )
    m.write(manifest_path)
    logger.info("wrote manifest %s", manifest_path)

    # Final headline + cost log to swanlab (codex stop-time fix 2026-05-29).
    # step=max_iters+1 places these AFTER all per-iter points on the timeline.
    final_step = args.max_iters + 1
    sl_log({
        "final/phase_A_success_rate": A_summary["success_rate"],
        "final/phase_A_mean_reward": A_summary["mean_reward"],
        "final/phase_A_early_stop_rate": A_summary["early_stop_rate"],
        "final/phase_F_success_rate": F_summary["success_rate"],
        "final/phase_F_mean_reward": F_summary["mean_reward"],
        "final/phase_F_early_stop_rate": F_summary["early_stop_rate"],
        "final/promoted_loop_escape_count": len(loop_escape),
        "final/best_child_loop_escape_count": len(best_child_loop_escape_set),
        "final/gepa_iters_accepted": accepted_children,
        "final/gepa_iters_attempted": args.max_iters,
        "final/n_patches_in_final_prompt": len(parent_prompt.behavioral_patches),
        "cost/backbone_input_tokens": bb_in,
        "cost/backbone_output_tokens": bb_out,
        "cost/backbone_reasoning_tokens": bb_reason,
        "cost/backbone_cost_usd": backbone_cost_usd,
        "cost/reflection_input_tokens": total_fcvr_input_tokens,
        "cost/reflection_output_tokens": total_fcvr_output_tokens,
        "cost/reflection_cost_usd": reflection_cost_usd,
        "cost/total_cost_usd": cost_usd,
    }, step=final_step)
    if swanlab_active:
        try:
            swanlab.finish()
        except Exception as e:  # noqa: BLE001
            logger.warning("swanlab.finish failed: %s", e)

    return 0 if overall["pass_criteria_met"]["overall"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
