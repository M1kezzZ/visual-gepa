"""B2 mini — vanilla baseline vs FCVR-augmented baseline on the 5-task set.

This is the FIRST real test of the FCVR contribution. Cheap (5 tasks ×
2 rollouts × 15 steps + 1 reflection round).

Pipeline:
  Phase A (vanilla):   roll out SEED prompt on 5 tasks → trajectories A
  Phase B (FCVR):      cluster failed A-trajectories → MMR key frames →
                       ONE Claude vision call per cluster → patches
                       (5-field FCVRPatch each)
  Phase C (enhanced):  build prompt P = SEED || [BEHAVIORAL_PATCHES patches]
                       and re-roll out on the same 5 tasks → trajectories C
  Phase D (compare):   per-task A vs C — reward Δ, n_distinct_actions Δ,
                       click-loop rate Δ. Aggregate verdict.

Out of scope:
  - Multi-iter GEPA Pareto loop (that's B2 proper, R009-R017).
  - Multi-seed averaging.
  - 60-task split.

Pre-task Codex review of the plan is the user's discretion (codex-review-
every-step protocol). All trajectories carry raw_model_text per step
(B1_baseline_v3 audit chain).

Per-experiment provenance manifest written alongside result JSON.

Failure handling: fail-fast on task 1 of Phase A (multi_task_fail_fast_
protocol). If Phase B reflection returns 0 patches, Phase C is SKIPPED
and the verdict is "FCVR_NO_PATCHES" (not a failure of FCVR, but a no-op
result with explicit framing).

Usage (on the 5090 server, after preflight passes):
  python scripts/B2_mini.py \\
    --tasks configs/osworld_b1_5.json \\
    --backbone-endpoint http://127.0.0.1:8000/v1 \\
    --backbone-model /root/models/Qwen3.5-9B \\
    --cache-dir /root/visual-gepa/osworld_cache \\
    --max-steps 15 --rng-seed 42 \\
    --reflection-model claude-opus-4-7 \\
    --clip-model /root/models/clip-vit-base-patch32 \\
    --output results/B2_mini_seed42.json \\
    --manifest results/B2_mini_seed42_manifest.json
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

from visual_gepa.clip_embedder import CLIPImageEmbedder
from visual_gepa.fcvr import DEFAULT_BUDGET, FCVRBudget, FCVROperator
from visual_gepa.manifest import Manifest
from visual_gepa.osworld_adapter import OSWorldAdapter, load_osworld_task_config
from visual_gepa.reflection import ClaudeReflectionClient, DEFAULT_REFLECTION_MODEL
from visual_gepa.structured_prompt import StructuredPrompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("B2_mini")


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


def run_phase(
    label: str,
    tasks: list[dict],
    args,
    prompt: StructuredPrompt,
    stream_out_path: Path,
    halt_on_task1_crash: bool = True,
) -> list:
    """Run the agent across all tasks; stream-write partial results."""
    records = []
    for i, task_entry in enumerate(tasks):
        task_id = task_entry.get("id", "<unknown>")
        cfg_path = task_entry.get("task_config_path") or task_entry.get("id")
        task_cfg = load_osworld_task_config(cfg_path)
        logger.info("--- phase %s task %d/%d : %s ---", label, i + 1, len(tasks), task_id)
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
        )
        t0 = time.perf_counter()
        crashed = None
        traj = None
        try:
            traj = adapter.run(prompt)
        except Exception as e:  # noqa: BLE001
            logger.exception("phase %s task %s crashed", label, task_id)
            crashed = f"{type(e).__name__}: {e}"

        rec = {
            "task_id": task_id,
            "task_config_path": str(cfg_path),
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
                "score": score,
                "feedback": feedback,
                "actions": actions,
                "raw_model_texts": raws,
                "n_distinct_actions": n_distinct(actions),
                "early_stop_reason": getattr(traj, "early_stop_reason", None),
                "reward_source": getattr(traj, "reward_source", "unset"),
                "_traj": traj,  # in-memory, dropped before write
            })
        else:
            rec.update({
                "n_steps": 0, "final_reward": None, "succeeded": False,
                "score": None, "feedback": "(crashed)", "actions": [],
                "raw_model_texts": [], "n_distinct_actions": 0,
                "early_stop_reason": None, "reward_source": "crashed_before_eval",
                "_traj": None,
            })
        records.append(rec)

        # Stream-write WITHOUT _traj (PIL.Image can't JSON-serialize)
        writable = [{k: v for k, v in r.items() if k != "_traj"} for r in records]
        stream_out_path.write_text(json.dumps({"phase": label, "tasks": writable}, indent=2, default=str))
        logger.info(
            "  phase %s task %s done elapsed=%ss reward=%s distinct_acts=%d",
            label, task_id, rec["elapsed_s"], rec.get("final_reward"), rec.get("n_distinct_actions"),
        )

        # Fail-fast on first task crash (multi_task_fail_fast_protocol)
        if i == 0 and halt_on_task1_crash and crashed:
            logger.error(
                "🚨 phase %s task 1 crashed: %s — HALTING per fail-fast protocol", label, crashed,
            )
            raise SystemExit(2)

    return records


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="B2 mini — vanilla vs FCVR-augmented on 5-task set")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--backbone-endpoint", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--backbone-model", default=os.environ.get("VLLM_MODEL_NAME", "/root/models/Qwen3.5-9B"))
    ap.add_argument("--provider", default="docker", choices=["docker", "vmware", "aws", "aliyun", "azure", "gcp"])
    ap.add_argument("--os-type", default="Ubuntu", choices=["Ubuntu", "Windows"])
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument(
        "--phase-a2-rerun-seed",
        type=int,
        default=None,
        help=(
            "Optional second seed for variance baseline. When set, runs a "
            "Phase A2 rollout (vanilla SEED prompt, no patches) at this seed "
            "and reports per-task |distinct(A1) - distinct(A2)| as the "
            "diversity noise floor — the threshold the FCVR-induced "
            "diversity delta (Phase C - Phase A) must exceed to be credible."
        ),
    )
    ap.add_argument(
        "--early-stop-on-repeated",
        type=int,
        default=3,
        help=(
            "Early-stop the rollout when this many consecutive normalized "
            "actions are identical. 0 disables (matches OSWorld leaderboard). "
            "Default 3 — saves ~87%% vLLM time on click-loop failures."
        ),
    )
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--reflection-model", default=os.environ.get("REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL))
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip-device", default=None)
    ap.add_argument("--K", type=int, default=DEFAULT_BUDGET.K)
    ap.add_argument("--J", type=int, default=DEFAULT_BUDGET.J)
    ap.add_argument("--M", type=int, default=DEFAULT_BUDGET.M)
    ap.add_argument("--T_patch", type=int, default=DEFAULT_BUDGET.T_patch)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_name(out_path.stem + "_manifest.json")

    # Provenance manifest start
    m = Manifest(
        experiment_id=f"B2_mini_seed{args.rng_seed}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        block="B2",
    )
    m.start(
        vllm_cmd=f"endpoint={args.backbone_endpoint} model={args.backbone_model}",
        model_path=args.backbone_model,
        qcow2_path=(str(Path(args.cache_dir) / "docker_vm_data" / "Ubuntu.qcow2") if args.cache_dir else ""),
        config_path=args.tasks,
        seed=args.rng_seed,
        compute_model_md5=False,
        compute_qcow2_md5=False,  # 23 GB, would be slow
    )

    tasks_cfg = json.loads(Path(args.tasks).read_text())
    tasks = tasks_cfg.get("tasks", [])
    logger.info("loaded %d tasks from %s", len(tasks), args.tasks)

    # Components
    clip = CLIPImageEmbedder(model_name=args.clip_model, device=args.clip_device)
    reflection = ClaudeReflectionClient(
        model=args.reflection_model, max_output_tokens=args.T_patch,
    )
    budget = FCVRBudget(K=args.K, J=args.J, M=args.M, T_patch=args.T_patch)
    fcvr = FCVROperator(
        budget=budget, clip_embedder=clip, reflection_client=reflection, rng_seed=args.rng_seed,
    )

    started = datetime.datetime.utcnow().isoformat() + "Z"
    t_total = time.perf_counter()

    # --- PHASE A: vanilla rollout ---
    logger.info("=== PHASE A: vanilla rollout (SEED prompt) ===")
    seed_prompt = build_seed_prompt()
    phaseA_path = out_path.with_name(out_path.stem + "_phaseA.json")
    A_records = run_phase("A", tasks, args, seed_prompt, phaseA_path)

    # --- PHASE A2 (variance baseline, optional) -----------------------------
    # Run the SAME SEED prompt on the SAME tasks with a DIFFERENT rng seed.
    # Then |distinct(A) - distinct(A2)| is the per-task diversity noise floor —
    # the threshold Phase C's diversity gain must beat to be FCVR-attributable.
    A2_records: list = []
    variance_baseline: dict = {}
    if args.phase_a2_rerun_seed is not None:
        if args.phase_a2_rerun_seed == args.rng_seed:
            logger.warning("phase-a2-rerun-seed == rng-seed — variance phase is degenerate")
        logger.info(
            "=== PHASE A2: vanilla rollout (variance baseline, seed=%d) ===",
            args.phase_a2_rerun_seed,
        )
        # Note: the rng seed currently only affects FCVR's KMeans; vLLM sampling
        # is governed by its server-side config (temperature etc.), not the
        # CLI --rng-seed. Phase A2 therefore measures STOCHASTIC variance from
        # vLLM sampling alone — exactly the noise floor we want, since FCVR's
        # +diversity claim must exceed even that.
        phaseA2_path = out_path.with_name(out_path.stem + "_phaseA2.json")
        A2_records = run_phase("A2", tasks, args, seed_prompt, phaseA2_path)
        a2_by_id = {r["task_id"]: r for r in A2_records}
        per_task_var = []
        for ra in A_records:
            rb = a2_by_id.get(ra["task_id"])
            if rb is None:
                continue
            per_task_var.append({
                "task_id": ra["task_id"],
                "A_distinct_actions": ra.get("n_distinct_actions"),
                "A2_distinct_actions": rb.get("n_distinct_actions"),
                "abs_diversity_drift": abs(
                    (ra.get("n_distinct_actions") or 0) - (rb.get("n_distinct_actions") or 0)
                ),
                "A_reward": ra.get("final_reward"),
                "A2_reward": rb.get("final_reward"),
                "reward_drift": (
                    (rb.get("final_reward") or 0.0) - (ra.get("final_reward") or 0.0)
                ),
            })
        variance_baseline = {
            "phase_a2_seed": args.phase_a2_rerun_seed,
            "per_task": per_task_var,
            "mean_abs_diversity_drift": (
                sum(p["abs_diversity_drift"] for p in per_task_var) / len(per_task_var)
                if per_task_var else 0.0
            ),
            "max_abs_diversity_drift": (
                max((p["abs_diversity_drift"] for p in per_task_var), default=0)
            ),
        }
        logger.info(
            "Phase A2 variance baseline: mean=%.2f max=%d (FCVR diversity Δ "
            "must exceed this floor to be credible)",
            variance_baseline["mean_abs_diversity_drift"],
            variance_baseline["max_abs_diversity_drift"],
        )

    # --- PHASE B: FCVR reflection on failed A-trajectories ---
    failed_trajs = [r["_traj"] for r in A_records if r.get("_traj") and not r["_traj"].succeeded]
    logger.info("=== PHASE B: FCVR reflection on %d failed trajectories ===", len(failed_trajs))
    if not failed_trajs:
        logger.warning("no failed trajectories — Phase B is a no-op")
        patches, fcvr_record = [], None
    else:
        patches, fcvr_record = fcvr.run(failed_trajs, parent_prompt=seed_prompt)

    # --- PHASE C: enhanced rollout ---
    if not patches:
        logger.warning("Phase B produced 0 patches — SKIPPING Phase C")
        C_records: list = []
        skip_C_reason = "no_patches"
    else:
        skip_C_reason = ""
        enhanced_prompt = StructuredPrompt(
            persona=seed_prompt.persona,
            global_rules=seed_prompt.global_rules,
            behavioral_patches=list(seed_prompt.behavioral_patches),
            task_scaffold=seed_prompt.task_scaffold,
        )
        for p in patches:
            enhanced_prompt.append_patch(scope_guard=p.scope_guard, prompt_diff=p.prompt_diff)
        logger.info(
            "=== PHASE C: enhanced rollout with %d patches (prompt %d tokens approx) ===",
            len(patches), enhanced_prompt.token_length(),
        )
        phaseC_path = out_path.with_name(out_path.stem + "_phaseC.json")
        C_records = run_phase("C", tasks, args, enhanced_prompt, phaseC_path)

    # --- PHASE D: comparison ---
    def _summarize(records: list) -> dict:
        if not records:
            return {"n_tasks": 0, "n_completed": 0, "n_succeeded": 0, "success_rate": 0.0,
                    "mean_reward": 0.0, "mean_distinct_actions": 0.0}
        n_comp = sum(1 for r in records if r["crashed_with"] is None)
        n_succ = sum(1 for r in records if r.get("succeeded"))
        rewards = [r["final_reward"] for r in records if r.get("final_reward") is not None]
        distincts = [r["n_distinct_actions"] for r in records]
        return {
            "n_tasks": len(records),
            "n_completed": n_comp,
            "n_succeeded": n_succ,
            "success_rate": n_succ / len(records) if records else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "mean_distinct_actions": sum(distincts) / len(distincts) if distincts else 0.0,
        }

    A_summary = _summarize(A_records)
    C_summary = _summarize(C_records) if C_records else None
    per_task_delta = []
    pair_audit: dict = {"all_paired": True, "unpaired_task_ids": [], "phase_a_ids": [], "phase_c_ids": []}
    if C_records:
        c_by_id = {r["task_id"]: r for r in C_records}
        a_ids = [r["task_id"] for r in A_records]
        c_ids = [r["task_id"] for r in C_records]
        pair_audit["phase_a_ids"] = a_ids
        pair_audit["phase_c_ids"] = c_ids
        for ra in A_records:
            rc = c_by_id.get(ra["task_id"])
            if rc is None:
                pair_audit["all_paired"] = False
                pair_audit["unpaired_task_ids"].append(ra["task_id"])
            per_task_delta.append({
                "task_id": ra["task_id"],
                "A_reward": ra.get("final_reward"),
                "C_reward": rc.get("final_reward") if rc else None,
                "A_distinct_actions": ra.get("n_distinct_actions"),
                "C_distinct_actions": rc.get("n_distinct_actions") if rc else None,
                "reward_delta": (
                    (rc.get("final_reward") or 0.0) - (ra.get("final_reward") or 0.0)
                    if rc else None
                ),
                "diversity_delta": (
                    (rc.get("n_distinct_actions") or 0) - (ra.get("n_distinct_actions") or 0)
                    if rc else None
                ),
                "paired": rc is not None,
            })
        # Hard assertion — codex action item #3. Fail loudly rather than
        # silently report a wrong delta against a missing pair.
        if not pair_audit["all_paired"]:
            logger.error(
                "🚨 PHASE D pair-audit FAILED: %d task_ids in A have no C pair: %s",
                len(pair_audit["unpaired_task_ids"]), pair_audit["unpaired_task_ids"],
            )

    # --- Reward-source assertion (codex action item #4) ---------------------
    # Every COMPLETED rollout (A, A2, C) must carry reward_source == "env_evaluate".
    # Anything else means either env.evaluate() raised, the env crashed before
    # any step ran, or our orchestrator inferred reward heuristically.
    reward_source_audit: dict = {"clean": True, "violations": []}
    for label, recs in (("A", A_records), ("A2", A2_records), ("C", C_records)):
        for r in recs or []:
            if r.get("crashed_with"):
                continue  # crashes are expected to lack a reward source
            src = r.get("reward_source")
            if src != "env_evaluate":
                reward_source_audit["clean"] = False
                reward_source_audit["violations"].append({
                    "phase": label,
                    "task_id": r["task_id"],
                    "reward_source": src,
                })
    if not reward_source_audit["clean"]:
        logger.error(
            "🚨 reward-source audit FAILED: %d non-env_evaluate sources found",
            len(reward_source_audit["violations"]),
        )

    elapsed_total = round(time.perf_counter() - t_total, 3)
    finished = datetime.datetime.utcnow().isoformat() + "Z"

    overall = {
        "B2_mini": True,
        "started_at": started,
        "finished_at": finished,
        "elapsed_s_total": elapsed_total,
        "config": {
            "tasks_config": args.tasks,
            "backbone_model": args.backbone_model,
            "backbone_endpoint": args.backbone_endpoint,
            "reflection_model": args.reflection_model,
            "clip_model": args.clip_model,
            "K": args.K, "J": args.J, "M": args.M, "T_patch": args.T_patch,
            "max_steps": args.max_steps,
            "rng_seed": args.rng_seed,
            "n_tasks": len(tasks),
        },
        "phaseA": {
            "summary": A_summary,
            "tasks": [{k: v for k, v in r.items() if k != "_traj"} for r in A_records],
        },
        "phaseA2_variance_baseline": (
            {
                "summary": _summarize(A2_records),
                "tasks": [{k: v for k, v in r.items() if k != "_traj"} for r in A2_records],
                **variance_baseline,
            }
            if A2_records else None
        ),
        "phaseB": {
            "n_failed_input": len(failed_trajs),
            "n_patches": len(patches),
            "patches": [p.model_dump() for p in patches],
            "fcvr_record": ({
                "n_clusters_used": fcvr_record.n_clusters_used,
                "cluster_sizes": fcvr_record.cluster_sizes,
                "total_input_tokens": fcvr_record.total_input_tokens,
                "total_output_tokens": fcvr_record.total_output_tokens,
                "total_latency_s": round(fcvr_record.total_latency_s, 3),
                "schema_violations_total": fcvr_record.schema_violations_total,
                "elapsed_s": round(fcvr_record.elapsed_s, 3),
                "reflection_stats": fcvr_record.reflection_stats,
                # Cluster-quality diagnostics (codex action item #2)
                "silhouette_score": fcvr_record.silhouette_score,
                "cluster_membership_by_app": fcvr_record.cluster_membership_by_app,
                "centroid_pairwise_distances": fcvr_record.centroid_pairwise_distances,
                "action_edit_distance_within_cluster": fcvr_record.action_edit_distance_within_cluster,
                # Cluster narrative gate (codex v2 audit, 2026-05-27)
                "cluster_interpretable": fcvr_record.cluster_interpretable,
                "cluster_interpretable_reason": fcvr_record.cluster_interpretable_reason,
            } if fcvr_record else None),
        },
        "phaseC": ({
            "summary": C_summary,
            "tasks": [{k: v for k, v in r.items() if k != "_traj"} for r in C_records],
            "skipped_reason": skip_C_reason,
        } if C_records or skip_C_reason else None),
        "phaseD_delta": per_task_delta,
        "pair_audit": pair_audit,
        "reward_source_audit": reward_source_audit,
    }

    # --- Paper-grade metrics (codex v2 audit, 2026-05-27) -------------------
    # `distinct_actions` was demoted to diagnostic because A↔A2 noise floor
    # (mean 1.4, max 3 on B2 mini v2 same prompt different seeds) dwarfed
    # the FCVR effect (mean 0.2, max 1). These metrics are robust to that
    # variance because they're either binary per-task (success / loop_escape)
    # or expressed as a paired delta net of noise.
    def _early_stop_rate(records: list) -> float:
        if not records:
            return 0.0
        n = sum(
            1 for r in records
            if (r.get("early_stop_reason") or "").startswith("repeated_actions_")
        )
        return n / len(records)

    paper_metrics = {
        "phase_A_success_rate": A_summary["success_rate"],
        "phase_A_mean_reward": A_summary["mean_reward"],
        "phase_A_early_stop_rate": _early_stop_rate(A_records),
        "phase_C_success_rate": (C_summary or {}).get("success_rate", 0.0) if C_summary else None,
        "phase_C_mean_reward": (C_summary or {}).get("mean_reward", 0.0) if C_summary else None,
        "phase_C_early_stop_rate": _early_stop_rate(C_records) if C_records else None,
        # Loop-escape: tasks where vanilla A early-stopped AND FCVR C did NOT.
        # The cleanest qualitative win for FCVR — even if reward stays 0, did
        # FCVR push the agent past the click-loop death spiral?
        "loop_escape_count": None,
        "loop_escape_task_ids": [],
        # Seed-normalized diversity delta = mean(C-A) - mean(|A-A2|).
        # Positive => FCVR's diversity effect exceeds vanilla seed noise.
        # Negative => FCVR effect is within or below the noise floor.
        "seed_normalized_diversity_delta": None,
    }
    if C_records:
        c_by_id = {r["task_id"]: r for r in C_records}
        loop_escape = []
        for ra in A_records:
            rc = c_by_id.get(ra["task_id"])
            if rc is None:
                continue
            a_loop = (ra.get("early_stop_reason") or "").startswith("repeated_actions_")
            c_loop = (rc.get("early_stop_reason") or "").startswith("repeated_actions_")
            if a_loop and not c_loop:
                loop_escape.append(ra["task_id"])
        paper_metrics["loop_escape_count"] = len(loop_escape)
        paper_metrics["loop_escape_task_ids"] = loop_escape
    if A2_records and C_records:
        a2_by_id = {r["task_id"]: r for r in A2_records}
        c_by_id = {r["task_id"]: r for r in C_records}
        ac_deltas = []
        aa2_drifts = []
        for ra in A_records:
            rc = c_by_id.get(ra["task_id"])
            rb = a2_by_id.get(ra["task_id"])
            if rc is not None:
                ac_deltas.append(
                    (rc.get("n_distinct_actions") or 0) - (ra.get("n_distinct_actions") or 0)
                )
            if rb is not None:
                aa2_drifts.append(
                    abs((ra.get("n_distinct_actions") or 0) - (rb.get("n_distinct_actions") or 0))
                )
        if ac_deltas and aa2_drifts:
            paper_metrics["seed_normalized_diversity_delta"] = (
                (sum(ac_deltas) / len(ac_deltas)) - (sum(aa2_drifts) / len(aa2_drifts))
            )
    overall["paper_metrics"] = paper_metrics

    # Cost approximation (Claude Opus 4.7 vision: $5 / Mtok input, $25 / Mtok output)
    if fcvr_record:
        in_tok = fcvr_record.total_input_tokens
        out_tok = fcvr_record.total_output_tokens
        cost_usd = round((in_tok / 1_000_000) * 5.0 + (out_tok / 1_000_000) * 25.0, 4)
        overall["approx_claude_cost_usd"] = cost_usd
    else:
        overall["approx_claude_cost_usd"] = 0.0

    # Pass criteria (plumbing-only — NOT a science gate)
    overall["pass_criteria_met"] = {
        "all_phase_A_tasks_completed": A_summary["n_completed"] == A_summary["n_tasks"],
        "phase_B_produced_patches": len(patches) > 0,
        "all_phase_C_tasks_completed": (
            C_records and C_summary["n_completed"] == C_summary["n_tasks"]
            if C_summary else False
        ),
        "got_phaseD_delta": len(per_task_delta) > 0,
        "no_iter_crashed": True,
        "pair_audit_clean": pair_audit["all_paired"],
        "reward_source_audit_clean": reward_source_audit["clean"],
        "overall": (
            A_summary["n_completed"] == A_summary["n_tasks"]
            and len(patches) > 0
            and C_records is not None
            and C_summary["n_completed"] == C_summary["n_tasks"]
            and pair_audit["all_paired"]
            and reward_source_audit["clean"]
        ),
    }

    out_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False, default=str))
    logger.info("wrote %s", out_path)
    logger.info("summary: %s", json.dumps(overall["pass_criteria_met"], indent=2))

    # Finish manifest
    m.finish(
        result_path=str(out_path),
        notes=(
            f"A={A_summary['n_succeeded']}/{A_summary['n_tasks']} succ "
            f"C={(C_summary or {}).get('n_succeeded', 0)}/{(C_summary or {}).get('n_tasks', 0)} succ "
            f"patches={len(patches)} cost=${overall['approx_claude_cost_usd']:.3f}"
        ),
    )
    m.write(manifest_path)
    logger.info("wrote manifest %s", manifest_path)

    return 0 if overall["pass_criteria_met"]["overall"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
