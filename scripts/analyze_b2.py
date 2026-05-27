"""B2 result analyzer — bootstrap CIs on paired A↔C metrics.

Works on B2 mini result JSONs (single seed, with phaseA / phaseA2 /
phaseC) and on B2 proper aggregate JSONs (multi-seed). Output is a
plain-text table + per-seed breakdown.

Codex v2 audit (2026-05-27) action item: report seed-normalized effects
with bootstrap CIs so `distinct_actions` (high-variance metric) is
interpretable. Headline metrics are reward, success_rate,
early_stop_rate, loop_escape_count.

Usage:
    python scripts/analyze_b2.py results/B2_mini_v2_seed42.json
    python scripts/analyze_b2.py results/B2_proper_*.json   # glob OK
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path
from statistics import mean, median

logger = logging.getLogger("analyze_b2")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def bootstrap_ci(samples: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high). Empty → all zeros."""
    if not samples:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n = len(samples)
    means = []
    for _ in range(n_boot):
        resamp = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resamp) / n)
    means.sort()
    lo = means[int(n_boot * (alpha / 2))]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (sum(samples) / n), lo, hi


def fmt_ci(m: float, lo: float, hi: float, prec: int = 2) -> str:
    return f"{m:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def extract_paired_metrics(data: dict) -> dict:
    """Pull per-task paired (A, C) and (A, A2) records from one result JSON.

    Returns dict with: ac_reward_deltas, ac_diversity_deltas,
    aa2_diversity_drifts, a_succ, c_succ, a_early_stops, c_early_stops,
    loop_escapes, n_tasks.
    """
    A = data["phaseA"]["tasks"]
    C_pkg = data.get("phaseC") or {}
    C = C_pkg.get("tasks") or []
    A2_pkg = data.get("phaseA2_variance_baseline") or {}
    A2 = A2_pkg.get("tasks") or []

    a_by_id = {r["task_id"]: r for r in A}
    c_by_id = {r["task_id"]: r for r in C}
    a2_by_id = {r["task_id"]: r for r in A2}

    out: dict[str, list] = {
        "ac_reward_deltas": [],
        "ac_diversity_deltas": [],
        "aa2_diversity_drifts": [],
        "a_succ": [],
        "c_succ": [],
        "a_early_stops": [],
        "c_early_stops": [],
        "loop_escapes": [],
        "task_ids": [],
    }
    for tid, ra in a_by_id.items():
        out["task_ids"].append(tid)
        out["a_succ"].append(1 if ra.get("succeeded") else 0)
        a_es = (ra.get("early_stop_reason") or "").startswith("repeated_actions_")
        out["a_early_stops"].append(1 if a_es else 0)
        rc = c_by_id.get(tid)
        if rc:
            out["c_succ"].append(1 if rc.get("succeeded") else 0)
            c_es = (rc.get("early_stop_reason") or "").startswith("repeated_actions_")
            out["c_early_stops"].append(1 if c_es else 0)
            out["ac_reward_deltas"].append(
                (rc.get("final_reward") or 0.0) - (ra.get("final_reward") or 0.0)
            )
            out["ac_diversity_deltas"].append(
                (rc.get("n_distinct_actions") or 0) - (ra.get("n_distinct_actions") or 0)
            )
            out["loop_escapes"].append(1 if (a_es and not c_es) else 0)
        rb = a2_by_id.get(tid)
        if rb:
            out["aa2_diversity_drifts"].append(
                abs((ra.get("n_distinct_actions") or 0) - (rb.get("n_distinct_actions") or 0))
            )
    out["n_tasks"] = len(out["task_ids"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="result JSON file(s) or glob(s)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    if not files:
        print("no input files", file=sys.stderr)
        return 2

    # Aggregate across seeds
    agg = {
        "ac_reward_deltas": [],
        "ac_diversity_deltas": [],
        "aa2_diversity_drifts": [],
        "a_succ": [],
        "c_succ": [],
        "a_early_stops": [],
        "c_early_stops": [],
        "loop_escapes": [],
    }
    per_file = []
    for f in files:
        try:
            data = json.loads(Path(f).read_text())
        except Exception as e:  # noqa: BLE001
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        seed = data.get("config", {}).get("rng_seed")
        m = extract_paired_metrics(data)
        per_file.append((Path(f).name, seed, m))
        for k in agg:
            if k in m:
                agg[k].extend(m[k])

    print(f"\n=== B2 analyzer — {len(per_file)} file(s) ===\n")

    # Per-file headline
    print(f"{'file':<50} {'seed':>5} {'n':>4} {'A_succ':>7} {'C_succ':>7} {'A_es%':>6} {'C_es%':>6} {'loop_esc':>9}")
    for fn, seed, m in per_file:
        n = m["n_tasks"]
        a_succ = sum(m["a_succ"]) / max(1, len(m["a_succ"]))
        c_succ = sum(m["c_succ"]) / max(1, len(m["c_succ"]))
        a_es = sum(m["a_early_stops"]) / max(1, len(m["a_early_stops"]))
        c_es = sum(m["c_early_stops"]) / max(1, len(m["c_early_stops"]))
        loop_esc = sum(m["loop_escapes"])
        print(f"{fn:<50} {str(seed):>5} {n:>4} {a_succ:>7.2f} {c_succ:>7.2f} {a_es:>6.0%} {c_es:>6.0%} {loop_esc:>9}")

    print()
    print(f"=== Aggregate ({len(files)} files; bootstrap n={args.n_boot}, alpha={args.alpha}) ===\n")

    if agg["ac_reward_deltas"]:
        m, lo, hi = bootstrap_ci(agg["ac_reward_deltas"], args.n_boot, args.alpha)
        print(f"  A→C reward delta:           {fmt_ci(m, lo, hi, 3)}")
    if agg["ac_diversity_deltas"]:
        m, lo, hi = bootstrap_ci(agg["ac_diversity_deltas"], args.n_boot, args.alpha)
        print(f"  A→C distinct_actions delta: {fmt_ci(m, lo, hi)}   ← diagnostic only")
    if agg["aa2_diversity_drifts"]:
        m, lo, hi = bootstrap_ci(agg["aa2_diversity_drifts"], args.n_boot, args.alpha)
        print(f"  A↔A2 vanilla drift |Δ|:     {fmt_ci(m, lo, hi)}   ← noise floor")
    if agg["ac_diversity_deltas"] and agg["aa2_diversity_drifts"]:
        # Seed-normalized: FCVR effect mean minus vanilla noise mean
        ac_m = mean(agg["ac_diversity_deltas"])
        aa2_m = mean(agg["aa2_diversity_drifts"])
        signed_normalized = [d - aa2_m for d in agg["ac_diversity_deltas"]]
        m, lo, hi = bootstrap_ci(signed_normalized, args.n_boot, args.alpha)
        print(f"  Seed-normalized (C-A)-|A-A2|: {fmt_ci(m, lo, hi)}   ← FCVR > 0 means above noise")
    if agg["loop_escapes"]:
        m, lo, hi = bootstrap_ci(agg["loop_escapes"], args.n_boot, args.alpha)
        n_total = len(agg["loop_escapes"])
        print(f"  Loop-escape rate:           {fmt_ci(m, lo, hi)}   (n_escapes={sum(agg['loop_escapes'])}/{n_total})")
    if agg["a_succ"]:
        m, lo, hi = bootstrap_ci(agg["a_succ"], args.n_boot, args.alpha)
        print(f"  Phase A success rate:       {fmt_ci(m, lo, hi)}")
    if agg["c_succ"]:
        m, lo, hi = bootstrap_ci(agg["c_succ"], args.n_boot, args.alpha)
        print(f"  Phase C success rate:       {fmt_ci(m, lo, hi)}")
    if agg["a_early_stops"]:
        m, lo, hi = bootstrap_ci(agg["a_early_stops"], args.n_boot, args.alpha)
        print(f"  Phase A early-stop rate:    {fmt_ci(m, lo, hi)}")
    if agg["c_early_stops"]:
        m, lo, hi = bootstrap_ci(agg["c_early_stops"], args.n_boot, args.alpha)
        print(f"  Phase C early-stop rate:    {fmt_ci(m, lo, hi)}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
