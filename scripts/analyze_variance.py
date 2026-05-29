"""Variance-curve analyzer — quantify the sample-complexity barrier.

Input: a B2_proper result JSON whose phaseA_vanilla.tasks carry per-task
`sample_rewards` lists (run with --eval-samples N large, e.g. N=10).

Produces, from the measured per-task reward samples:
  1. CI-width-vs-N: bootstrap the per-prompt mean reward at N=1,3,5,10,...
     and report the 95% CI half-width — how tight the estimate gets.
  2. False-promotion-probability-vs-N: under the NULL (a "child" drawn from
     the SAME per-task reward distributions as the parent — i.e. NO true
     effect), simulate the GEPA-lite accept rule (mean_reward strictly
     greater) and report P(accept). This is how often promotion fires on
     pure noise at each N — the killer figure for the mechanism story.
  3. Required-N to detect a target effect (default +0.15 mean reward) such
     that the two-sided 95% CIs of parent and child separate.

This is the evidence that decides whether a decisive seed-vs-evolved test
is affordable, and the centerpiece figure for the mechanism framing.

Usage:
  python scripts/analyze_variance.py results/B2_variance_curve_seed_N10.json
  python scripts/analyze_variance.py <json> --target-effect 0.15 --n-boot 5000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path


def load_per_task_samples(path: str) -> dict[str, list[float]]:
    """Return {task_id: [reward per sample]} from a B2_proper result."""
    d = json.loads(Path(path).read_text())
    pkg = d.get("phaseA_vanilla") or d.get("phaseA") or {}
    out: dict[str, list[float]] = {}
    for t in pkg.get("tasks", []):
        sr = t.get("sample_rewards")
        if sr is None:
            # fall back to per_sample_rewards_with_crash_as_0 if present
            sr = t.get("per_sample_rewards_with_crash_as_0")
        if sr:
            out[t["task_id"]] = [float(x) for x in sr]
    return out


def prompt_mean_from_subsample(samples_by_task: dict[str, list[float]],
                               n: int, rng: random.Random) -> float:
    """One prompt-level mean: for each task draw n samples (with replacement
    if n > available), average per task, then average across tasks."""
    task_means = []
    for _tid, rewards in samples_by_task.items():
        if not rewards:
            continue
        draw = [rewards[rng.randrange(len(rewards))] for _ in range(n)]
        task_means.append(sum(draw) / n)
    return sum(task_means) / len(task_means) if task_means else 0.0


def ci_halfwidth_vs_n(samples_by_task, ns, n_boot, rng):
    """For each N, bootstrap the prompt-level mean and report 95% CI half-width."""
    rows = []
    for n in ns:
        means = [prompt_mean_from_subsample(samples_by_task, n, rng) for _ in range(n_boot)]
        means.sort()
        lo = means[int(n_boot * 0.025)]
        hi = means[int(n_boot * 0.975)]
        rows.append((n, sum(means) / len(means), (hi - lo) / 2, lo, hi))
    return rows


def false_promotion_prob(samples_by_task, ns, n_boot, rng, epsilon=0.0):
    """Under the NULL (parent and child both drawn from the SAME per-task
    distributions = no true effect), P(child mean > parent mean + epsilon)
    using the GEPA-lite strict-greater accept rule, at each N."""
    rows = []
    for n in ns:
        accepts = 0
        for _ in range(n_boot):
            p = prompt_mean_from_subsample(samples_by_task, n, rng)
            c = prompt_mean_from_subsample(samples_by_task, n, rng)
            if c > p + epsilon:
                accepts += 1
        rows.append((n, accepts / n_boot))
    return rows


def required_n_for_effect(samples_by_task, target, n_boot, rng, n_max=200):
    """Smallest N such that a child with +target true effect has its 95% CI
    lower bound above the parent's 95% CI upper bound (CIs separate).
    Child modeled as parent samples shifted by +target, clipped to [0,1]."""
    shifted = {tid: [min(1.0, r + target) for r in rs]
               for tid, rs in samples_by_task.items()}
    n = 1
    while n <= n_max:
        pm = [prompt_mean_from_subsample(samples_by_task, n, rng) for _ in range(n_boot)]
        cm = [prompt_mean_from_subsample(shifted, n, rng) for _ in range(n_boot)]
        pm.sort(); cm.sort()
        p_hi = pm[int(n_boot * 0.975)]
        c_lo = cm[int(n_boot * 0.025)]
        if c_lo > p_hi:
            return n
        n = n + 1 if n < 10 else n + 5
    return None  # not separable within n_max


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_json")
    ap.add_argument("--target-effect", type=float, default=0.15)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = load_per_task_samples(args.result_json)
    if not samples:
        print("no per-task sample_rewards found", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)

    n_avail = min(len(v) for v in samples.values())
    print(f"\n=== Variance analysis: {Path(args.result_json).name} ===")
    print(f"tasks={len(samples)}  samples/task available={n_avail}\n")
    print("Per-task observed sample rewards:")
    for tid, rs in samples.items():
        app = tid.split("/")[0]
        mean = sum(rs) / len(rs)
        sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0
        print(f"  {app:18} n={len(rs)} mean={mean:.3f} sd={sd:.3f} rewards={rs}")

    ns = [n for n in (1, 3, 5, 8, 10) if n <= n_avail]

    print(f"\n=== (1) Prompt-mean 95% CI half-width vs N (bootstrap {args.n_boot}) ===")
    for n, m, hw, lo, hi in ci_halfwidth_vs_n(samples, ns, args.n_boot, rng):
        print(f"  N={n:2}  mean={m:.3f}  95%CI=[{lo:.3f},{hi:.3f}]  half-width=±{hw:.3f}")

    print(f"\n=== (2) FALSE-PROMOTION probability vs N (null: no true effect) ===")
    print("    P(a noise-only 'child' beats parent under strict-greater accept):")
    for n, p in false_promotion_prob(samples, ns, args.n_boot, rng):
        bar = "#" * int(p * 50)
        print(f"  N={n:2}  P(false accept)={p:.3f}  {bar}")

    print(f"\n=== (3) Required N to reliably detect a +{args.target_effect} effect ===")
    req = required_n_for_effect(samples, args.target_effect, args.n_boot, rng)
    if req is None:
        print(f"  NOT separable within N=200 — effect +{args.target_effect} is")
        print(f"  smaller than the irreducible per-prompt noise at feasible N.")
    else:
        print(f"  N={req} samples/task needed for parent & child 95% CIs to separate.")
        n_tasks = len(samples)
        print(f"  → decisive test cost: 2 prompts × {n_tasks} tasks × {req} = "
              f"{2 * n_tasks * req} rollouts (~${2*n_tasks*req*0.18:.0f} @ $0.18/rollout)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
