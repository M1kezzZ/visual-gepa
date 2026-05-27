"""FCVR operator — Failure-Clustered Visual Reflection.

Signature:
    FCVR_B : (F, P_t) -> {patch_k}_{k=1..K}
    where B = (K=4, J=3, M=2, T_patch=512 tokens)

This is the dominant contribution of Visual-GEPA. Everything else (Qwen3.5-9B,
GEPA Pareto manager, OSWorld env loop, CLIP embedder) is frozen and reused.

Pipeline (one call):
  1. Embed failed trajectories (CLIP per frame → mean-pool per traj).
  2. Cluster into ≤ K clusters (KMeans on per-traj vectors).
  3. Pick M representative trajectories per cluster (closest-to-centroid).
  4. Per-trajectory: action_boundaries + MMR → J key frames.
  5. ONE Claude vision reflection call per cluster → ONE schema-valid FCVRPatch.

Token-budget invariant:
    T_total(FCVR) ≤ K · T_one_vanilla_GEPA_reflection_call
    With K=4 and exactly one VLM call per cluster, honest by construction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .key_frame import action_boundaries, mmr_select
from .patch_schema import FCVRPatch
from .structured_prompt import StructuredPrompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FCVRBudget:
    K: int = 4           # cluster count
    J: int = 3           # key frames per trajectory
    M: int = 2           # trajectories per cluster
    T_patch: int = 512   # max patch output tokens


DEFAULT_BUDGET = FCVRBudget()


@dataclass
class FCVRRunRecord:
    """Per-FCVR-call record. Persisted to results/ for audit."""
    n_failed_input: int
    n_clusters_used: int
    cluster_sizes: list[int] = field(default_factory=list)
    patches: list[dict[str, Any]] = field(default_factory=list)
    reflection_stats: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_s: float = 0.0
    schema_violations_total: int = 0
    elapsed_s: float = 0.0
    # Cluster-quality diagnostics (added 2026-05-27 per Codex post-run audit).
    # Distinguishes "failure space is genuinely homogeneous" (acceptable —
    # patches will all read similar) from "KMeans collapsed onto a single
    # dominant visual cluster on CLIP" (NOT acceptable — K is over-specified).
    silhouette_score: float | None = None  # None if n_samples<=K or K<2
    cluster_membership_by_app: list[dict[str, int]] = field(default_factory=list)
    centroid_pairwise_distances: list[list[float]] = field(default_factory=list)
    action_edit_distance_within_cluster: list[dict[str, float]] = field(default_factory=list)


def _app_of(task_id: str) -> str:
    return (task_id or "").split("/", 1)[0] or "unknown"


def _levenshtein_tokens(a: list[str], b: list[str]) -> int:
    """Edit distance over token sequences (insert/delete/substitute=1)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[lb]


def _kmeans_clusters(
    vectors: np.ndarray,
    k: int,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap sklearn KMeans; returns (labels, centroids).

    Falls back to a single-cluster assignment if N < k or vectors degenerate.
    """
    from sklearn.cluster import KMeans

    n = vectors.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros((0, vectors.shape[1]), dtype=np.float32)
    k_eff = max(1, min(k, n))
    if k_eff == 1:
        labels = np.zeros(n, dtype=int)
        centroid = vectors.mean(axis=0, keepdims=True)
        return labels, centroid

    km = KMeans(n_clusters=k_eff, n_init=10, random_state=rng_seed)
    labels = km.fit_predict(vectors)
    centroids = km.cluster_centers_
    return labels, centroids


def _pick_reps_for_cluster(
    vectors: np.ndarray, member_indices: np.ndarray, centroid: np.ndarray, m: int
) -> list[int]:
    if member_indices.size == 0:
        return []
    cluster_vecs = vectors[member_indices]
    dists = np.linalg.norm(cluster_vecs - centroid[None, :], axis=1)
    order = np.argsort(dists)
    keep = member_indices[order[: max(1, m)]]
    return keep.tolist()


class FCVROperator:
    """One budget-constrained failure-trace compression operator.

    Args:
        budget: FCVR hyperparameters.
        clip_embedder: callable that exposes .encode_trajectory_frames(traj) and
            .encode_trajectory_means([trajs]).
        reflection_client: a ClaudeReflectionClient (or compatible) with
            .reflect_cluster(cluster_id, trajs, key_frames_per_traj,
            action_summaries, parent_prompt, n_frames_per_traj).
        rng_seed: KMeans seeding.
    """

    def __init__(
        self,
        budget: FCVRBudget = DEFAULT_BUDGET,
        clip_embedder=None,
        reflection_client=None,
        rng_seed: int = 42,
    ) -> None:
        if clip_embedder is None:
            raise ValueError("clip_embedder is required (use CLIPImageEmbedder).")
        if reflection_client is None:
            raise ValueError("reflection_client is required (use ClaudeReflectionClient).")
        self.budget = budget
        self.clip_embedder = clip_embedder
        self.reflection_client = reflection_client
        self.rng_seed = rng_seed

    # ------------------------------------------------------------------
    def run(
        self,
        failed_trajectories: list,
        parent_prompt: StructuredPrompt,
    ) -> tuple[list[FCVRPatch], FCVRRunRecord]:
        """Main entry point. Returns up to `K` schema-valid patches + audit record."""
        t_start = time.perf_counter()
        record = FCVRRunRecord(
            n_failed_input=len(failed_trajectories),
            n_clusters_used=0,
        )

        if not failed_trajectories:
            record.elapsed_s = time.perf_counter() - t_start
            logger.info("FCVR: no failed trajectories; returning 0 patches.")
            return [], record

        # 1) Per-trajectory mean-pool CLIP embeddings.
        traj_vecs = self.clip_embedder.encode_trajectory_means(failed_trajectories)
        if traj_vecs.size == 0:
            record.elapsed_s = time.perf_counter() - t_start
            logger.warning("FCVR: clip_embedder returned empty matrix; no patches.")
            return [], record

        # 2) Cluster into K clusters (capped at #trajs).
        labels, centroids = _kmeans_clusters(traj_vecs, k=self.budget.K, rng_seed=self.rng_seed)
        k_eff = int(centroids.shape[0])
        record.n_clusters_used = k_eff
        cluster_sizes = [int((labels == c).sum()) for c in range(k_eff)]
        record.cluster_sizes = cluster_sizes
        logger.info("FCVR: %d failed → %d clusters sizes=%s", len(failed_trajectories), k_eff, cluster_sizes)

        # 2b) Cluster-quality diagnostics (Codex post-B2-mini action item).
        # Silhouette: requires k_eff >= 2 AND n_samples > k_eff.
        if k_eff >= 2 and traj_vecs.shape[0] > k_eff:
            try:
                from sklearn.metrics import silhouette_score
                record.silhouette_score = float(silhouette_score(traj_vecs, labels))
            except Exception as e:  # noqa: BLE001
                logger.warning("silhouette_score failed: %s", e)
                record.silhouette_score = None
        # Per-cluster app membership (parses task_id → "<app>/<uuid>").
        record.cluster_membership_by_app = []
        for c in range(k_eff):
            member_idx = np.where(labels == c)[0]
            app_counts: dict[str, int] = {}
            for i in member_idx:
                app = _app_of(getattr(failed_trajectories[int(i)], "task_id", ""))
                app_counts[app] = app_counts.get(app, 0) + 1
            record.cluster_membership_by_app.append(app_counts)
        # Pairwise centroid distances (K×K, symmetric, zero diagonal).
        record.centroid_pairwise_distances = [
            [
                float(np.linalg.norm(centroids[i] - centroids[j]))
                for j in range(k_eff)
            ]
            for i in range(k_eff)
        ]
        # Action-trace edit distance within each cluster.
        # Tokenize action by `parse_action` output stripped of arg whitespace —
        # i.e. the same string the rollout's repeated-actions early-stop sees.
        record.action_edit_distance_within_cluster = []
        for c in range(k_eff):
            member_idx = np.where(labels == c)[0]
            seqs = [
                [(s.action or "").strip() for s in failed_trajectories[int(i)].steps]
                for i in member_idx
            ]
            if len(seqs) < 2:
                record.action_edit_distance_within_cluster.append({
                    "cluster_id": int(c),
                    "n_pairs": 0,
                    "mean_edit_distance": 0.0,
                    "max_edit_distance": 0.0,
                    "mean_len": float(len(seqs[0])) if seqs else 0.0,
                })
                continue
            dists = []
            for i in range(len(seqs)):
                for j in range(i + 1, len(seqs)):
                    dists.append(_levenshtein_tokens(seqs[i], seqs[j]))
            record.action_edit_distance_within_cluster.append({
                "cluster_id": int(c),
                "n_pairs": len(dists),
                "mean_edit_distance": float(sum(dists) / len(dists)),
                "max_edit_distance": float(max(dists)),
                "mean_len": float(sum(len(s) for s in seqs) / len(seqs)),
            })

        patches: list[FCVRPatch] = []
        for c in range(k_eff):
            member_idx = np.where(labels == c)[0]
            if member_idx.size == 0:
                continue
            reps_idx = _pick_reps_for_cluster(traj_vecs, member_idx, centroids[c], self.budget.M)
            cluster_trajs = [failed_trajectories[i] for i in reps_idx]

            # 3) per-trajectory key frames (action_boundaries + MMR).
            cluster_key_frames: list[list] = []
            cluster_action_summaries: list[str] = []
            for traj in cluster_trajs:
                frame_embs = self.clip_embedder.encode_trajectory_frames(traj)
                seeds = action_boundaries(traj)
                idxs = mmr_select(frame_embs, seeds, budget=self.budget.J)
                frames = [traj.steps[i].screenshot for i in idxs if 0 <= i < len(traj.steps)]
                cluster_key_frames.append(frames)
                action_str = " | ".join(
                    f"{i + 1}.{(s.action.split('#')[0]).strip()[:48]}"
                    for i, s in enumerate(traj.steps)
                )
                cluster_action_summaries.append(action_str)

            # 4) ONE Claude reflection call per cluster.
            patch, stats = self.reflection_client.reflect_cluster(
                cluster_id=c,
                cluster_trajectories=cluster_trajs,
                cluster_key_frames=cluster_key_frames,
                cluster_action_summaries=cluster_action_summaries,
                parent_prompt=parent_prompt,
                n_frames_per_traj=self.budget.J,
            )
            record.reflection_stats.append(
                {
                    "cluster_id": c,
                    "cluster_size": cluster_sizes[c],
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "cache_input_tokens": stats.cache_input_tokens,
                    "latency_s": round(stats.latency_s, 3),
                    "schema_violations": stats.schema_violations,
                    "error": stats.error,
                }
            )
            record.total_input_tokens += stats.input_tokens
            record.total_output_tokens += stats.output_tokens
            record.total_latency_s += stats.latency_s
            record.schema_violations_total += stats.schema_violations

            if patch is not None:
                patches.append(patch)
                record.patches.append(patch.model_dump())
                logger.info(
                    "FCVR cluster %d → patch failure_pattern=%r",
                    c,
                    patch.failure_pattern[:80],
                )
            else:
                logger.warning("FCVR cluster %d → NO patch (schema violation or API error).", c)

        record.elapsed_s = time.perf_counter() - t_start
        return patches, record
