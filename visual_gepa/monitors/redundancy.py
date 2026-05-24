"""Patch-accumulation overlap monitor.

The residual failure mode flagged by Round 4 of /research-refine is overlapping
"When X: do Y" rules in BEHAVIORAL_PATCHES accumulating with iterations.

Mitigation:
  - Any two patches with `scope_guard` cosine similarity ≥ 0.85 are flagged.
  - The lower-impact patch (by leave-one-out val-success delta) is dropped.
  - Per-iter `len(P_t)` and `|BEHAVIORAL_PATCHES|` are logged.

Stub — implement before B2 (see SETUP.md Step 6 pitfalls table).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatchRedundancyMonitor:
    cosine_threshold: float = 0.85
    drop_lower_impact: bool = True

    def scan(self, behavioral_patches, text_embedder, val_scores_per_patch) -> list[int]:
        """Return indices of patches to drop.

        Args:
            behavioral_patches: list of (scope_guard, prompt_diff) tuples.
            text_embedder: callable str -> ndarray, for scope_guard embeddings.
            val_scores_per_patch: dict mapping patch_idx -> leave-one-out val Δ.

        Returns:
            Indices (into behavioral_patches) to drop.
        """
        raise NotImplementedError("Implement before B2.")
