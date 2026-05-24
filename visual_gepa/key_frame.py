"""Deterministic MMR + action-boundary key-frame selection.

NO VLM call inside FCVR's key-frame selection. This is the choice that makes
the token-budget constraint honest by construction.

  KF_{k,i} = MMR_select(
    CLIP(frames(τ_i)),
    seeds = action_boundaries(τ_i),
    budget = J = 3
  )

  action_boundaries(τ_i) := union of frames at which any of the following holds:
    - the action type changes (e.g. click → type)
    - a click is observed to be a no-op or failed action
    - a page / window / focus transition occurs
    - the terminal two steps of the trajectory.

Stub — implement during Day-1 (see SETUP.md Step 6).
"""

from __future__ import annotations


def action_boundaries(trajectory) -> list[int]:
    """Return frame indices that are candidate seeds for MMR.

    Implementation outline:
      1. Iterate steps in order.
      2. Flag a frame index when:
         - step.action.kind != prev_step.action.kind
         - step.feedback indicates a failed click / no-op
         - step.observation indicates a page/window/focus change
      3. Always include the last two step indices.
      4. Deduplicate while preserving order.
    """
    raise NotImplementedError("Implement during Day-1.")


def mmr_select(
    frame_embeddings,
    seeds: list[int],
    budget: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal-Marginal-Relevance selection over CLIP frame embeddings.

    Args:
        frame_embeddings: (T, D) tensor / ndarray of CLIP embeddings, one per frame.
        seeds: starting indices from action_boundaries; MMR starts seeded with these.
        budget: number of frames to return (J = 3 by default).
        lambda_: trade-off between diversity (low) and seed-similarity (high).

    Returns:
        Sorted list of selected frame indices (length == budget).

    Reference: Carbonell & Goldstein (1998). For our case the "query" of MMR is
    the centroid of the seeds, and "diversity" is computed over already-selected
    frames.
    """
    raise NotImplementedError("Implement during Day-1.")
