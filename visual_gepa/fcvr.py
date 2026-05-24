"""FCVR operator — Failure-Clustered Visual Reflection.

Signature:
    FCVR_B : (F, P_t) -> {patch_k}_{k=1..K}
    where B = (K=4, J=3, M=2, T_patch=512 tokens)

This is the dominant contribution of Visual-GEPA. Everything else (Qwen3.5-9B,
GEPA Pareto manager, OSWorld env loop, CLIP embedder) is frozen and reused.

Stub — implement during Day-1 / B0 (see SETUP.md Step 6).
"""

from __future__ import annotations

from dataclasses import dataclass

from .patch_schema import FCVRPatch
from .structured_prompt import StructuredPrompt


@dataclass(frozen=True)
class FCVRBudget:
    K: int = 4              # cluster count
    J: int = 3              # key frames per trajectory
    M: int = 2              # trajectories per cluster
    T_patch: int = 512      # max patch tokens


DEFAULT_BUDGET = FCVRBudget()


class FCVROperator:
    """One budget-constrained failure-trace compression operator.

    Token-budget constraint:
        T_total(FCVR) <= K * T_one_vanilla_GEPA_reflection_call
    With K=4 and exactly one VLM call per cluster, this is honest by construction.
    """

    def __init__(
        self,
        budget: FCVRBudget = DEFAULT_BUDGET,
        clip_embedder=None,
        text_embedder=None,
        reflection_client=None,
    ) -> None:
        self.budget = budget
        self.clip_embedder = clip_embedder
        self.text_embedder = text_embedder
        self.reflection_client = reflection_client

    def run(self, failed_trajectories, parent_prompt: StructuredPrompt) -> list[FCVRPatch]:
        """Main entry point. Returns up to K structured patches.

        Stages:
          1. Embed each failed trajectory  (CLIP + text)
          2. Cluster into K clusters       (KMeans)
          3. Key-frame select              (deterministic MMR + action-boundary)
          4. Reflect once per cluster      (ONE VLM call; emits one FCVRPatch)
        """
        raise NotImplementedError("Implement during Day-1 / B0 smoke test.")
