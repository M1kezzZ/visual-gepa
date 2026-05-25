"""Visual-GEPA — Failure-Clustered Visual Reflection for prompt evolution.

Public exports (stable across B0 → B7):
  - FCVROperator, FCVRBudget, DEFAULT_BUDGET (visual_gepa.fcvr)
  - FCVRPatch                                 (visual_gepa.patch_schema)
  - StructuredPrompt                          (visual_gepa.structured_prompt)
  - CLIPImageEmbedder                         (visual_gepa.clip_embedder)
  - ClaudeReflectionClient                    (visual_gepa.reflection)
  - OSWorldAdapter, MultimodalTrajectory,
    MultimodalStep                            (visual_gepa.osworld_adapter)
"""

__version__ = "0.0.1"

from .fcvr import DEFAULT_BUDGET, FCVRBudget, FCVROperator
from .osworld_adapter import MultimodalStep, MultimodalTrajectory, OSWorldAdapter
from .patch_schema import FCVRPatch
from .structured_prompt import StructuredPrompt

__all__ = [
    "DEFAULT_BUDGET",
    "FCVRBudget",
    "FCVROperator",
    "FCVRPatch",
    "MultimodalStep",
    "MultimodalTrajectory",
    "OSWorldAdapter",
    "StructuredPrompt",
    "__version__",
]
