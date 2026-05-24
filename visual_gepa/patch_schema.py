"""5-field structured patch schema for FCVR reflection output.

Each FCVR cluster produces one patch. Patches are accepted into BEHAVIORAL_PATCHES
only if (i) schema-valid AND (ii) the child candidate survives one GEPA rollout
(not pruned by Pareto dominance).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FCVRPatch(BaseModel):
    """One patch emitted by FCVR per failure cluster."""

    failure_pattern: str = Field(
        ...,
        description="One-sentence VLM-named description of what this cluster's failures share.",
        min_length=4,
        max_length=512,
    )
    visual_evidence: str = Field(
        ...,
        description="Quotes / references to key frames that justify the failure_pattern.",
        min_length=4,
        max_length=1024,
    )
    prompt_diff: str = Field(
        ...,
        description="The additive instruction text. Becomes part of BEHAVIORAL_PATCHES.",
        min_length=4,
        max_length=512,
    )
    scope_guard: str = Field(
        ...,
        description=(
            "ONE-LINE natural-language condition. Inserted in front of prompt_diff at "
            "inference. NOT a runtime router — no parser, no executable guard."
        ),
        min_length=4,
        max_length=256,
    )
    expected_behavior_change: str = Field(
        ...,
        description="One-sentence prediction of how the agent's behavior should change.",
        min_length=4,
        max_length=512,
    )

    model_config = {"frozen": True, "extra": "forbid"}
