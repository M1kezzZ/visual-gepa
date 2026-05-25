"""OSWorld env ↔ Visual-GEPA bridge.

This adapter is the highest-effort piece of the Day-1 integration (budget 3-5 days,
per SETUP.md Step 6). The interfaces below are placeholders; the actual env-loop
implementation depends on the OSWorld version pinned in third_party/OSWorld.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .structured_prompt import StructuredPrompt


@dataclass
class MultimodalStep:
    action: str
    screenshot: Image.Image
    accessibility_tree: str
    reward: float | None = None
    feedback: str = ""


@dataclass
class MultimodalTrajectory:
    task_id: str
    instruction: str
    steps: list[MultimodalStep] = field(default_factory=list)
    final_reward: float = 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def succeeded(self) -> bool:
        return self.final_reward > 0


class OSWorldAdapter:
    """Adapts OSWorld env to GEPA's expected interface."""

    def __init__(self, task_id: str, vllm_endpoint: str) -> None:
        self.task_id = task_id
        self.vllm_endpoint = vllm_endpoint
        # self.env = DesktopEnv(task_config=task_id)  # from third_party/OSWorld

    def run(self, prompt: StructuredPrompt) -> MultimodalTrajectory:
        """Roll out one episode under the given structured prompt."""
        raise NotImplementedError("Implement during Day-1.")

    def metric(self, traj: MultimodalTrajectory) -> tuple[float, str]:
        """Return (score, structured-feedback-string).

        The feedback string is what FCVR's reflection LM reads alongside the
        key-frame images. Must capture: terminal observation, error string,
        action-trace summary.
        """
        raise NotImplementedError("Implement during Day-1.")
