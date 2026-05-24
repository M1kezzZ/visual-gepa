"""Named-section structured prompt.

  P_t = [PERSONA] || [GLOBAL_RULES] || [BEHAVIORAL_PATCHES] || [TASK_SCAFFOLD]

Each FCVR patch is appended to BEHAVIORAL_PATCHES as a (scope_guard, prompt_diff)
tuple. At inference the agent reads:

    [PERSONA]
    ...
    [GLOBAL_RULES]
    ...
    [BEHAVIORAL_PATCHES]
    When <scope_guard_1>: <prompt_diff_1>
    When <scope_guard_2>: <prompt_diff_2>
    ...
    [TASK_SCAFFOLD]
    ...

scope_guard is a natural-language condition, NOT a runtime router. There is no
parser or executable guard — the agent reads the guard alongside the patch and
applies it only when the guard holds, identical to human-written "if X, do Y"
rules in system prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StructuredPrompt:
    persona: str = ""
    global_rules: str = ""
    behavioral_patches: list[tuple[str, str]] = field(default_factory=list)
    task_scaffold: str = ""

    def append_patch(self, scope_guard: str, prompt_diff: str) -> None:
        self.behavioral_patches.append((scope_guard, prompt_diff))

    def token_length(self, tokenizer=None) -> int:
        """Approximate token length for the parsimony tie-breaker.

        If a tokenizer is provided, use it. Otherwise approximate by 4 chars/token.
        """
        text = self.render()
        if tokenizer is None:
            return max(1, len(text) // 4)
        return len(tokenizer.encode(text))

    def render(self) -> str:
        if self.behavioral_patches:
            patch_block = "\n".join(
                f"When {sg}: {pd}" for sg, pd in self.behavioral_patches
            )
        else:
            patch_block = "(none yet)"

        return "\n\n".join(
            [
                f"[PERSONA]\n{self.persona}".rstrip(),
                f"[GLOBAL_RULES]\n{self.global_rules}".rstrip(),
                f"[BEHAVIORAL_PATCHES]\n{patch_block}",
                f"[TASK_SCAFFOLD]\n{self.task_scaffold}".rstrip(),
            ]
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()
