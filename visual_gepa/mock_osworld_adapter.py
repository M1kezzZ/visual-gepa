"""Mock OSWorld adapter — synthetic but realistic trajectories for B0 plumbing.

WHY THIS EXISTS
---------------
B0 is an end-to-end *plumbing* check (see EXPERIMENT_PLAN.md §3 B0). Real
OSWorld needs KVM-capable hosts or a cloud provider (AWS/AliYun). Our RTX 5090
AutoDL container has neither. So we mock the env loop ONLY — every other
component (vLLM Qwen3.5-9B backbone, CLIP embedding, KMeans clustering, MMR
key-frame selection, Claude Opus 4.7 vision reflection, FCVRPatch schema,
structured-prompt merge, GEPA Pareto loop) runs end-to-end for real.

CONTRACT
--------
The adapter returns deterministic `MultimodalTrajectory` objects whose
screenshots are programmatically rendered "fake desktop" PIL images that CLIP
can geometrically distinguish — so MMR + KMeans behavior is non-degenerate.

For B1 we swap this for a real `OSWorldAdapter` that talks to an OSWorld
DesktopEnv (Docker / AWS / AliYun provider). The interface is identical so
nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .osworld_adapter import MultimodalStep, MultimodalTrajectory
from .structured_prompt import StructuredPrompt

logger = logging.getLogger(__name__)


def _stable_seed(*parts: object) -> int:
    """Cross-process-stable 32-bit int seed (Python `hash()` is PYTHONHASHSEED-salted)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:4], "big")


# Per-task scripted action sequences. Each entry: a list of (action_str, feedback, mark_failed).
# These are intentionally diverse in length and action mix so that:
#   - action_boundaries() flags different positions per traj
#   - KMeans on mean-pooled CLIP embeddings produces > 1 cluster
#   - At least 3 tasks are scripted as failing → FCVR has material to reflect on
_TASK_SCRIPTS: dict[str, dict[str, Any]] = {
    "libreoffice_calc_pivot": {
        "instruction": (
            "Create a Pivot Table in a new sheet (Sheet2) to count how many "
            'times each "Invoice No." appears.'
        ),
        "app": "libreoffice_calc",
        "aliases": ["libreoffice", "calc", "pivot", "spreadsheet", "sheet"],
        "color": (200, 220, 240),
        "title": "LibreOffice Calc — Invoices.xlsx",
        "actions": [
            ("pyautogui.click(x=120, y=80)  # Sheet1 tab", "click registered on Sheet1 tab", False),
            ("pyautogui.click(x=350, y=200)  # cell A1", "selected cell A1", False),
            ("pyautogui.hotkey('ctrl', 'a')", "all cells selected", False),
            ("pyautogui.click(x=140, y=40)  # Data menu", "Data menu opened", False),
            ("pyautogui.click(x=160, y=160)  # Pivot Table item", "dialog opened: Pivot Table Layout", False),
            ("pyautogui.click(x=600, y=400)  # drag Invoice No.", "no effect, dragged onto wrong field", True),
            ("pyautogui.click(x=300, y=520)  # OK", "Pivot table inserted on Sheet1 (NOT Sheet2)", False),
            ("FAIL", "pivot exists but not in Sheet2 as required; failed to satisfy task", True),
        ],
        "expected_failure": True,
    },
    "chrome_tab_close_dialog": {
        "instruction": "Close the modal dialog blocking google.com and search 'arxiv 2507.19457'.",
        "app": "chrome",
        "aliases": ["chrome", "browser", "dialog", "modal", "google", "popup"],
        "color": (240, 240, 240),
        "title": "Chrome — Google",
        "actions": [
            ("pyautogui.click(x=400, y=300)  # blocked by dialog", "no effect, dialog blocks input", True),
            ("pyautogui.click(x=400, y=300)", "no effect, dialog blocks input", True),
            ("pyautogui.click(x=400, y=300)", "no effect, dialog blocks input", True),
            ("pyautogui.click(x=400, y=300)", "no effect, dialog blocks input", True),
            ("FAIL", "8 retries elapsed without dismissing dialog", True),
        ],
        "expected_failure": True,
    },
    "vlc_play_video": {
        "instruction": "Open the video file Bunny.mp4 from ~/Videos and play it.",
        "app": "vlc",
        "aliases": ["vlc", "video", "media", "play", "movie", "player"],
        "color": (40, 40, 60),
        "title": "VLC media player",
        "actions": [
            ("pyautogui.click(x=80, y=40)  # Media menu", "Media menu opened", False),
            ("pyautogui.click(x=120, y=80)  # Open File...", "file picker opened", False),
            ("pyautogui.typewrite('~/Videos/Bunny.mp4')", "path typed", False),
            ("pyautogui.press('enter')", "video loaded", False),
            ("pyautogui.click(x=20, y=560)  # play", "playback started", False),
            ("DONE", "task completed: video playing", False),
        ],
        "expected_failure": False,
    },
    "file_manager_rename": {
        "instruction": "Rename ~/Documents/report.txt to ~/Documents/report_final.txt.",
        "app": "files",
        "aliases": ["file", "files", "rename", "explorer", "manager", "document"],
        "color": (250, 245, 230),
        "title": "Files — Documents",
        "actions": [
            ("pyautogui.doubleclick(x=200, y=240)  # report.txt", "file selected and opened", True),
            ("pyautogui.hotkey('alt', 'F4')  # close opened preview by accident", "preview window closed", False),
            ("pyautogui.rightclick(x=200, y=240)", "context menu opened", False),
            ("pyautogui.click(x=240, y=280)  # Rename", "rename inline edit active", False),
            ("pyautogui.hotkey('ctrl', 'a')", "name highlighted", False),
            ("pyautogui.typewrite('report_final.txt')", "new name typed", False),
            ("pyautogui.press('enter')", "rename confirmed", False),
            ("DONE", "task completed: file renamed", False),
        ],
        "expected_failure": False,
    },
    "gimp_export_png": {
        "instruction": "Export the open GIMP image as ~/Pictures/out.png with default settings.",
        "app": "gimp",
        "aliases": ["gimp", "export", "png", "image", "picture", "edit"],
        "color": (60, 60, 60),
        "title": "GIMP — Untitled.xcf",
        "actions": [
            ("pyautogui.click(x=40, y=40)  # File menu", "File menu opened", False),
            ("pyautogui.click(x=80, y=200)  # Export As...", "Export As dialog opened: new dialog", False),
            ("pyautogui.typewrite('~/Pictures/out.png')", "path typed", False),
            ("pyautogui.click(x=440, y=520)  # Export button", "PNG export options dialog opened: new dialog", False),
            ("pyautogui.click(x=300, y=520)  # Export with defaults", "no effect, button missed by 20px", True),
            ("pyautogui.click(x=300, y=520)", "no effect, dialog still open", True),
            ("FAIL", "button never clicked — task incomplete", True),
        ],
        "expected_failure": True,
    },
}


def _try_font(size: int = 18):
    """Best-effort load a font; fall back to PIL default if none available."""
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_fake_screenshot(
    width: int,
    height: int,
    bg: tuple[int, int, int],
    title: str,
    step_idx: int,
    n_steps: int,
    action: str,
    feedback: str,
    rng: random.Random,
) -> Image.Image:
    """Render a deterministic-given-rng PIL screenshot mimicking a desktop app.

    Important: the visual layout depends on `rng` and step content so CLIP
    embeddings are *not* collapsed across frames or across trajectories.
    """
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Title bar
    title_font = _try_font(20)
    body_font = _try_font(14)
    draw.rectangle([0, 0, width, 32], fill=(70, 90, 120))
    draw.text((12, 6), title, fill=(255, 255, 255), font=title_font)

    # "Toolbar"
    draw.rectangle([0, 32, width, 56], fill=(180, 195, 210))
    for i in range(6):
        x = 16 + i * 70
        draw.rectangle([x, 38, x + 50, 50], fill=(150, 165, 180))

    # Pseudo content — colored rectangles that depend on step content.
    # `_stable_seed` uses sha256 so the same (action, feedback, step) always
    # produces the same image across processes / runs.
    seed = _stable_seed("frame", action, feedback, step_idx)
    rng2 = random.Random(seed)
    for _ in range(20):
        x = rng2.randint(8, width - 80)
        y = rng2.randint(70, height - 80)
        w = rng2.randint(40, 200)
        h = rng2.randint(20, 80)
        c = (rng2.randint(60, 250), rng2.randint(60, 250), rng2.randint(60, 250))
        draw.rectangle([x, y, x + w, y + h], outline=(40, 40, 40), fill=c)

    # Footer with step info — keeps the image text-distinguishable.
    draw.rectangle([0, height - 60, width, height], fill=(240, 240, 240))
    draw.text((10, height - 56), f"step {step_idx + 1}/{n_steps}", fill=(40, 40, 40), font=body_font)
    draw.text((10, height - 38), f"action: {action[:60]}", fill=(40, 40, 40), font=body_font)
    draw.text((10, height - 22), f"obs: {feedback[:60]}", fill=(40, 40, 40), font=body_font)

    return img


@dataclass
class MockOSWorldAdapter:
    """Drop-in mock replacement for `OSWorldAdapter`.

    Same interface (`run`, `metric`) so B0_smoke can swap in the real impl later.
    """

    task_id: str
    vllm_endpoint: str = "http://localhost:8000/v1"
    screenshot_size: tuple[int, int] = (640, 480)
    rng_seed: int = 42

    def __post_init__(self) -> None:
        if self.task_id not in _TASK_SCRIPTS:
            raise ValueError(
                f"task_id={self.task_id!r} not in mock script. Known: "
                f"{list(_TASK_SCRIPTS)}"
            )
        self.script = _TASK_SCRIPTS[self.task_id]

    # ------------------------------------------------------------------
    def run(self, prompt: StructuredPrompt) -> MultimodalTrajectory:
        """Roll out one mock episode.

        Determinism: bound to (task_id, rng_seed, prompt-token-length). The
        prompt influences whether some scripted failures get "fixed" — this is
        how B0 verifies that BEHAVIORAL_PATCHES are being read by the agent
        loop. We simulate by checking if the parent prompt contains a "fix"
        substring matched to this task.
        """
        rng = random.Random(_stable_seed("rollout", self.task_id, self.rng_seed))
        steps_def = list(self.script["actions"])

        # "Patches help" signal — any behavioral patch whose scope_guard OR
        # prompt_diff mentions ANY alias for this task is treated as helpful.
        # Aliases cover app name, task verb, file-type, common UI words. This
        # makes the synthetic learning signal robust to Claude's free-form
        # natural-language guards (the codex review flagged that the previous
        # strict prefix match could miss matches).
        aliases = [a.lower() for a in self.script.get("aliases", [])]
        agent_helped = False
        for sg, pd in prompt.behavioral_patches:
            blob = ((sg or "") + " " + (pd or "")).lower()
            if any(a in blob for a in aliases):
                agent_helped = True
                break

        steps: list[MultimodalStep] = []
        n = len(steps_def)
        for i, (action_str, feedback_str, mark_failed) in enumerate(steps_def):
            screenshot = _render_fake_screenshot(
                self.screenshot_size[0],
                self.screenshot_size[1],
                self.script["color"],
                self.script["title"],
                i,
                n,
                action_str,
                feedback_str,
                rng,
            )
            # If patches help and this was a failed step, soften the feedback.
            soft_feedback = feedback_str
            if agent_helped and mark_failed:
                soft_feedback = "(patched) " + feedback_str.replace("no effect", "took effect")
            steps.append(
                MultimodalStep(
                    action=action_str,
                    screenshot=screenshot,
                    accessibility_tree=f"<app name='{self.script['app']}' step={i}/>",
                    reward=None,
                    feedback=soft_feedback,
                )
            )

        # Final reward: succeeded if not flagged as expected failure OR if
        # the agent was "helped" by an existing patch.
        succeeded = (not self.script["expected_failure"]) or agent_helped
        return MultimodalTrajectory(
            task_id=self.task_id,
            instruction=self.script["instruction"],
            steps=steps,
            final_reward=1.0 if succeeded else 0.0,
        )

    def metric(self, traj: MultimodalTrajectory) -> tuple[float, str]:
        score = float(traj.final_reward)
        action_summary = " | ".join(
            f"{i + 1}.{(s.action.split('#')[0]).strip()[:48]}"
            for i, s in enumerate(traj.steps)
        )
        terminal = traj.steps[-1].feedback if traj.steps else "(no steps)"
        feedback_str = (
            f"task_id={traj.task_id} succeeded={traj.succeeded} "
            f"n_steps={traj.n_steps}\n"
            f"action_trace_summary: {action_summary}\n"
            f"terminal_feedback: {terminal}"
        )
        return score, feedback_str


def known_task_ids() -> list[str]:
    """Return the 5 mock task ids — used by B0 to enumerate tasks."""
    return list(_TASK_SCRIPTS.keys())
