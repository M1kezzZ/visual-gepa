"""Real OSWorld env-loop adapter — drives a vLLM-served Qwen3.5-9B agent
against the OSWorld DesktopEnv (Docker provider) and produces a
`MultimodalTrajectory` in Visual-GEPA's format.

Interface contract — same `.run(prompt) → MultimodalTrajectory` shape as
`MockOSWorldAdapter`, so existing `scripts/B0_smoke.py` / future `B1_baseline.py`
orchestration code swaps in the real adapter unchanged.

Runtime requirements:
  - OSWorld submodule importable as `desktop_env.DesktopEnv`
    (third_party/OSWorld/desktop_env), with `provider_name='docker'`.
    Host must have **Docker + KVM** (vast.ai's container-only hosts won't work;
    we run this on a privileged 4090 instance with `/dev/kvm`).
  - vLLM OpenAI-compatible endpoint serving Qwen3.5-9B vision (native
    multimodal — early-fusion, not stitched).
  - OSWorld task config JSONs at
    `third_party/OSWorld/evaluation_examples/examples/<domain>/<uuid>.json`.

Out of scope for this adapter (deliberately):
  - GEPA Pareto candidate management (lives in `visual_gepa.fcvr` / GEPA core).
  - FCVR reflection (B2; uses the trajectories this adapter produces).
  - Cache (B2 implements per-task CLIP + trajectory caching).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .structured_prompt import StructuredPrompt

logger = logging.getLogger(__name__)


# --- Trajectory data classes (shared with mock adapter) --------------------
@dataclass
class MultimodalStep:
    action: str
    screenshot: Image.Image
    accessibility_tree: str
    reward: float | None = None
    feedback: str = ""
    # Raw model output BEFORE parse_action. Optional because synthetic /
    # scripted env adapters (e.g. MockOSWorldAdapter) don't have a model
    # call. Retained for the paper-grade audit chain: Qwen-text → parse →
    # action → env.step (per Codex audit of B1 baseline mini 2026-05-26).
    raw_model_text: str | None = None


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


# --- Action parsing --------------------------------------------------------
# Match (in order): triple-backtick python block; bare pyautogui.* line;
# special control tokens. Standard OSWorld PromptAgent format.
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.S | re.I)
_PYAUTOGUI_LINE_RE = re.compile(r"^\s*(pyautogui\.[A-Za-z_]+\(.*?\))\s*$", re.M)
_SPECIAL_TOKEN_RE = re.compile(r"\b(WAIT|DONE|FAIL)\b", re.I)
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.S | re.I)


def parse_action(response_text: str) -> str:
    """Extract an OSWorld-executable action string from the model's reply.

    Order of preference:
      1. Strip any `<think>...</think>` reasoning wrapper (Qwen3.5 leaks these
         even with `enable_thinking=False` per Codex review).
      2. Code block fenced by ``` (most common in trained models).
      3. Bare `pyautogui.X(...)` line (Qwen sometimes skips the fence).
      4. Special token WAIT / DONE / FAIL.
      5. Fall through → return raw text trimmed (let OSWorld surface the error).
    """
    # Strip <think> blocks first so they don't confuse later matchers.
    text = _THINK_TAG_RE.sub("", response_text).strip()

    m = _CODE_BLOCK_RE.search(text)
    if m:
        code = m.group(1).strip()
        if code:
            return code

    m = _PYAUTOGUI_LINE_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _SPECIAL_TOKEN_RE.search(text)
    if m:
        return m.group(1).upper()

    return text[:512]  # bounded — OSWorld will return a parse error if invalid


# --- vLLM client wrapper ---------------------------------------------------
def _pil_to_b64_png(img: Image.Image, max_side: int = 1280) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class _VLLMAgent:
    """Thin wrapper around an OpenAI-compatible vLLM endpoint.

    Disables Qwen's `<think>` mode for backbone use (per CLAUDE.md tech-stack
    note) and sends history as a list-of-messages with image content blocks.
    Holds the per-rollout chat history so the agent has multi-turn context.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        system_prompt: str,
        instruction: str,
        max_history_steps: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        # Local import — keeps test-time imports of visual_gepa free of openai.
        from openai import OpenAI

        self._client = OpenAI(base_url=endpoint, api_key="EMPTY")
        self.model = model
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.max_history_steps = max_history_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._history: list[dict] = []

    def step(self, screenshot: Image.Image, accessibility_tree: str) -> str:
        """Send the current observation; return raw model text."""
        # Truncate history to last `max_history_steps` user/assistant pairs.
        keep = self._history[-(2 * self.max_history_steps):]

        user_content = [
            {
                "type": "text",
                "text": (
                    f"task instruction: {self.instruction}\n"
                    f"accessibility tree (truncated): {accessibility_tree[:2000]}\n"
                    "Reply with ONE pyautogui action (fenced in ```python```), "
                    "or the literal token WAIT / DONE / FAIL."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_pil_to_b64_png(screenshot)}"},
            },
        ]

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + keep
            + [{"role": "user", "content": user_content}]
        )

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = resp.choices[0].message.content or ""

        # Add to history. Image tokens are expensive, so we drop the image
        # from history and keep only the action text.
        self._history.append(
            {
                "role": "user",
                "content": (
                    f"task instruction: {self.instruction}\n"
                    f"(screenshot omitted from history at step "
                    f"{len(self._history) // 2 + 1})"
                ),
            }
        )
        self._history.append({"role": "assistant", "content": text})
        return text


# --- Public adapter --------------------------------------------------------
DEFAULT_TASK_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "OSWorld" / "evaluation_examples" / "examples"


def load_osworld_task_config(task_config_path: str | Path) -> dict:
    """Load an OSWorld task config JSON (path or domain/uuid form)."""
    p = Path(task_config_path)
    if not p.suffix:
        # Allow "<domain>/<uuid>" shorthand.
        p = DEFAULT_TASK_ROOT / f"{task_config_path}.json"
    if not p.exists():
        raise FileNotFoundError(f"OSWorld task config not found: {p}")
    return json.loads(p.read_text())


class OSWorldAdapter:
    """Real OSWorld adapter — replace `MockOSWorldAdapter` once KVM host is up.

    Args:
      task_config_path: path to OSWorld task config JSON, OR `"<domain>/<uuid>"`
        shorthand resolved against `third_party/OSWorld/evaluation_examples/`.
      task_dict: alternatively, pass a task dict directly.
      vllm_endpoint: OpenAI-compatible URL (default localhost:8000).
      vllm_model: model id registered with vLLM.
      provider_name: OSWorld provider — `"docker"` is default; `"vmware"`,
        `"aws"`, `"aliyun"`, etc. also valid (see OSWorld README).
      os_type: `"Ubuntu"` or `"Windows"`.
      max_steps: hard cutoff for trajectory length (B1 spec says 30).
      headless: forward to DesktopEnv when supported.
    """

    def __init__(
        self,
        task_config_path: str | Path | None = None,
        task_dict: dict | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        vllm_model: str = "Qwen/Qwen3.5-9B",
        provider_name: str = "docker",
        os_type: str = "Ubuntu",
        max_steps: int = 30,
        headless: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        if task_config_path is None and task_dict is None:
            raise ValueError("provide either task_config_path or task_dict")
        if task_dict is None:
            task_dict = load_osworld_task_config(task_config_path)  # type: ignore[arg-type]

        self.task_config = task_dict
        self.task_id = task_dict.get("id", "<unknown>")
        self.instruction = task_dict.get("instruction", "")
        self.vllm_endpoint = vllm_endpoint
        self.vllm_model = vllm_model
        self.provider_name = provider_name
        self.os_type = os_type
        self.max_steps = max_steps
        self.headless = headless
        # OSWorld DesktopEnv defaults `cache_dir` to "cache" (relative to CWD)
        # and downloads Ubuntu.qcow2.zip to `./docker_vm_data` (also relative).
        # Pin both to an absolute path so re-runs from different working
        # directories don't trigger re-downloads (Codex review item P1-#6).
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir else str(
            (Path.cwd() / "osworld_cache").resolve()
        )
        self._env = None  # lazy

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        # Lazy import — host without OSWorld submodule (e.g. local laptop)
        # can still `import visual_gepa.osworld_adapter` for type checks.
        from desktop_env.desktop_env import DesktopEnv
        from desktop_env.providers.docker import manager as docker_manager

        # OSWorld stores TWO independent caches relative to CWD by default:
        #   ./cache/<task_id>/...                 (per-task setup files)
        #   ./docker_vm_data/Ubuntu.qcow2.zip    (10-20 GB VM image)
        # The DesktopEnv ctor only takes cache_dir for the first; the
        # second is the module-level constant `manager.VMS_DIR` (codex
        # B1.2 review flagged this gap). Monkey-patch it onto our
        # cache_dir so re-runs from different CWDs don't re-download.
        if self.provider_name == "docker":
            vms_dir = str(Path(self.cache_dir) / "docker_vm_data")
            Path(vms_dir).mkdir(parents=True, exist_ok=True)
            docker_manager.VMS_DIR = vms_dir
            logger.info("pinned docker VMS_DIR=%s", vms_dir)

        logger.info(
            "DesktopEnv init task_id=%s provider=%s os_type=%s cache_dir=%s",
            self.task_id,
            self.provider_name,
            self.os_type,
            self.cache_dir,
        )
        self._env = DesktopEnv(
            provider_name=self.provider_name,
            os_type=self.os_type,
            headless=self.headless,
            cache_dir=self.cache_dir,
        )

    # --- public API --------------------------------------------------------
    def run(self, prompt: StructuredPrompt) -> MultimodalTrajectory:
        """Roll out one OSWorld episode under the given structured prompt.

        Closes the env on exit (success or exception) so Docker containers
        don't accumulate.
        """
        self._ensure_env()
        env = self._env
        assert env is not None

        # Render the structured prompt as the agent's system message.
        system_prompt = prompt.render()
        agent = _VLLMAgent(
            endpoint=self.vllm_endpoint,
            model=self.vllm_model,
            system_prompt=system_prompt,
            instruction=self.instruction,
        )

        traj = MultimodalTrajectory(
            task_id=self.task_id,
            instruction=self.instruction,
            steps=[],
            final_reward=0.0,
        )
        try:
            obs = env.reset(task_config=self.task_config)
            done = False
            for step_i in range(self.max_steps):
                screenshot = obs.get("screenshot")
                if isinstance(screenshot, bytes):
                    screenshot = Image.open(io.BytesIO(screenshot))
                axtree = obs.get("accessibility_tree") or ""

                response_text = agent.step(screenshot, axtree)
                action = parse_action(response_text)

                try:
                    obs, reward, done, info = env.step(action)
                except Exception as e:  # noqa: BLE001
                    feedback = f"env.step exception: {type(e).__name__}: {e}"
                    logger.warning("step %d: %s", step_i, feedback)
                    traj.steps.append(
                        MultimodalStep(
                            action=action,
                            screenshot=screenshot,
                            accessibility_tree=axtree,
                            reward=None,
                            feedback=feedback,
                            raw_model_text=response_text,
                        )
                    )
                    break

                feedback = (info or {}).get("feedback", "")
                traj.steps.append(
                    MultimodalStep(
                        action=action,
                        screenshot=screenshot,
                        accessibility_tree=axtree,
                        reward=reward,
                        feedback=str(feedback),
                        raw_model_text=response_text,
                    )
                )

                if action.upper().strip() in {"DONE", "FAIL"} or done:
                    break

            # Final reward via official OSWorld evaluator.
            try:
                final_reward = float(env.evaluate())
            except Exception as e:  # noqa: BLE001
                logger.warning("env.evaluate failed: %s", e)
                final_reward = 0.0
            traj.final_reward = final_reward
            return traj
        finally:
            try:
                env.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("env.close failed: %s", e)
            self._env = None

    def metric(self, traj: MultimodalTrajectory) -> tuple[float, str]:
        """Return (score, structured-feedback-string) for FCVR reflection.

        The feedback string captures terminal observation + action trace.
        """
        score = float(traj.final_reward)
        if traj.steps:
            action_summary = " | ".join(
                f"{i + 1}.{(s.action.split('#')[0]).strip()[:60]}"
                for i, s in enumerate(traj.steps)
            )
            terminal = traj.steps[-1].feedback or "(no terminal feedback)"
        else:
            action_summary = "(empty trajectory)"
            terminal = "(no steps)"
        return score, (
            f"task_id={traj.task_id} succeeded={traj.succeeded} "
            f"n_steps={traj.n_steps}\n"
            f"action_trace_summary: {action_summary}\n"
            f"terminal_feedback: {terminal}"
        )
