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
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .structured_prompt import StructuredPrompt

logger = logging.getLogger(__name__)


# --- OSWorld VM watchdog --------------------------------------------------
# Kill the active OSWorld container if a single env operation hangs (B2 mini
# v2 found OSWorld's at-spi axtree fetch can wedge indefinitely on gimp).
# `docker kill` makes the in-flight HTTP call from the OSWorld client raise
# ConnectionError, which our rollout's except handler catches as a clean
# env_step_timeout, and the run terminates without losing the rest of the
# batch.
_OSWORLD_DOCKER_IMAGE = "happysixd/osworld-docker"


def _docker_kill_osworld_containers(reason: str) -> int:
    """Kill all running OSWorld containers. Returns count killed."""
    killed = 0
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"ancestor={_OSWORLD_DOCKER_IMAGE}"],
            capture_output=True, text=True, timeout=10,
        )
        for cid in result.stdout.strip().split("\n"):
            cid = cid.strip()
            if not cid:
                continue
            subprocess.run(["docker", "kill", cid], capture_output=True, timeout=10)
            killed += 1
        if killed:
            logger.warning(
                "watchdog: killed %d OSWorld container(s) — reason=%s",
                killed, reason,
            )
    except Exception as e:  # noqa: BLE001
        logger.error("watchdog kill failed: %s", e)
    return killed


def _with_watchdog(timeout_s: int, label: str, callable_, *args, **kwargs):
    """Run callable_ under a wall-clock watchdog. On timeout, docker-kill
    OSWorld containers so the in-flight HTTP call raises and the python
    code can recover.

    Returns the callable's result. If the watchdog fires, the callable
    still runs to completion — it's just that the underlying HTTP/IPC
    will have been forcibly broken, so it should fail fast after.
    Caller handles the resulting exception.
    """
    if timeout_s <= 0:
        return callable_(*args, **kwargs)
    timer = threading.Timer(
        timeout_s,
        lambda: _docker_kill_osworld_containers(f"{label}_timeout_{timeout_s}s"),
    )
    timer.daemon = True
    timer.start()
    try:
        return callable_(*args, **kwargs)
    finally:
        timer.cancel()


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
    # If non-None, the rollout exited before `max_steps` because of a stop
    # heuristic (e.g. `repeated_actions_3`, `done_token`, `fail_token`,
    # `env_done`, `env_step_exception`). `None` = ran to `max_steps`.
    # Provenance: reward still comes from env.evaluate() in all cases.
    early_stop_reason: str | None = None
    # Reward provenance: how `final_reward` was set. One of:
    #   "env_evaluate" — the official `desktop_env.DesktopEnv.evaluate()`
    #   "env_evaluate_failed" — env.evaluate() raised; final_reward defaulted to 0.0
    #   "no_steps" — env reset failed before any step ran
    # Codex post-run audit (2026-05-27) flagged: "verify rewards come from
    # OSWorld evaluator not inferred logs." This field is the assertion target.
    reward_source: str = "unset"

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


# --- OpenAI-compatible API agent (anyaigc / direct OpenAI / direct Anthropic) ---
#
# Drop-in replacement for `_VLLMAgent` when we want a frontier-model backbone
# instead of self-hosted Qwen. Added 2026-05-28 after B2 proper full found
# Qwen3.5-9B at 1.3% Phase A success on our 25 tasks — far below the 41.8%
# HF-reported number, suggesting Qwen at this scale is the capability
# bottleneck. GPT-5.5 holds 78.7% on OSWorld-Verified (May 2026 leaderboard).
#
# Key differences from `_VLLMAgent`:
#   * No `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
#     (that's Qwen-specific; GPT-5.5 uses reasoning by design).
#   * Uses anyaigc.com base URL with the user-provided API key by default.
#   * System prompt mirrors OSWorld's official `SYS_PROMPT_IN_BOTH_OUT_CODE`
#     for max comparability with leaderboard numbers.
#   * History truncation governed by `max_history_steps` (OSWorld official
#     `max_trajectory_length=3`).
#
class OpenAIAgent:
    """OpenAI-compatible backbone (anyaigc proxy or direct OpenAI/Anthropic).

    Args:
        endpoint: e.g. https://anyaigc.com/v1
        api_key: API key
        model: e.g. "gpt-5.5", "claude-opus-4-7"
        system_prompt: full system message
        instruction: per-task instruction (rendered into every user turn)
        max_history_steps: keep last N agent/env turn pairs in context.
            OSWorld official default is 3 (max_trajectory_length=3).
        temperature: OSWorld official default 1.0
        top_p: OSWorld official default 0.9
        max_tokens: completion budget (output side); OSWorld official 1500
        image_detail: OpenAI vision "low" | "high" | None (default).
            None lets the provider decide. Claude ignores this field.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        system_prompt: str,
        instruction: str,
        max_history_steps: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 1500,
        image_detail: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=endpoint, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.max_history_steps = max_history_steps
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.image_detail = image_detail
        self._history: list[dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0

    def step(self, screenshot: Image.Image, accessibility_tree: str) -> str:
        """Send the current observation; return raw model text."""
        # Truncate history per OSWorld official: only last N user/assistant pairs.
        keep = self._history[-(2 * self.max_history_steps):]

        image_block: dict = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_pil_to_b64_png(screenshot)}"},
        }
        if self.image_detail in {"low", "high"}:
            image_block["image_url"]["detail"] = self.image_detail

        user_content = [
            {
                "type": "text",
                "text": (
                    f"Task instruction: {self.instruction}\n\n"
                    f"Accessibility tree (truncated to 2000 chars):\n"
                    f"{accessibility_tree[:2000]}\n\n"
                    "Reply with ONE pyautogui action in a single ```python ... ``` "
                    "code block, OR the literal token DONE / FAIL / WAIT in its "
                    "own code block, per the system prompt."
                ),
            },
            image_block,
        ]

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + keep
            + [{"role": "user", "content": user_content}]
        )

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        # Some reasoning models (gpt-5-pro, o-series) reject temperature/top_p.
        # Use a try/except cascade so the adapter degrades gracefully if so.
        try:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            if "temperature" in str(e).lower() or "top_p" in str(e).lower():
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                logger.warning("backbone rejected temp/top_p, retrying without: %s", e)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise

        text = resp.choices[0].message.content or ""
        # Track token usage for cost accounting.
        usage = getattr(resp, "usage", None)
        if usage:
            self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                self.total_reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0

        # Update history (drop image to save context tokens; keep terse user/agent
        # exchange so the agent knows what it tried).
        self._history.append({
            "role": "user",
            "content": (
                f"Task: {self.instruction}\n"
                f"(screenshot at step {len(self._history) // 2 + 1} omitted from history)"
            ),
        })
        self._history.append({"role": "assistant", "content": text})
        return text


# --- OSWorld official-style system prompt ---------------------------------------
# Mirrors `mm_agents/prompts.py` SYS_PROMPT_IN_BOTH_OUT_CODE so backbone behavior
# matches OSWorld leaderboard convention as closely as possible.
OSWORLD_OFFICIAL_SYSTEM_PROMPT = (
    "You are an agent which follow my instruction and perform desktop computer tasks "
    "as instructed.\n"
    "You have good knowledge of computer and good internet connection and assume "
    "your code will run on a computer for controlling the mouse and keyboard.\n"
    "For each step, you will get an observation of the desktop by 1) a screenshot; "
    "and 2) accessibility tree, which is based on AT-SPI library. And you will "
    "predict the action of the computer based on the screenshot and accessibility "
    "tree.\n\n"
    "You are required to use `pyautogui` to perform the action grounded to the "
    "observation, but DONT use the `pyautogui.locateCenterOnScreen` function to "
    "locate the element you want to operate with since we have no image of the "
    "element you want to operate with. DONT USE `pyautogui.screenshot()` to make "
    "screenshot.\n"
    "Return one line or multiple lines of python code to perform the action each "
    "time, be time efficient. When predicting multiple lines of code, make some "
    "small sleep like `time.sleep(0.5);` interval so that the machine could take "
    "break between actions. You need to to specify the coordinates of by yourself "
    "based on your observation of current observation, but you should be careful "
    "to ensure that the coordinates are correct.\n\n"
    "You ONLY need to return the code inside a code block, like this:\n"
    "```python\n"
    "# your code here\n"
    "```\n\n"
    "Specially, it is also allowed to return the following special code:\n"
    "When you think you have to wait for some time, return ```WAIT```\n"
    "When you think the task can not be done, return ```FAIL```, don't easily say "
    "`FAIL`, try your best to do the task.\n"
    "When you think the task is done, return ```DONE```.\n\n"
    "My computer's password is 'password', feel free to use it when you need sudo "
    "rights."
)


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
      early_stop_on_repeated_actions: if the last N normalized actions are
        all equal AND N >= this threshold, exit the rollout early with
        `traj.early_stop_reason = "repeated_actions_<N>"`. Default 3.
        Set to 0 to disable (matches OSWorld leaderboard convention).
        Rationale: B2 mini (2026-05-27) found 4/5 tasks produced 1 distinct
        action across all 15 steps — 87% of vLLM cycles were waste on
        screenshot-identical click-loops. Symmetric application across
        vanilla & enhanced rollouts keeps A↔C comparisons unbiased.
      step_watchdog_seconds: per-`env.step` wall-clock cap. If a single
        step (incl. screenshot + axtree fetch + pyautogui execute) takes
        longer than this, kill the active OSWorld docker container
        (image `happysixd/osworld-docker`), which makes the in-flight HTTP
        call raise ConnectionError, our except handler records the crash
        as `env_step_timeout`, and the rollout terminates cleanly. Default
        180s (3 min). Set to 0 to disable. Driven by 2026-05-27 gimp
        Phase C hang: at-spi axtree fetch wedged for 10+ min until we
        manually `docker kill`'d the container.
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
        early_stop_on_repeated_actions: int = 3,
        step_watchdog_seconds: int = 180,
        reset_watchdog_seconds: int = 600,
        evaluate_watchdog_seconds: int = 60,
        # --- Backbone selection (added 2026-05-28) -----------------------------
        # backbone_kind: "vllm" (self-hosted) or "openai_api" (anyaigc / OpenAI /
        # Anthropic via OpenAI-compatible proxy). Default stays "vllm" for
        # backward compat with B0/B1/B2 mini/B2 proper full runs. New
        # experiments should pass backbone_kind="openai_api" + the openai_* args.
        backbone_kind: str = "vllm",
        openai_endpoint: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        agent_max_trajectory_length: int = 3,
        agent_temperature: float = 1.0,
        agent_top_p: float = 0.9,
        agent_max_tokens: int = 1500,
        agent_image_detail: str | None = None,
        agent_system_prompt: str | None = None,
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
        if not isinstance(early_stop_on_repeated_actions, int) or early_stop_on_repeated_actions < 0:
            raise ValueError("early_stop_on_repeated_actions must be int >= 0")
        self.early_stop_on_repeated_actions = early_stop_on_repeated_actions
        for name, v in (
            ("step_watchdog_seconds", step_watchdog_seconds),
            ("reset_watchdog_seconds", reset_watchdog_seconds),
            ("evaluate_watchdog_seconds", evaluate_watchdog_seconds),
        ):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be int >= 0 (0 disables)")
        self.step_watchdog_seconds = step_watchdog_seconds
        self.reset_watchdog_seconds = reset_watchdog_seconds
        self.evaluate_watchdog_seconds = evaluate_watchdog_seconds
        # Backbone selection
        if backbone_kind not in {"vllm", "openai_api"}:
            raise ValueError(f"backbone_kind must be 'vllm' or 'openai_api', got {backbone_kind!r}")
        self.backbone_kind = backbone_kind
        self.openai_endpoint = openai_endpoint
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.agent_max_trajectory_length = agent_max_trajectory_length
        self.agent_temperature = agent_temperature
        self.agent_top_p = agent_top_p
        self.agent_max_tokens = agent_max_tokens
        self.agent_image_detail = agent_image_detail
        self.agent_system_prompt = agent_system_prompt
        if backbone_kind == "openai_api":
            if not openai_endpoint or not openai_api_key or not openai_model:
                raise ValueError(
                    "backbone_kind='openai_api' requires openai_endpoint, "
                    "openai_api_key, and openai_model"
                )
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
        # For openai_api backbone, we PRE-PEND OSWorld's official system prompt
        # (or a user-supplied one) so vision-language agents that haven't been
        # fine-tuned on our SEED_PROMPT still know the action vocabulary.
        rendered_prompt = prompt.render()
        if self.backbone_kind == "vllm":
            agent = _VLLMAgent(
                endpoint=self.vllm_endpoint,
                model=self.vllm_model,
                system_prompt=rendered_prompt,
                instruction=self.instruction,
            )
        else:  # openai_api
            base_sys = self.agent_system_prompt or OSWORLD_OFFICIAL_SYSTEM_PROMPT
            # Append our structured prompt (persona / global_rules / patches /
            # task_scaffold) AFTER the official system prompt so frontier
            # backbones get OSWorld's official action vocabulary first, then
            # our research-side prompt patches.
            combined_sys = (
                f"{base_sys}\n\n"
                "---\n"
                "Additional research-side instructions (Visual-GEPA "
                "structured prompt):\n\n"
                f"{rendered_prompt}"
            )
            agent = OpenAIAgent(
                endpoint=self.openai_endpoint,
                api_key=self.openai_api_key,
                model=self.openai_model,
                system_prompt=combined_sys,
                instruction=self.instruction,
                max_history_steps=self.agent_max_trajectory_length,
                temperature=self.agent_temperature,
                top_p=self.agent_top_p,
                max_tokens=self.agent_max_tokens,
                image_detail=self.agent_image_detail,
            )

        traj = MultimodalTrajectory(
            task_id=self.task_id,
            instruction=self.instruction,
            steps=[],
            final_reward=0.0,
        )
        try:
            try:
                obs = _with_watchdog(
                    self.reset_watchdog_seconds, "env_reset",
                    env.reset, task_config=self.task_config,
                )
            except Exception as e:  # noqa: BLE001
                feedback = f"env.reset exception: {type(e).__name__}: {e}"
                logger.warning("reset: %s", feedback)
                traj.early_stop_reason = "env_reset_exception"
                traj.reward_source = "no_steps"
                return traj
            done = False
            for step_i in range(self.max_steps):
                screenshot = obs.get("screenshot")
                if isinstance(screenshot, bytes):
                    screenshot = Image.open(io.BytesIO(screenshot))
                axtree = obs.get("accessibility_tree") or ""

                response_text = agent.step(screenshot, axtree)
                action = parse_action(response_text)

                try:
                    obs, reward, done, info = _with_watchdog(
                        self.step_watchdog_seconds, "env_step",
                        env.step, action,
                    )
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
                    traj.early_stop_reason = "env_step_exception"
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

                action_upper = action.upper().strip()
                if action_upper == "DONE":
                    traj.early_stop_reason = "done_token"
                    break
                if action_upper == "FAIL":
                    traj.early_stop_reason = "fail_token"
                    break
                if done:
                    traj.early_stop_reason = "env_done"
                    break

                # Early-stop on N-in-a-row identical normalized actions.
                # Skip the check during the first (N-1) steps; require strictly
                # N consecutive equal actions to fire (N=0 disables entirely).
                N = self.early_stop_on_repeated_actions
                if N > 0 and len(traj.steps) >= N:
                    tail = [s.action.strip() for s in traj.steps[-N:]]
                    if all(a == tail[0] and a != "" for a in tail):
                        traj.early_stop_reason = f"repeated_actions_{N}"
                        logger.info(
                            "early-stop: %d consecutive identical actions=%r at step %d",
                            N, tail[0][:80], step_i,
                        )
                        break

            # Final reward via official OSWorld evaluator (always — even on
            # early-stop). Reward provenance recorded for the post-run audit.
            try:
                final_reward = float(_with_watchdog(
                    self.evaluate_watchdog_seconds, "env_evaluate",
                    env.evaluate,
                ))
                traj.reward_source = "env_evaluate"
            except Exception as e:  # noqa: BLE001
                logger.warning("env.evaluate failed: %s", e)
                final_reward = 0.0
                traj.reward_source = "env_evaluate_failed"
            traj.final_reward = final_reward
            if not traj.steps:
                traj.reward_source = "no_steps"
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
