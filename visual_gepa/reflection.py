"""Claude Opus 4.7 vision reflection client.

For each FCVR cluster, ONE call:
    failed trajectory key-frames + structured prompt → FCVRPatch (schema-valid)

Token budget per call is capped (default `T_patch=512` output tokens). The cap
is what makes Visual-GEPA's `T_total(FCVR) ≤ K · T_one_vanilla_GEPA_reflection`
honest by construction (per EXPERIMENT_PLAN.md §3 B2 setup).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image
from pydantic import ValidationError

from .patch_schema import FCVRPatch
from .structured_prompt import StructuredPrompt

logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_MODEL = "claude-opus-4-7"  # Claude Opus 4.7 vision (current API id)
# Raised from 512 → 1024 on 2026-05-28 per B2 proper full audit: short patches
# (~100 tok/field × 5 fields) were generic "verify after click" advice rather
# than task-specific actionable instructions. OSWorld backbone uses max_tokens=
# 1500; reflection output should be at least comparable.
DEFAULT_MAX_OUTPUT_TOKENS = 1024


SYSTEM_PROMPT = """You are a research assistant for Visual-GEPA — an automated
prompt-evolution pipeline for long-horizon multimodal computer-use agents
(OSWorld). You are shown a failure cluster: several failed trajectories that
share a hypothetical failure mode, plus a few key frames per trajectory.

Your task:
  1. Identify the SINGLE shared failure pattern across the cluster.
  2. Cite specific visual evidence in the key frames.
  3. Propose ONE additive behavioral patch the agent should follow to avoid
     this failure on similar tasks.
  4. Express the patch as a 5-field JSON object — described in the user prompt
     — and emit ONLY that JSON object, no prose.

Hard constraints on the JSON object (ALL FIVE fields are PLAIN STRINGS, not
objects, lists, or dicts):
  - failure_pattern   : str, one sentence,  4–512  chars.
  - visual_evidence   : str, refers to frames,  4–1024 chars.
  - prompt_diff       : str, ONE additive natural-language instruction the
                        agent should follow. 4–512 chars. Do NOT wrap it in
                        {"added": [...], "removed": [...]} or any structure.
                        Just one English sentence, exactly the text that
                        will be appended verbatim to [BEHAVIORAL_PATCHES].
  - scope_guard       : str, one-line condition,  4–256  chars. NOT a runtime
                        router — a natural clause like "When the LibreOffice
                        Calc Pivot Table dialog is open and …".
  - expected_behavior_change : str, one sentence,  4–512  chars.

Output strictly ONE JSON object with EXACTLY those five keys, all string
values. No extra keys, no markdown fences, no leading or trailing prose.

Example of the required shape (content is illustrative only):
  {"failure_pattern":"...","visual_evidence":"...","prompt_diff":"...","scope_guard":"...","expected_behavior_change":"..."}
"""


USER_TEMPLATE = """Cluster id: {cluster_id}
Cluster size: {cluster_size} failed trajectories
Task instructions in this cluster:
{instructions}

Current parent prompt (the prompt the agent ran under):
<<<PARENT_PROMPT
{parent_prompt}
PARENT_PROMPT

For each trajectory in the cluster, you see:
  - terminal feedback string
  - action-trace summary
  - {n_frames_per_traj} key frames selected deterministically (action-boundary + MMR).

Then emit ONE FCVRPatch JSON object with fields:
  failure_pattern, visual_evidence, prompt_diff, scope_guard, expected_behavior_change.
"""


@dataclass
class ReflectionCallStats:
    cluster_id: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_input_tokens: int = 0
    latency_s: float = 0.0
    schema_violations: int = 0
    raw_text: str = ""
    error: str | None = None
    duration_attempts: list[float] = field(default_factory=list)


def _pil_to_b64_png(image: Image.Image, max_side: int = 1024) -> str:
    """Encode a PIL image as base64 PNG, optionally downscaled (Claude vision cap)."""
    img = image
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _coerce_patch_fields(obj: dict[str, Any]) -> dict[str, Any]:
    """Defensively coerce the 5 patch fields to plain strings.

    Claude occasionally returns `prompt_diff` as
        {"added": [...], "removed": [...]}
    or as a list of strings. We flatten these to a single string so Pydantic's
    `str` constraint is met *without* an extra Claude call (which would
    violate the FCVR K-budget invariant).
    """
    for key in (
        "failure_pattern",
        "visual_evidence",
        "prompt_diff",
        "scope_guard",
        "expected_behavior_change",
    ):
        v = obj.get(key)
        if v is None or isinstance(v, str):
            continue
        if isinstance(v, list):
            obj[key] = " ".join(str(x) for x in v if x is not None).strip()
            continue
        if isinstance(v, dict):
            parts: list[str] = []
            for sub_key in ("added", "instruction", "text", "diff", "patch"):
                sub = v.get(sub_key)
                if isinstance(sub, str):
                    parts.append(sub)
                elif isinstance(sub, list):
                    parts.extend(str(s) for s in sub if s)
            if not parts:  # last resort: serialize
                import json as _json
                obj[key] = _json.dumps(v, ensure_ascii=False)[:512]
            else:
                obj[key] = " ".join(parts).strip()
            continue
        obj[key] = str(v)
    return obj


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Lenient JSON extraction: prefer raw parse, fall back to first `{...}` block."""
    text = text.strip()
    if text.startswith("```"):
        # strip markdown fence if model violated the instruction
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class ClaudeReflectionClient:
    """One-shot vision reflection: cluster context → FCVRPatch."""

    def __init__(
        self,
        model: str = DEFAULT_REFLECTION_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        api_key: str | None = None,
        base_url: str | None = None,
        schema_violation_retries: int = 0,
    ) -> None:
        if schema_violation_retries < 0:
            raise ValueError(
                f"schema_violation_retries must be >= 0, got {schema_violation_retries}"
            )
        self.model = model
        self.max_output_tokens = max_output_tokens
        # IMPORTANT: each retry is a second Claude call → violates the FCVR
        # K-invariant T_total ≤ K · T_one_vanilla_reflection. Default 0 keeps
        # the invariant honest. Set > 0 only for ablations where the budget
        # claim is explicitly relaxed.
        self.schema_violation_retries = schema_violation_retries
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set (neither in env nor in .env). FCVR "
                "reflection requires Claude vision access."
            )
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None

        # Lazy import so test-time `import visual_gepa.reflection` works without
        # the anthropic SDK being a runtime requirement.
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key, base_url=base_url)

    # --- public API ----------------------------------------------------------
    def reflect_cluster(
        self,
        cluster_id: int,
        cluster_trajectories: list,
        cluster_key_frames: list[list[Image.Image]],
        cluster_action_summaries: list[str],
        parent_prompt: StructuredPrompt,
        n_frames_per_traj: int = 3,
    ) -> tuple[FCVRPatch | None, ReflectionCallStats]:
        """Run ONE Claude vision call → optional FCVRPatch + stats.

        Returns (patch_or_none, stats). patch_or_none == None on schema failure
        even after retries (the cluster is then dropped).
        """
        stats = ReflectionCallStats(cluster_id=cluster_id)
        instructions = "\n  - ".join(
            getattr(t, "instruction", f"<task {i}>") for i, t in enumerate(cluster_trajectories)
        )
        user_text = USER_TEMPLATE.format(
            cluster_id=cluster_id,
            cluster_size=len(cluster_trajectories),
            instructions=("  - " + instructions) if instructions else "  (none)",
            parent_prompt=parent_prompt.render(),
            n_frames_per_traj=n_frames_per_traj,
        )

        # Compose multi-modal user message.
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for traj, frames, summary in zip(
            cluster_trajectories, cluster_key_frames, cluster_action_summaries, strict=False
        ):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"--- traj task_id={getattr(traj, 'task_id', '?')} "
                        f"n_steps={getattr(traj, 'n_steps', '?')} "
                        f"final_reward={getattr(traj, 'final_reward', '?')} ---\n"
                        f"action_trace_summary: {summary}\n"
                        f"terminal_feedback: {getattr(traj.steps[-1], 'feedback', '') if getattr(traj, 'steps', None) else ''}\n"
                        f"key_frames ({len(frames)}):"
                    ),
                }
            )
            for img in frames:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _pil_to_b64_png(img),
                        },
                    }
                )

        last_error: str | None = None
        attempt = 0
        max_attempts = 1 + self.schema_violation_retries
        while attempt < max_attempts:
            t0 = time.perf_counter()
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                )
            except Exception as e:  # noqa: BLE001 — surface API errors as a stat
                last_error = f"{type(e).__name__}: {e}"
                logger.warning("Claude API call failed (attempt %d): %s", attempt + 1, last_error)
                stats.duration_attempts.append(time.perf_counter() - t0)
                attempt += 1
                # transient back-off
                time.sleep(min(2 ** attempt, 8))
                continue

            dt = time.perf_counter() - t0
            stats.duration_attempts.append(dt)
            stats.latency_s += dt
            # Anthropic SDK ≥ 0.40 returns usage on the response.
            usage = getattr(resp, "usage", None)
            if usage is not None:
                stats.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                stats.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                stats.cache_input_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)

            text_parts: list[str] = []
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", ""))
            raw = "\n".join(text_parts).strip()
            stats.raw_text = raw

            obj = _extract_json_object(raw)
            if obj is None:
                stats.schema_violations += 1
                last_error = "no JSON object in response"
                attempt += 1
                continue
            obj = _coerce_patch_fields(obj)  # deterministic; not a retry
            try:
                patch = FCVRPatch(**obj)
                return patch, stats
            except ValidationError as ve:
                stats.schema_violations += 1
                last_error = f"schema: {ve.errors()[:1]}"
                logger.warning(
                    "FCVRPatch schema violation (attempt %d): %s", attempt + 1, last_error
                )
                attempt += 1

        stats.error = last_error
        return None, stats
