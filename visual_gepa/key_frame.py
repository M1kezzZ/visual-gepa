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
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


# --- Action-kind classifier --------------------------------------------------
# OSWorld agent actions are typically pyautogui-style strings, plus special
# tokens (WAIT / DONE / FAIL). Order matters: more specific patterns first.
_ACTION_KIND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(pyautogui\.)?double[_ ]?click\b", re.I), "double_click"),
    (re.compile(r"^\s*(pyautogui\.)?right[_ ]?click\b", re.I), "right_click"),
    (re.compile(r"^\s*(pyautogui\.)?(left[_ ]?)?click\b", re.I), "click"),
    (re.compile(r"^\s*(pyautogui\.)?(type|write|typewrite)\b", re.I), "type"),
    (re.compile(r"^\s*(pyautogui\.)?press\b", re.I), "press"),
    (re.compile(r"^\s*(pyautogui\.)?hotkey\b", re.I), "hotkey"),
    (re.compile(r"^\s*(pyautogui\.)?scroll\b", re.I), "scroll"),
    (re.compile(r"^\s*(pyautogui\.)?drag(to)?\b", re.I), "drag"),
    (re.compile(r"^\s*(pyautogui\.)?(moveto|move)\b", re.I), "move"),
    (re.compile(r"^\s*WAIT\b", re.I), "wait"),
    (re.compile(r"^\s*DONE\b", re.I), "done"),
    (re.compile(r"^\s*FAIL\b", re.I), "fail"),
]


def action_kind(action: str) -> str:
    """Classify an OSWorld action string into a coarse kind."""
    if not action:
        return "other"
    for pat, kind in _ACTION_KIND_PATTERNS:
        if pat.search(action):
            return kind
    return "other"


# --- Failure / transition cues -----------------------------------------------
_FAILED_FEEDBACK_PATTERNS = [
    re.compile(r"\b(no[-_ ]?op|nothing happened|did nothing|no effect|failed to)\b", re.I),
    re.compile(r"\b(error|exception|traceback)\b", re.I),
    re.compile(r"\bnot found\b", re.I),
    re.compile(r"\b(missed|out of bounds|outside window)\b", re.I),
]

_PAGE_TRANSITION_PATTERNS = [
    re.compile(r"\bnew\s+(window|page|tab|dialog|modal)\b", re.I),
    re.compile(r"\b(activated|focused|switched\s+to)\s+\w+", re.I),
    re.compile(r"\bnavigated\s+to\b", re.I),
    re.compile(r"\bdialog\s+(opened|appeared)\b", re.I),
]


def _is_failed_step(feedback: str | None) -> bool:
    if not feedback:
        return False
    return any(p.search(feedback) for p in _FAILED_FEEDBACK_PATTERNS)


def _is_page_transition(observation: str | None) -> bool:
    if not observation:
        return False
    return any(p.search(observation) for p in _PAGE_TRANSITION_PATTERNS)


# --- Public API --------------------------------------------------------------
def action_boundaries(trajectory) -> list[int]:
    """Return frame indices that are candidate MMR seeds.

    A frame i is flagged when ANY of:
      - kind(step_i.action) != kind(step_{i-1}.action)
      - step_i.feedback indicates a failed click / no-op / error
      - step_i.accessibility_tree indicates a page/window/focus transition
      - i is one of the terminal two step indices.

    Returns a deduped, order-preserving list of valid indices.
    """
    steps = getattr(trajectory, "steps", None) or []
    n = len(steps)
    if n == 0:
        return []

    boundaries: list[int] = []
    prev_kind: str | None = None
    for i, step in enumerate(steps):
        kind = action_kind(getattr(step, "action", "") or "")
        flag = False
        if prev_kind is not None and kind != prev_kind:
            flag = True
        if _is_failed_step(getattr(step, "feedback", "") or ""):
            flag = True
        if _is_page_transition(getattr(step, "accessibility_tree", "") or ""):
            flag = True
        if i >= n - 2:  # last two steps always included
            flag = True
        if flag:
            boundaries.append(i)
        prev_kind = kind

    # Dedupe, preserve insertion order.
    seen: set[int] = set()
    out: list[int] = []
    for i in boundaries:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def mmr_select(
    frame_embeddings: np.ndarray,
    seeds: Sequence[int],
    budget: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal-Marginal-Relevance selection over CLIP frame embeddings.

    Stages:
      1. L2-normalize embeddings.
      2. Query vector q = mean of embeddings at `seeds` (or global mean if empty).
      3. Seed output with `seeds` in order, up to `budget`.
      4. If still short, greedy MMR over remaining indices using
         score(i) = λ · sim(q, e_i) − (1−λ) · max_{j ∈ selected} sim(e_j, e_i).
      5. Return *sorted* (chronological) indices.

    Determinism is preserved because ties are broken by lowest index (numpy's
    argmax behavior on ties) which is deterministic given fixed embeddings.
    """
    if frame_embeddings is None or len(frame_embeddings) == 0 or budget <= 0:
        return []
    emb = _l2_normalize(np.asarray(frame_embeddings, dtype=np.float32))
    n = emb.shape[0]
    budget = min(budget, n)

    valid_seeds = [int(i) for i in (seeds or []) if 0 <= int(i) < n]
    if valid_seeds:
        q = emb[valid_seeds].mean(axis=0)
    else:
        q = emb.mean(axis=0)
    qn = np.linalg.norm(q)
    q = q / qn if qn > 1e-12 else q

    selected: list[int] = []
    seen: set[int] = set()
    for s in valid_seeds:
        if s not in seen and len(selected) < budget:
            selected.append(s)
            seen.add(s)

    if len(selected) < budget:
        sim_q = emb @ q  # (n,)
        sim_pairs = emb @ emb.T  # (n,n)
        while len(selected) < budget:
            best_i: int | None = None
            best_score = -np.inf
            for i in range(n):
                if i in seen:
                    continue
                if selected:
                    max_sim = float(sim_pairs[i, selected].max())
                else:
                    max_sim = 0.0
                score = lambda_ * float(sim_q[i]) - (1.0 - lambda_) * max_sim
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is None:
                break
            selected.append(best_i)
            seen.add(best_i)

    return sorted(selected)


def select_key_frames(
    trajectory,
    frame_embeddings: np.ndarray,
    budget: int = 3,
    lambda_: float = 0.7,
) -> list[int]:
    """Full pipeline: action_boundaries → MMR-select → chronological indices."""
    seeds = action_boundaries(trajectory)
    return mmr_select(frame_embeddings, seeds, budget=budget, lambda_=lambda_)
