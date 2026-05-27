"""Post-hoc audit table for B1 baseline v3 result.

Purpose (per codex retrospective R7 + top-action 2):
  Defend the click-loop finding against a reviewer-2 attack of the form
  "your parser collapsed distinct Qwen outputs into identical actions —
  the failure mode is in your code, not in the agent."

How:
  For each step, extract the *code block* from the raw model output (what
  Qwen literally wrote as the action), and compare across steps:
    - QWEN_LOOP      : Qwen's code-block text is identical to a prior step's
                       code-block text → Qwen is the looper (failure upstream
                       of the parser).
    - PARSE_COLLAPSE : Qwen's code-block text DIFFERS from a prior step's,
                       but parse_action mapped both to the same final action
                       → the parser is implicated. THIS IS THE FAIL CASE.
    - PROSE_VARY     : Distinct raw prose but identical code-block text
                       (Qwen ruminates differently but emits the same action).
                       Treated as QWEN_LOOP for the verdict — the action came
                       from Qwen, not the parser.
    - FRESH          : First occurrence of this parsed action.

Output: markdown + JSON tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.S | re.I)


def _hash_short(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _head(s: str, n: int = 100) -> str:
    if not s:
        return ""
    flat = " ".join(s.split())
    return flat[:n] + ("…" if len(flat) > n else "")


def extract_code_block(raw: str) -> str:
    """Extract the LAST python code block from Qwen's raw output.

    Mirrors what parse_action sees as the agent's intended action — but
    returns the FULL code block string, not the trimmed action, so we can
    compare verbatim what Qwen wrote across steps.
    """
    if not raw:
        return ""
    matches = _CODE_BLOCK_RE.findall(raw)
    if matches:
        return matches[-1].strip()  # use last block — Qwen sometimes shows example then real
    return ""


def audit_task(task: dict) -> dict:
    raws: list[str] = task.get("raw_model_texts", []) or []
    acts: list[str] = task.get("actions", []) or []
    n = min(len(raws), len(acts))

    code_blocks: list[str] = [extract_code_block(r) for r in raws[:n]]
    raw_hashes = [_hash_short(r) for r in raws[:n]]
    code_hashes = [_hash_short(c) for c in code_blocks]

    rows = []
    parse_collapses = 0
    qwen_loops_strict = 0  # identical raw → identical action
    prose_vary_same_code = 0  # different prose but identical code-block → still Qwen's choice
    fresh = 0

    for i in range(n):
        same_act_idx = [j for j in range(i) if acts[j] == acts[i]]
        if not same_act_idx:
            category = "FRESH"
            fresh += 1
        else:
            prior = same_act_idx[-1]
            if code_hashes[prior] == code_hashes[i] and code_blocks[i]:
                # Same code block (and non-empty) — Qwen really wrote the same action
                if raw_hashes[prior] == raw_hashes[i]:
                    category = "QWEN_LOOP"
                    qwen_loops_strict += 1
                else:
                    category = "PROSE_VARY"
                    prose_vary_same_code += 1
            else:
                # Same parsed action but different code block (or empty) — parser is suspect
                category = "PARSE_COLLAPSE"
                parse_collapses += 1

        rows.append({
            "step": i,
            "code_hash": code_hashes[i],
            "code_block": _head(code_blocks[i], 80),
            "raw_hash": raw_hashes[i],
            "raw_head": _head(raws[i], 100),
            "parsed_action": _head(acts[i], 80),
            "category": category,
        })

    # Qwen is "the looper" if either QWEN_LOOP (identical raw) or PROSE_VARY (different prose,
    # same code block) — both attribute the repeated action to Qwen.
    qwen_attributable = qwen_loops_strict + prose_vary_same_code

    return {
        "task_id": task.get("task_id"),
        "n_steps": n,
        "n_distinct_raws": len(set(raw_hashes)),
        "n_distinct_code_blocks": len(set(code_hashes)),
        "n_distinct_actions": len(set(acts[:n])),
        "qwen_loops_strict": qwen_loops_strict,
        "prose_vary_same_code": prose_vary_same_code,
        "qwen_attributable": qwen_attributable,
        "parse_collapses": parse_collapses,
        "fresh": fresh,
        "rows": rows,
    }


def render_markdown(audits: list[dict], src_path: str) -> str:
    lines = [
        "# B1 Baseline v3 — Audit Table: raw_model_text → code_block → parse_action → action\n",
        f"**Source**: `{src_path}`\n",
        "**Per-step category** (smarter than v1 — compares Qwen's code-block to defend against parser-collapse attack):",
        "- 🟢 `FRESH` — first occurrence of this parsed action",
        "- 🔁 `QWEN_LOOP` — identical raw text → identical action (Qwen literally repeated)",
        "- 🗣️ `PROSE_VARY` — different prose but identical code-block → Qwen ruminated differently but emitted the same action (still attributable to Qwen, not parser)",
        "- ⚠️ `PARSE_COLLAPSE` — different code-block but identical parsed action (parser is implicated — investigate `parse_action`)",
        "\n---\n",
    ]

    overall_qwen = sum(a["qwen_attributable"] for a in audits)
    overall_collapse = sum(a["parse_collapses"] for a in audits)
    overall_steps = sum(a["n_steps"] for a in audits)
    if overall_collapse == 0 and overall_qwen > 0:
        verdict = (
            f"✅ **CLICK-LOOP IS IN QWEN'S OUTPUT** ({overall_qwen}/{overall_steps} repeat-steps "
            "attributable to Qwen; **0** parser collapses). Qwen literally wrote the same "
            "`pyautogui.click(...)` call repeatedly — sometimes verbatim (`QWEN_LOOP`), sometimes "
            "with different surrounding reasoning prose (`PROSE_VARY`) — but the action code-block "
            "is identical. **The reviewer-2 attack 'your parser manufactured the failure mode' is "
            "defanged.**"
        )
    elif overall_collapse > 0:
        verdict = (
            f"⚠️ **Parser collapses {overall_collapse}/{overall_steps} steps** — distinct "
            "code-blocks mapped to the same parsed action. The click-loop narrative IS contaminated "
            "by parser artifact. Investigate `parse_action` before claiming the failure mode is in Qwen."
        )
    else:
        verdict = "ℹ️ Insufficient looping signal in this audit."
    lines.append(f"## Verdict\n\n{verdict}\n\n")

    for a in audits:
        lines.append(f"## {a['task_id']}\n")
        lines.append(
            f"- n_steps: **{a['n_steps']}** | distinct raws: **{a['n_distinct_raws']}** | "
            f"distinct code-blocks: **{a['n_distinct_code_blocks']}** | "
            f"distinct actions: **{a['n_distinct_actions']}**\n"
            f"- QWEN_LOOP: **{a['qwen_loops_strict']}** | "
            f"PROSE_VARY: **{a['prose_vary_same_code']}** | "
            f"PARSE_COLLAPSE: **{a['parse_collapses']}** | "
            f"FRESH: **{a['fresh']}**\n\n"
        )
        lines.append("| step | code_hash | category | parsed_action | code_block (Qwen wrote) |")
        lines.append("|---|---|---|---|---|")
        for r in a["rows"]:
            cat_em = {
                "FRESH": "🟢", "QWEN_LOOP": "🔁", "PROSE_VARY": "🗣️", "PARSE_COLLAPSE": "⚠️"
            }.get(r["category"], "")
            lines.append(
                f"| {r['step']} | `{r['code_hash']}` | {cat_em} {r['category']} | "
                f"`{r['parsed_action']}` | `{r['code_block']}` |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="results/B1_baseline_5task_seed42_v3.json")
    ap.add_argument("--out-md", default="results/B1_audit_v3.md")
    ap.add_argument("--out-json", default="results/B1_audit_v3.json")
    args = ap.parse_args()

    src = Path(args.in_path)
    if not src.exists():
        print(f"ERROR: input not found: {src}")
        return 1
    d = json.loads(src.read_text())
    audits = [audit_task(t) for t in d.get("tasks", []) if t.get("crashed_with") is None]
    Path(args.out_md).write_text(render_markdown(audits, str(src)))
    Path(args.out_json).write_text(json.dumps({"audits": audits}, indent=2, ensure_ascii=False))

    print(f"wrote {args.out_md} + {args.out_json}\n")
    total_steps = sum(a["n_steps"] for a in audits)
    total_qwen = sum(a["qwen_attributable"] for a in audits)
    total_strict = sum(a["qwen_loops_strict"] for a in audits)
    total_vary = sum(a["prose_vary_same_code"] for a in audits)
    total_collapse = sum(a["parse_collapses"] for a in audits)
    total_fresh = sum(a["fresh"] for a in audits)
    print(f"=== AGGREGATE ===")
    print(f"  tasks audited:        {len(audits)}")
    print(f"  total steps:          {total_steps}")
    print(f"  FRESH:                {total_fresh:3d} ({100*total_fresh/max(total_steps,1):.0f}%)")
    print(f"  QWEN_LOOP (strict):   {total_strict:3d} ({100*total_strict/max(total_steps,1):.0f}%)")
    print(f"  PROSE_VARY:           {total_vary:3d} ({100*total_vary/max(total_steps,1):.0f}%)")
    print(f"  QWEN-attributable:    {total_qwen:3d} ({100*total_qwen/max(total_steps,1):.0f}%)")
    print(f"  PARSE_COLLAPSE:       {total_collapse:3d} ({100*total_collapse/max(total_steps,1):.0f}%)")
    print()
    if total_collapse == 0 and total_qwen > 0:
        print("VERDICT: click-loop in Qwen's code-block output (not parser) — finding defended ✓")
    elif total_collapse > 0:
        print(f"VERDICT: parser COLLAPSED {total_collapse} step(s) — investigate parse_action ⚠️")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
