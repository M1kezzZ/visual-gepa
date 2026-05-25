# Visual-GEPA

> **Failure-Clustered Visual Reflection (FCVR)** for prompt evolution on long-horizon multimodal computer-use agents.
> Extension of [GEPA (ICLR 2026 Oral)](https://arxiv.org/abs/2507.19457) to OSWorld / WebArena.

**Status (2026-05-25)**: B0 plumbing smoke PASS — full FCVR pipeline (CLIP + KMeans + MMR + Claude Opus 4.7 vision + 5-field FCVRPatch + structured-prompt merge) verified end-to-end for $0.063. Raw result at `results/B0_smoke.json`. Audit verdict *partial* (mock env; real OSWorld migrates to B1). Code drops over the next 6–8 weeks.

**Target venue**: NeurIPS 2026 workshop (primary) / ICML 2027 main (if signal warrants).

## At a glance

| | |
|---|---|
| Method thesis | Visual-GEPA replaces GEPA's text-only random-sample reflection with **one** budget-constrained failure-trace compression operator (FCVR) — clustering by joint screenshot + error embedding, deterministic MMR key-frame compression, 5-field structured patches, named-section structured-prompt merge |
| Backbone | **Qwen3.5-9B** (self-host BF16 on RTX 5090 32GB, or DashScope API) |
| Reflection LM | **Claude Opus 4.7 vision** (Anthropic API or compatible proxy) |
| Eval envs | OSWorld-Verified (primary) + WebArena (transfer + positioning) |
| Reference paper | GEPA, [arxiv:2507.19457](https://arxiv.org/abs/2507.19457) |
| Code dependency | [gepa-ai/gepa](https://github.com/gepa-ai/gepa) (Apache 2.0, vendored as git submodule) |

## What FCVR is

```
FCVR_B : (failed_trajectories, parent_prompt) -> {structured_patch_k}_{k=1..K}
  where B = (K=4, J=3, M=2, T_patch=512 tokens)  [fixed budget]

Stages (deterministic except for one VLM call per cluster):
  1. Embed:    e_i = [CLIP(final_screenshot); embed(action_summary + error)]
  2. Cluster:  KMeans, K = 4
  3. Compress: MMR + action-boundary key-frame selection (NO VLM call)
  4. Reflect:  ONE VLM call per cluster -> structured patch
                 {failure_pattern, visual_evidence, prompt_diff,
                  scope_guard, expected_behavior_change}

Merge: named-section structured prompt
  P_t = [PERSONA] || [GLOBAL_RULES] || [BEHAVIORAL_PATCHES] || [TASK_SCAFFOLD]
  Each accepted patch appended as (scope_guard, prompt_diff) tuple.
  scope_guard is a natural-language condition, NOT a runtime router.

Token budget (honest by construction):
  T_total(FCVR) ≤ 4 × T_one_vanilla_GEPA_reflection_call
```

## Why it might matter

Vanilla GEPA's text-only random-3 reflection stalls on 30+ step GUI trajectories:
1. **Modality gap** — GUI failures are visually local (wrong button / hidden menu / OS dialog); text traces strip that signal.
2. **Long-horizon credit assignment** — random-3 sampling cannot localize defects across 30+ steps.

FCVR replaces exactly the reflection-input step that breaks, and nothing else. Pareto frontier, candidate manager, mutation operators, agent backbone, env loop, and CLIP embedder are reused unchanged.

## Repo layout

```
visual-gepa/
├── README.md                      ← you are here
├── CLAUDE.md                      ← operating rules + pointer to planning repo
├── LICENSE                        ← Apache 2.0
├── pyproject.toml
├── visual_gepa/                   ← THE contribution
│   ├── __init__.py
│   ├── fcvr.py                    ← FCVR operator (CLIP → KMeans → MMR → 1 Claude call/cluster)
│   ├── key_frame.py               ← deterministic MMR + action-boundary selector (no VLM call)
│   ├── clip_embedder.py           ← HuggingFace CLIPModel wrapper
│   ├── reflection.py              ← ClaudeReflectionClient (vision call → FCVRPatch)
│   ├── patch_schema.py            ← 5-field strict Pydantic patch schema
│   ├── structured_prompt.py       ← named-section [PERSONA]||[GLOBAL_RULES]||[BEHAVIORAL_PATCHES]||[TASK_SCAFFOLD]
│   ├── osworld_adapter.py         ← real env ↔ trajectory bridge (B1+, needs KVM host)
│   ├── mock_osworld_adapter.py    ← synthetic trajectories for B0 plumbing only
│   └── monitors/redundancy.py     ← cosine ≥ 0.85 LOO impact monitor (B2+)
├── scripts/B0_smoke.py            ← B0 orchestration (B1–B7 entry points land here)
├── configs/osworld_smoke_5.json   ← 5 hand-picked task IDs for B0
├── third_party/
│   ├── gepa/                      ← git submodule of gepa-ai/gepa (pinned commit)
│   └── OSWorld/                   ← git submodule of xlang-ai/OSWorld (pinned v0.1.16-574)
└── results/                       ← gitignored EXCEPT for per-block summary JSONs
    └── B0_smoke.json              ← canonical B0 evidence (audit-chain reference)
```

## Quick start

```bash
git clone --recurse-submodules https://github.com/M1kezzZ/visual-gepa.git
cd visual-gepa
python -m venv venv && source venv/bin/activate   # python 3.11 or 3.12
pip install -e .

# Reflection LM config (Anthropic API or compatible proxy):
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-...
# Optional: route through a compatible gateway (e.g. AnyAIGC)
# ANTHROPIC_BASE_URL=https://anyaigc.com
EOF

# B0 smoke (plumbing-only; mock OSWorld env so it runs without KVM/Docker):
python scripts/B0_smoke.py \
  --tasks configs/osworld_smoke_5.json \
  --iterations 5 \
  --output results/B0_smoke.json

# B1+ — real OSWorld; needs vLLM serving Qwen3.5-9B and a KVM-capable host:
vllm serve Qwen/Qwen3.5-9B --port 8000 --dtype bfloat16 --max-model-len 32768
```

Full server-side setup (AutoDL specifics, HF mirror, venv pin, etc.): see `CLAUDE.md` here and `SETUP.md` in the planning repo.

## Reproducibility

- **Open-weight backbone**: Qwen3.5-9B (Apache 2.0). No closed-source SOTA in the agent loop.
- **Pinned dependency**: GEPA submodule commit recorded in this repo.
- **Fixed splits**: OSWorld-Verified 100 → 24 train / 16 val / 60 held-out test (file released in `configs/`).
- **Fixed seeds**: `{42, 1337, 2024}`.
- **Preregistered claims**: pass criterion (2/3-seed dominance + ≥ +3 pp), SkillWeaver positioning cutoff (2026-07-15), abstract A/B variants — committed before run.

## License

Apache 2.0. See `LICENSE`.

## Citation

If this repo helps your work, please cite GEPA (the parent method):

```bibtex
@inproceedings{agrawal2026gepa,
  title  = {GEPA: Reflective Prompt Evolution Outperforms Reinforcement Learning},
  author = {Agrawal, Lakshya A. and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year   = {2026}
}
```

Visual-GEPA citation will be added once the workshop / arxiv paper is out.
