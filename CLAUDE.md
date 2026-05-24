# CLAUDE.md

Guidance for Claude Code sessions opened in `~/code/visual-gepa/` (the **code** side of Visual-GEPA).

## What this directory is

The **public, Apache-2.0 code repo** for Visual-GEPA — extension of [GEPA (ICLR 2026 Oral)](https://arxiv.org/abs/2507.19457) to long-horizon multimodal computer-use agents.

The corresponding **private planning / paper / refine-audit** repo lives at:

- Local: `/Users/mike/Documents/Visual-GEPA/`
- Remote: 🔒 `github.com/M1kezzZ/visual-gepa-research` (private; Mike + Li Sa `@1qazse432w`)

**For research context, claims, experiment plan, refine audit, paper drafts → read the planning repo.** This code repo only contains the implementation.

## Quick orientation

| | |
|---|---|
| Remote | 🌐 https://github.com/M1kezzZ/visual-gepa (public) |
| License | Apache 2.0 |
| Backbone agent | Qwen3.5-9B (self-host BF16, see `SETUP.md` in planning repo for hardware notes) |
| Reflection LM | Claude Opus 4.7 vision (Anthropic API; not Claude.ai subscription) |
| Method | **FCVR** — Failure-Clustered Visual Reflection, single budget-constrained operator |
| Budget | K=4 clusters, J=3 key frames, M=2 trajs per cluster, T_patch=512 tokens (frozen) |
| Pass criterion | Full FCVR > both ablations in ≥ 2/3 seeds AND mean Δ ≥ +3 pp on OSWorld holdout |

## Repo layout

```
visual-gepa/
├── visual_gepa/                ← THE contribution
│   ├── fcvr.py                 ← FCVR_B operator
│   ├── key_frame.py            ← deterministic MMR + action-boundary
│   ├── patch_schema.py         ← 5-field Pydantic FCVRPatch
│   ├── structured_prompt.py    ← [PERSONA] || [GLOBAL_RULES] || [BEHAVIORAL_PATCHES] || [TASK_SCAFFOLD]
│   ├── osworld_adapter.py      ← env ↔ trajectory bridge
│   └── monitors/redundancy.py  ← cosine ≥ 0.85 LOO impact drop
├── scripts/B0_smoke.py         ← B0-B7 entry points land here
├── configs/                    ← per-block YAML configs
└── third_party/
    ├── gepa/                   ← submodule of M1kezzZ/gepa (fork of gepa-ai/gepa)
    └── OSWorld/                ← submodule of xlang-ai/OSWorld
```

## Operating rules

- **NEVER modify `third_party/gepa/`** — subclass from `visual_gepa/` instead. We pin GEPA by submodule commit for reproducibility traceability. If GEPA upstream renames a symbol, fix the import in `visual_gepa/fcvr.py`, don't patch upstream.
- **NEVER commit `.env`, API keys, large model checkpoints, or experiment result binaries** — see `.gitignore`. Use `~/models/` for checkpoints and `results/` (gitignored) for runs.
- **Per-repo git identity** is `M1kezzZ <peng824389@gmail.com>` (local override). Other repos use the global default (`Mike Peng <peng824389@gmail.com>`).
- **Do not push research strategy or paper-internal discussion here.** That belongs in the private planning repo.
- **Test before push**: code in `visual_gepa/` should be importable + `ruff` clean before commit.

## What this code repo does NOT contain

- ❌ Refine audit trail (private; in `visual-gepa-research/refine-logs/`)
- ❌ Paper drafts (private; in `visual-gepa-research/paper/`, future)
- ❌ Decision logs, scoop monitoring, claim disputes (private)
- ❌ Day-1 / SETUP detail (see `SETUP.md` in planning repo)

## Cross-repo workflow

```
Research / refine / paper writing       Implementation / experiments
─────────────────────────────────       ────────────────────────────
/Users/mike/Documents/Visual-GEPA/      ~/code/visual-gepa/
        │                                        │
        │ git push                       git push │
        ▼                                        ▼
M1kezzZ/visual-gepa-research (🔒)       M1kezzZ/visual-gepa (🌐)
        │                                        │
   ─────┼──────── Mike + Li Sa ──────────────────│
                                                 │
                                          third_party/gepa  ──► M1kezzZ/gepa (fork)
                                          third_party/OSWorld ──► xlang-ai/OSWorld
```

## When in doubt

- **"Where do I put this?"** — Is it code an open-source user could run to reproduce a result? → here. Is it strategy, decision, paper, or audit? → planning repo.
- **"Why is GEPA in `third_party/`?"** — because it's a submodule (pinned upstream commit). Treat it as a vendored library.
- **"Where's the experiment plan?"** — `EXPERIMENT_PLAN.md` in planning repo; tracker at `refine-logs/EXPERIMENT_TRACKER.md`.
