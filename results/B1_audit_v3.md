# B1 Baseline v3 — Audit Table: raw_model_text → code_block → parse_action → action

**Source**: `results/B1_baseline_5task_seed42_v3.json`

**Per-step category** (smarter than v1 — compares Qwen's code-block to defend against parser-collapse attack):
- 🟢 `FRESH` — first occurrence of this parsed action
- 🔁 `QWEN_LOOP` — identical raw text → identical action (Qwen literally repeated)
- 🗣️ `PROSE_VARY` — different prose but identical code-block → Qwen ruminated differently but emitted the same action (still attributable to Qwen, not parser)
- ⚠️ `PARSE_COLLAPSE` — different code-block but identical parsed action (parser is implicated — investigate `parse_action`)

---

## Verdict

✅ **CLICK-LOOP IS IN QWEN'S OUTPUT** (64/75 repeat-steps attributable to Qwen; **0** parser collapses). Qwen literally wrote the same `pyautogui.click(...)` call repeatedly — sometimes verbatim (`QWEN_LOOP`), sometimes with different surrounding reasoning prose (`PROSE_VARY`) — but the action code-block is identical. **The reviewer-2 attack 'your parser manufactured the failure mode' is defanged.**


## libreoffice_calc/1954cced-e748-45c4-9c26-9855b97fbc5e

- n_steps: **15** | distinct raws: **11** | distinct code-blocks: **4** | distinct actions: **4**
- QWEN_LOOP: **0** | PROSE_VARY: **11** | PARSE_COLLAPSE: **0** | FRESH: **4**


| step | code_hash | category | parsed_action | code_block (Qwen wrote) |
|---|---|---|---|---|
| 0 | `0cf3b36d` | 🟢 FRESH | `pyautogui.hotkey('ctrl', 'm')` | `pyautogui.hotkey('ctrl', 'm')` |
| 1 | `53f287aa` | 🟢 FRESH | `pyautogui.click(x=214, y=67)` | `pyautogui.click(x=214, y=67)` |
| 2 | `d6efa57f` | 🟢 FRESH | `pyautogui.click(x=134, y=140)` | `pyautogui.click(x=134, y=140)` |
| 3 | `d6efa57f` | 🗣️ PROSE_VARY | `pyautogui.click(x=134, y=140)` | `pyautogui.click(x=134, y=140)` |
| 4 | `29179e62` | 🟢 FRESH | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 5 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 6 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 7 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 8 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 9 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 10 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 11 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 12 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 13 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |
| 14 | `29179e62` | 🗣️ PROSE_VARY | `pyautogui.click(x=117, y=67)` | `pyautogui.click(x=117, y=67)` |

## chrome/a728a36e-8bf1-4bb6-9a03-ef039a5233f0

- n_steps: **15** | distinct raws: **2** | distinct code-blocks: **2** | distinct actions: **2**
- QWEN_LOOP: **13** | PROSE_VARY: **0** | PARSE_COLLAPSE: **0** | FRESH: **2**


| step | code_hash | category | parsed_action | code_block (Qwen wrote) |
|---|---|---|---|---|
| 0 | `2ba28b07` | 🟢 FRESH | `pyautogui.click(x=431, y=194)` | `pyautogui.click(x=431, y=194)` |
| 1 | `af79efac` | 🟢 FRESH | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 2 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 3 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 4 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 5 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 6 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 7 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 8 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 9 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 10 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 11 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 12 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 13 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |
| 14 | `af79efac` | 🔁 QWEN_LOOP | `pyautogui.click(x=242, y=256)` | `pyautogui.click(x=242, y=256)` |

## libreoffice_writer/8472fece-c7dd-4241-8d65-9b3cd1a0b568

- n_steps: **15** | distinct raws: **1** | distinct code-blocks: **1** | distinct actions: **1**
- QWEN_LOOP: **14** | PROSE_VARY: **0** | PARSE_COLLAPSE: **0** | FRESH: **1**


| step | code_hash | category | parsed_action | code_block (Qwen wrote) |
|---|---|---|---|---|
| 0 | `ffd42a9c` | 🟢 FRESH | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 1 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 2 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 3 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 4 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 5 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 6 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 7 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 8 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 9 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 10 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 11 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 12 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 13 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |
| 14 | `ffd42a9c` | 🔁 QWEN_LOOP | `pyautogui.click(x=365, y=450)` | `pyautogui.click(x=365, y=450)` |

## vs_code/5e2d93d8-8ad0-4435-b150-1692aacaa994

- n_steps: **15** | distinct raws: **3** | distinct code-blocks: **3** | distinct actions: **3**
- QWEN_LOOP: **12** | PROSE_VARY: **0** | PARSE_COLLAPSE: **0** | FRESH: **3**


| step | code_hash | category | parsed_action | code_block (Qwen wrote) |
|---|---|---|---|---|
| 0 | `e8f8abc1` | 🟢 FRESH | `pyautogui.click(x=17, y=56)` | `pyautogui.click(x=17, y=56)` |
| 1 | `0319c227` | 🟢 FRESH | `pyautogui.click(x=17, y=183)` | `pyautogui.click(x=17, y=183)` |
| 2 | `bf934971` | 🟢 FRESH | `pyautogui.click(x=45, y=70)` | `pyautogui.click(x=45, y=70)` |
| 3 | `e8f8abc1` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=56)` | `pyautogui.click(x=17, y=56)` |
| 4 | `e8f8abc1` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=56)` | `pyautogui.click(x=17, y=56)` |
| 5 | `0319c227` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=183)` | `pyautogui.click(x=17, y=183)` |
| 6 | `e8f8abc1` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=56)` | `pyautogui.click(x=17, y=56)` |
| 7 | `0319c227` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=183)` | `pyautogui.click(x=17, y=183)` |
| 8 | `bf934971` | 🔁 QWEN_LOOP | `pyautogui.click(x=45, y=70)` | `pyautogui.click(x=45, y=70)` |
| 9 | `0319c227` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=183)` | `pyautogui.click(x=17, y=183)` |
| 10 | `e8f8abc1` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=56)` | `pyautogui.click(x=17, y=56)` |
| 11 | `0319c227` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=183)` | `pyautogui.click(x=17, y=183)` |
| 12 | `e8f8abc1` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=56)` | `pyautogui.click(x=17, y=56)` |
| 13 | `0319c227` | 🔁 QWEN_LOOP | `pyautogui.click(x=17, y=183)` | `pyautogui.click(x=17, y=183)` |
| 14 | `bf934971` | 🔁 QWEN_LOOP | `pyautogui.click(x=45, y=70)` | `pyautogui.click(x=45, y=70)` |

## gimp/7b7617bd-57cc-468e-9c91-40c4ec2bcb3d

- n_steps: **15** | distinct raws: **1** | distinct code-blocks: **1** | distinct actions: **1**
- QWEN_LOOP: **14** | PROSE_VARY: **0** | PARSE_COLLAPSE: **0** | FRESH: **1**


| step | code_hash | category | parsed_action | code_block (Qwen wrote) |
|---|---|---|---|---|
| 0 | `5454bae1` | 🟢 FRESH | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 1 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 2 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 3 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 4 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 5 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 6 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 7 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 8 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 9 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 10 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 11 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 12 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 13 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
| 14 | `5454bae1` | 🔁 QWEN_LOOP | `pyautogui.click(x=1789, y=965)` | `pyautogui.click(x=1789, y=965)` |
