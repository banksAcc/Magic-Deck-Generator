# agents.md — Magic-Deck-Generator
**Version:** v1 • **Status:** Active • **Owner:** DS/ML Team  
**Purpose:** Operational playbook for any agent (human or tool-driven) working on this repository. Built for “walking skeleton” development: minimal, robust, and easy to extend.

---

## 1) Mission, Scope, and Non-Goals
**Mission.** Build an end-to-end pipeline for *conditional* MTG-style card generation. The MTG corpus teaches the *format and mechanics*; the MHA theme is injected **only at generation time** via structured prompts.

**Primary model:** Mistral 7B QLoRA.  
**Baseline:** GPT-2 Large (full FT and LoRA).  
**Language:** English. **Seq len:** 512. **Layouts:** only "normal".  
**Leakage:** some is acceptable; PPL is indicative, not a production benchmark.

**Non-Goals (for this iteration):**
- No multi-face layouts (DFC/Adventure/Saga/flip/split).
- No constraint decoding or heavy grammar tooling (may come later).
- No fixed random seeds.

---

## 2) Repository Layout & Naming
```
Magic-Deck-Generator/
├─ configs/
│  └─ config.yaml              # SINGLE SOURCE OF TRUTH (all constants & params)
├─ data/
│  ├─ raw/                     # scryfall.json (bulk)
│  └─ processed/               # all.txt, splits, stats
├─ src/
│  ├─ mapping/
│  │  ├─ keywords.json
│  │  └─ mapping_seed.csv
│  │
│  ├─ __init__.py              # empty (marks package)
│  ├─ a00_utils.py             # config, W&B, regex, domain, basic I/O
│  ├─ a01_preprocessor.py      # bulk -> all.txt (+ stats, W&B lite)
│  ├─ a02_split.py             # (planned) 80/10/10 split
│  ├─ a03_validate.py          # (planned) hard validator
│  ├─ a04_train_mistral_lora.py
│  ├─ a04_train_gpt2.py        # (planned) GPT-2 baseline
│  ├─ a05_eval_ppl.py          # (planned) PPL eval (global & clusters)
│  ├─ a06_generate.py          # (planned) conditional generation
│  └─ a07_curate.py            # (planned) ranking + balancing
├─ outputs/
│  ├─ checkpoints/
│  ├─ generations/
│  └─ final_set/
│
├─ README.md
├─ requirements.txt
├─ operative_guide.txt
└─ agents.md                   # this file
```
**Naming convention:** prefix modules with `aXX_` to preserve execution order and simplify `python -m src.<module>` usage. Do **not** start filenames with digits only; use `aXX_` or `_XX_`.

---

## 3) Single Config Contract
All constants live in **`configs/config.yaml`** (no duplicates in code). Key sections:
- `data` — paths, language, layouts, `special_tokens` (`<|startofcard|>`, `<|gen_card|>`, `<|endofcard|>`), `seq_len`.
- `constants` — `colors`, `rarities`, `types`, `regex.mana_pattern`, `regex.pt_pattern`.
- `validator` — `enforce_field_order`, `require_eos`, `forbid_pt_without_creature`, `color_mana_subset_rule` (default false).
- `training.gpt2` / `training.mistral` — hyper-parameters.
- `generation` — decoding params (temperature, top_p, repetition_penalty, max_new_tokens, eos_token, num_samples).
- `curation` — `target_counts`, `balance_by`, `dedup_threshold` (Levenshtein-normalized threshold).
- `wandb` — `enabled`, `project`, `entity`, `run_name_prefix`, `tags`.

> Rule: Every script must read from this file—never hard-code project constants.

---

## 4) Canonical Workflows (Step-by-Step)

### 4.1 Preprocess (a01)
Converts Scryfall bulk (`data/raw/scryfall.json`) → `data/processed/all.txt` + `preprocess_stats.json`.
```bash
python -m src.a01_preprocessor --config configs/config.yaml
```
**Filters:** `lang=en`, `layout in ["normal"]`, require `name`, `type_line`, `rarity`.  
**Normalization:** replace " — " with " | " in type line; `text` collapsed to one line; `pt` only for Creature and numeric/X.  
**Block contract:**
```
<|startofcard|>
theme: Generic
character: N/A
color: W R
type: Legendary Creature | Human
rarity: rare
<|gen_card|>
name: ...
mana_cost: {2}{W}{R}
text: ...
pt: 3/4            # only if Creature
<|endofcard|>
```
**W&B (lite):** logs `pre.input_total`, `pre.kept`, `pre.dropped`, `pre.drop.*`, `pre.rarity.*`, `pre.type.*` when enabled.

### 4.2 Split (a02) — planned
- 80/10/10 with minimal stratification on `rarity`.  
- Avoid only exact duplicates 1:1.  
- Output: `train.txt`, `val.txt`, `test.txt` in `data/processed/`.  
- W&B (lite): `split.*` counts.

### 4.3 Validator (a03) — planned
- Checks strict **field order**, **EOS**, `pt`-only-if-Creature.  
- Optional `color_mana_subset_rule` (default off).  
- Output: `*_validated.jsonl` with `{valid, errors, raw, fields}`.  
- Pass-rate expected ~100% on the dataset; used heavily after generation.

### 4.4 Training & Eval (a04/a05) — planned
- **GPT-2 Large** (full & LoRA) smoke test to verify pipeline end-to-end.  
- **Mistral 7B QLoRA** as main (`a04_train_mistral_lora.py`).  
- Eval: **Perplexity** global + by **type/color/rarity** clusters.  
- W&B: `train.loss`, `val.ppl`, `test.ppl`, and cluster PPLs.

### 4.5 Generation & Curation (a06/a07) — planned
- **Mapping assets:** `src/mapping/mapping_seed.csv` (seed rows), `src/mapping/keywords.json` (synonyms/hints).  
- Batch generate 1k–2k candidates from mapping seed (MHA).  
- Validate (hard rules), score (syntax + light LM/dup heuristics), de-duplicate (`dedup_threshold`), balance by `color/type/rarity`.  
- Final export: **100 cards** set + distribution tables.

---

## 5) W&B Usage (Minimal)
Config:
```yaml
wandb:
  enabled: true
  project: "mtg-mha"
  entity: "your_entity"
  run_name_prefix: "exp"
  tags: ["mtg","mha","student-project"]
```
- Login once with `wandb login` (or set `WANDB_API_KEY`).  
- If `enabled: false`, scripts skip W&B.  
- **Preprocess metrics:** `pre.*` prefix.  
- Avoid heavy artifacts for now.

---

## 6) Data & Format Contracts
**Field order is a hard contract** (see §4.1).  
- `color`: space-separated “W U B R G” subset, possibly empty.  
- `type`: em-dash → `|`; keep category words (e.g., “Legendary Creature | Hero”).  
- `rarity`: `{common, uncommon, rare, mythic}` (lowercase).  
- `mana_cost`: Scryfall style `{...}{...}` (empty allowed).  
- `pt`: only for Creatures and numeric/X.  
- `EOS`: always `<|endofcard|>`.

---

## 7) Coding Standards for Agents
- **Imports:** `from src.a00_utils import ...` or relative (`from .a00_utils import ...`) and **run as module**.  
- **Execution:** from repo root `Magic-Deck-Generator/`:  
  ```bash
  python -m src.a01_preprocessor --config configs/config.yaml
  ```
- **No duplicates** of project constants in code; always read from `configs/config.yaml`.  
- **CLI:** every script takes `--config`. Print clear input/output paths and simple stats.  
- **Files:** keep `aXX_*.py` ordering, and include `src/__init__.py` (even empty).  
- **Seeds:** disabled in this project phase.

---

## 8) Quality Gates (Definition of Done)
**a01_preprocessor**
- `data/processed/all.txt` written and non-empty.  
- `preprocess_stats.json` with plausible counts.  
- If W&B enabled, `pre.*` metrics logged.  
- Manual spot-check of 3 blocks (order & fields OK).

**a02_split (planned)**
- `train/val/test` exist; 80/10/10; rarity distribution reported.  
- No aggressive deduplication (only exact 1:1).

**a03_validate (planned)**
- `*_validated.jsonl` with `valid=True` for dataset; error categories reported if any.  
- Config-driven toggles respected.

---

## 9) Troubleshooting
- **`ModuleNotFoundError: No module named 'src'`** → run as module from repo root: `python -m src.a01_preprocessor`. Ensure `src/__init__.py` exists.
- **Relative import error** → use `-m` or switch to absolute import `from src.a00_utils import ...`.
- **File not found** (scryfall.json) → check `configs/config.yaml → data.raw_path` and file location.
- **W&B not logging** → `wandb.enabled: true`, login done, API key set. Otherwise disable in config.
- **Malformed blocks** → re-check `_format_block` and the block order in outputs.

---

## 10) Change Management
- Any change to **format**, **special tokens**, **regex**, **mapping assets** (keywords/mapping_seed), or **validator** must be reflected **first** in `configs/config.yaml` and/or `src/mapping/*`.  
- Submit PR with:
  - Diff of `configs/config.yaml` and any mapping files,
  - Updated doc in `agents.md` (this file),
  - Short rationale under “Decision Log”.

**PR Checklist (copy in PR description):**
- [ ] Updated `configs/config.yaml` (no hard-coded constants added).
- [ ] Maintained block order & field names contract.
- [ ] Mapping files updated if needed (`src/mapping/keywords.json`, `src/mapping/mapping_seed.csv`).
- [ ] CLI `--config` still supported.
- [ ] W&B keys unchanged or documented (e.g., `pre.*`, `split.*`, `ppl.*`).
- [ ] Tested with `python -m src.a01_preprocessor` on a small sample.

---

## 11) Decision Log (Summary)
- Three special tokens only.  
- Placeholders for `theme/character` in training; real conditioning at generation.  
- Validator toggles in config; start minimal.  
- W&B lite in preprocess; more later.  
- Seeds removed for now.
- Mapping assets live under `src/mapping/` (keywords + seed CSV).

---

## 12) Onboarding — First 90 Minutes
1. Read `agents.md` and `configs/config.yaml`.  
2. Install minimal deps (`pyyaml|omegaconf`, `wandb` optional).  
3. Place Scryfall bulk at `data/raw/scryfall.json`.  
4. From `Magic-Deck-Generator/`, run:  
   ```bash
   python -m src.a01_preprocessor --config configs/config.yaml
   ```
5. Inspect `data/processed/all.txt` and `preprocess_stats.json`.  
6. (Optional) Check W&B dashboard for `pre.*` metrics.
