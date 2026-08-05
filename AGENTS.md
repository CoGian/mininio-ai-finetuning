# AGENTS.md — Mininio AI Fine-Tuning

## Project Overview

Synthetic data pipeline + fine-tuning for the Mininio carb-counting AI assistant.
Generates ~8,000 tool-calling conversations across 10 languages, then fine-tunes
two candidate models (LFM2.5 + Gemma 4) via Unsloth + TRL SFTTrainer.

## Environment

- Python 3.11+, virtualenv at `.venv/`
- Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (required for data generation only)
- Install: `uv sync`

## Code Conventions

- No comments unless specifically asked
- Follow existing patterns in each file (mimic imports, naming, structure)
- Pydantic models live in `data/scenarios.py`
- Type hints required on all function signatures
- `Optional[T]` not `T | None` (consistent with existing code)
- Ruff is configured in `pyproject.toml` — run manually with `ruff check .`
- Do not commit `.env` or `data/output/`

## Key Files

| File | Purpose |
|------|---------|
| `data/food_db_loader.py` | Loads 10 CSVs, UnitNormalizer, search/sample |
| `data/scenarios.py` | Pydantic models, 9 scenario types, weights |
| `data/mock_harness.py` | Deterministic tool executor + 5 rotating user settings configs |
| `data/generate.py` | Gemini-powered conversation generator (async) |
| `data/validator.py` | 8 validation checks on every generated conversation |
| `data/log_config.py` | Structured logging (loguru), StepTimer, ProgressTracker, ETA |
| `data/migrate_add_settings.py` | Annotates raw conversations with per-conversation settings idx |
| `data/validate_settings.py` | Cross-references assigned settings vs calculate_final results |
| `data/formatters/` | LFM ChatML + Gemma Unsloth format converters |
| `data/assemble.py` | Stratified 90/10 split, per-conversation settings injection |
| `data/stats.py` | Per-language/scenario distribution report |
| `finetuning/common/config.py` | TrainingConfig dataclass, env detection (Colab/Kaggle/local) |
| `finetuning/common/masking.py` | Custom loss masking — excludes tool results from loss |
| `finetuning/common/data_loader.py` | HuggingFace Dataset loader, token stats, BOS duplication check |
| `finetuning/lfm/train_lfm.py` | LFM2.5 training script (Unsloth + TRL SFTTrainer) |
| `finetuning/gemma/train_gemma.py` | Gemma 4 training script (Unsloth + TRL SFTTrainer) |
| `evaluation/criteria.py` | Weighted scoring: 6 metrics (40/25/15/10/5/5) |
| `evaluation/evaluate.py` | Harness-replay evaluator with tool result injection |
| `export/convert_lfm_gguf.py` | LFM → GGUF (q4_k_m/q5_k_m/q8_0/f16) |
| `export/convert_gemma_litertlm.py` | Gemma → LiteRT-LM (int4/int8) |
| `docs/TRAINING_PROCEDURE.md` | Training design rationale: format, loss, optimizer, hyperparams, post-training |

## Common Commands

```bash
# See plan (no API calls, no cost)
python data/generate.py --dry-run

# Verbose dry-run with file logging
python data/generate.py --dry-run --verbose --log-file

# Generate 1 test conversation (~$0.001)
python data/generate.py --count-per-lang 1 --languages en

# Generate 10 per language (~$0.15, good for testing)
python data/generate.py --count-per-lang 10 --languages en,el

# Full generation with file logging (~$12)
python data/generate.py --count-per-lang 800 --languages all --log-file

# Full generation with verbose console + harness traces
python data/generate.py --count-per-lang 800 --languages all --verbose --log-file --debug-harness

# Validate existing raw data (no API calls)
python data/generate.py --validate-only

# Assemble train/eval splits for both models
python data/assemble.py

# Annotate raw data with user settings index
python data/migrate_add_settings.py

# Validate settings assignments against tool results
python data/validate_settings.py

# Generate stats report
python data/stats.py

# Run tests
pytest tests/ -v

# Train LFM2.5 (local)
python -m finetuning.lfm.train_lfm --data-dir data/output --output-dir finetuning/output

# Train Gemma 4 (local)
python -m finetuning.gemma.train_gemma --data-dir data/output --output-dir finetuning/output

# Train with wandb monitoring (set WANDB_API_KEY in .env)
python -m finetuning.lfm.train_lfm --report-to wandb

# Evaluate fine-tuned model
python evaluation/evaluate.py --checkpoint-dir finetuning/output/lfm/lora_adapter --model-type lfm --eval-path data/output/lfm/eval.jsonl

# Export LFM → GGUF
python export/convert_lfm_gguf.py --lora-dir finetuning/output/lfm/lora_adapter --output-dir export/lfm --quant q4_k_m

# Export Gemma → LiteRT-LM
python export/convert_gemma_litertlm.py --merged-dir finetuning/output/gemma/merged_16bit --output-dir export/gemma --quant int4
```

## Testing

- `pytest tests/ -v` — 295 tests across 9 files
- Run from project root; fixtures use inline data (no file I/O except CSV load tests)
- Tests cover: UnitNormalizer, MockHarness, all 8 validators, formatters, work distribution, hashing, Pydantic models, assemble logic, custom masking

## Architecture Rule

- Gemini generates **SEMANTIC content** only (user text, assistant decisions, tool call sequences)
- Mock harness computes **ALL tool results** (math is deterministic, never from Gemini)
- Generation cycles through 5 user settings configs (different glucose thresholds, meal dividers, etc.)
- Each raw conversation stores its `user_settings_idx` — assembly injects the matching parameters into the system prompt
- Same conversations flow through both formatters — only template tokens differ

## Dependencies

- `google-genai` (Gemini API), `pydantic`, `loguru`, `python-dotenv`, `datasets`, `transformers`, `trl`, `unsloth`
- `pytest>=8.0` in `[dependency-groups].dev`
- `.env` auto-loads via `python-dotenv` in `data/__init__.py` — no manual `export` needed
- All pinned versions in `requirements.txt`
