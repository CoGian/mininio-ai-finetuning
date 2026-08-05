# AGENTS.md — Mininio AI Fine-Tuning

## Project Overview

Synthetic data pipeline + fine-tuning for the Mininio carb-counting AI assistant.
Generates ~8,000 tool-calling conversations across 10 languages, then fine-tunes
two candidate models (LFM2.5 + Gemma 4) via Unsloth + TRL SFTTrainer.

## Environment

- Python 3.11+, virtualenv at `.venv/`
- Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`
- Install: `uv pip install -r requirements.txt`

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
```

## Testing

- `pytest tests/ -v` — 276 tests across 8 files
- Run from project root; fixtures use inline data (no file I/O except CSV load tests)
- Tests cover: UnitNormalizer, MockHarness, all 8 validators, formatters, work distribution, hashing, Pydantic models, assemble logic

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
