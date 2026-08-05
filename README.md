# Mininio AI Fine-Tuning

Fine-tuning pipeline for the Mininio Carb & Insulin Calc app's on-device AI assistant.

## What This Repo Does

- **Data synthesis**: Generates ~8,000 multi-turn tool-calling conversations across 10 languages via Gemini API, with deterministic mathematical validation via a mock harness
- **Fine-tuning**: Fine-tunes two candidate models in parallel (LFM2.5-1.2B and Gemma 4 E2B QAT) using Unsloth + TRL SFTTrainer
- **Evaluation**: Evaluates both candidates and picks the winner based on weighted criteria
- **Export**: Exports the winning model for Android deployment (LFM → GGUF, Gemma → LiteRT-LM)

## Setup

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Create .env file from template
copy .env.example .env      # Windows
cp .env.example .env         # macOS/Linux

# Edit .env — add your GOOGLE_API_KEY
```

Get a Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## Quick Test

```bash
# See generation plan (no API calls, no cost)
python data/generate.py --dry-run

# Dry-run with verbose logging + file output
python data/generate.py --dry-run --verbose --log-file

# Generate 1 test conversation (~$0.001)
python data/generate.py --count-per-lang 1 --languages en --max-concurrent 1

# Generate 10 per language (~$0.15, good for testing)
python data/generate.py --count-per-lang 10 --languages en,el

# Full generation with file logging (~$12)
python data/generate.py --count-per-lang 800 --languages all --log-file

# Full generation with verbose console + harness traces
python data/generate.py --count-per-lang 800 --languages all --verbose --log-file --debug-harness

# Validate existing raw data (no API calls)
python data/generate.py --validate-only

# Migrate raw data (add settings index after structural changes)
python data/migrate_add_settings.py

# Validate settings assignments (cross-check against tool results)
python data/validate_settings.py

# View generation statistics
python data/stats.py
```

## Data Pipeline

The data synthesis pipeline has 8 components:

| Component | File | Purpose |
|-----------|------|---------|
| Food DB Loader | `data/food_db_loader.py` | Parses 10 CSVs (106 foods each), unit normalization, search/sample |
| Mock Harness | `data/mock_harness.py` | Deterministic tool executor — computes carbs, insulin, BG correction. Math is never from Gemini. Also includes 5 rotating user settings configs |
| Scenario Engine | `data/scenarios.py` | Pydantic models, 9 scenario types with weighted distribution |
| Generator | `data/generate.py` | Async Gemini 2.5 Flash API calls with retries, dedup, and validation |
| Validator | `data/validator.py` | 8 validation checks: tool sequence, food IDs, entry IDs, math, units, length, empty turns, context blocks |
| Migration | `data/migrate_add_settings.py` | Annotates raw conversations with per-conversation user settings index |
| Validation | `data/validate_settings.py` | Cross-references settings index against calculate_final tool results |
| Formatters | `data/formatters/` | Converts model-agnostic conversations to LFM2.5 ChatML and Gemma 4 Unsloth chat templates |
| Logging | `data/log_config.py` | Structured logging via loguru — step timers, per-language & global progress tracking with ETA, file logging

**Architecture rule**: Gemini generates semantic content only (user text, assistant decisions, tool call sequences). The mock harness computes all tool results using one of 5 rotating user settings configurations (different glucose thresholds, meal dividers, etc.). Each raw conversation stores its settings index. During assembly, the correct per-conversation settings are injected into the system prompt — the model always sees parameters matching the tool results it's learning from.

### Full Generation

```bash
# Generate 800 conversations per language x 10 languages (~$12)
python data/generate.py --count-per-lang 800 --languages all --log-file

# Migrate annotated data (if regenerating after structural changes)
python data/migrate_add_settings.py

# Validate settings consistency
python data/validate_settings.py

# Assemble train/eval splits for both models
python data/assemble.py
```

Output:
```
data/output/
├── raw/          # Raw conversations per language (JSONL)
├── lfm/          # LFM2.5 ChatML formatted (train.jsonl, eval.jsonl)
└── gemma/        # Gemma 4 Unsloth formatted (train.jsonl, eval.jsonl)
```

## Fine-Tuning

Load formatted data via the shared loader:

```python
from finetuning.common.data_loader import load_dataset_for_model

# Load LFM or Gemma formatted dataset
dataset = load_dataset_for_model("lfm")   # or "gemma"
# Returns DatasetDict with "train" and "eval" splits
```

Reference scripts:
- `lfm2_5_sft_with_unsloth.py` — LFM2.5 fine-tuning with LoRA + ChatML format
- `gemma_4_finetuning_quide` — Gemma 4 fine-tuning with QLoRA + Unsloth chat template

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GOOGLE_API_KEY` | Yes | — | Gemini API key for data generation |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model to use |
| `HF_TOKEN` | For training | — | HuggingFace token for model upload |
| `WANDB_API_KEY` | No | — | Weights & Biases for training metrics |
| `WANDB_PROJECT` | No | `mininio-ai-finetuning` | WandB project name |
| `LFM_MODEL_NAME` | No | `unsloth/LFM2.5-1.2B-Instruct` | LFM base model |
| `GEMMA_MODEL_NAME` | No | `google/gemma-4-e2b-it` | Gemma base model |

## File Structure

```
├── data/                       # Training data generation
│   ├── generate.py             # Gemini-powered conversation generator
│   ├── food_db_loader.py       # Food DB loader + UnitNormalizer
│   ├── scenarios.py            # Pydantic models, 9 scenario types
│   ├── mock_harness.py         # Deterministic tool executor
│   ├── validator.py            # Multi-layer validation (8 checks)
│   ├── assemble.py             # Stratified 90/10 split + formatting
│   ├── migrate_add_settings.py  # Annotate raw data with settings index
│   ├── validate_settings.py     # Cross-check settings vs tool results
│   ├── stats.py                # Generation statistics report
│   ├── log_config.py           # Structured logging (loguru), progress tracking, ETA
│   ├── prompts/                # System prompt templates
│   │   ├── system_generator.txt  # Gemini mega-prompt
│   │   ├── system_lfm.txt        # LFM2.5 ChatML system prompt
│   │   └── system_gemma.txt      # Gemma 4 Unsloth system prompt
│   ├── schemas/
│   │   └── tools.json          # 6 tool JSON schemas
│   ├── formatters/
│   │   ├── lfm_formatter.py    # ChatML converter
│   │   └── gemma_formatter.py  # Gemma Unsloth converter
│   └── food_db/                # Food database CSVs (10 languages)
│
├── tests/                      # pytest unit tests (276 tests)
│   ├── conftest.py
│   ├── test_food_db_loader.py
│   ├── test_mock_harness.py
│   ├── test_validator.py
│   ├── test_scenarios.py
│   ├── test_generate.py
│   ├── test_formatters.py
│   └── test_assemble.py
│
├── finetuning/                 # Model fine-tuning
│   ├── common/
│   │   └── data_loader.py      # HuggingFace Dataset loader
│   ├── lfm/                    # LFM2.5 fine-tuning notebook
│   └── gemma/                  # Gemma 4 fine-tuning notebook
│
├── evaluation/                 # Model evaluation & selection
├── export/                     # GGUF & LiteRT-LM conversion
│
├── docs/
│   ├── AI_INTEGRATION_PLAN.md  # Android AI integration plan
│   └── DATA_GENERATION_PLAN.md # Synthetic data pipeline plan
├── lfm2_5_sft_with_unsloth.py  # LFM reference training script
├── gemma_4_finetuning_quide    # Gemma reference training guide
├── AGENTS.md                   # Instructions for AI coding agents
├── .env.example                # Environment variable template
└── requirements.txt            # Python dependencies
```

## Cost

| Operation | Est. cost |
|-----------|-----------|
| 1 test conversation (--count-per-lang 1) | ~$0.001 |
| 10 per language test (--count-per-lang 10 --languages en,el) | ~$0.03 |
| Full generation (800 × 10 languages) | ~$12 |

Based on Gemini 2.5 Flash pricing ($0.15/1M input, $0.60/1M output).

## Documentation

- [AI Integration Plan](docs/AI_INTEGRATION_PLAN.md) — Android app tool schemas, agentic loop, fine-tuning strategy
- [Data Generation Plan](docs/DATA_GENERATION_PLAN.md) — Detailed synthesis pipeline design
- [AGENTS.md](AGENTS.md) — Instructions for AI coding agents working in this repo
