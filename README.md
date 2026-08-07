# Mininio AI Fine-Tuning

Fine-tuning pipeline for the Mininio Carb & Insulin Calc app's on-device AI assistant.

## What This Repo Does

- **Data synthesis**: Generates ~8,000 multi-turn tool-calling conversations across 10 languages via Gemini API, with deterministic mathematical validation via a mock harness
- **Fine-tuning**: Fine-tunes two candidate models in parallel (LFM2.5-1.2B and Gemma 4 E2B QAT) using Unsloth + TRL SFTTrainer
- **Evaluation**: Evaluates both candidates and picks the winner based on weighted criteria
- **Export**: Exports the winning model for Android deployment (LFM → GGUF, Gemma → LiteRT-LM)

## Setup

```bash
# Sync dependencies (creates .venv if needed)
uv sync

# Create .env file from template
copy .env.example .env      # Windows
cp .env.example .env         # macOS/Linux

# Edit .env — add your GOOGLE_API_KEY (required for data generation only)
```

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

Both candidates use Unsloth for 4-bit QLoRA + TRL SFTTrainer with custom masking
that excludes tool results from the loss (only assistant decisions are trained on).

**Training scripts:**

```bash
# LFM2.5-1.2B-Instruct (~2-3h on RTX 4060 8GB, 4-6h on A100)
python -m finetuning.lfm.train_lfm --data-dir data/output --output-dir finetuning/output

# Gemma 4 E2B (~4-6h on RTX 4060 8GB, 6-8h on A100)
python -m finetuning.gemma.train_gemma --data-dir data/output --output-dir finetuning/output

# With wandb monitoring (set WANDB_API_KEY in .env first)
python -m finetuning.lfm.train_lfm --report-to wandb

# Customize hyperparameters
python -m finetuning.lfm.train_lfm --epochs 3 --batch-size 2 --grad-accum 4 --lr 2e-4 --max-seq-length 4096
```

**Output:** `finetuning/output/{lfm,gemma}/` containing LoRA adapters + merged 16-bit weights.

**Programmatic access:**

```python
from finetuning.common.data_loader import load_dataset_for_model
from finetuning.common.masking import make_masking_fn
from finetuning.common.config import TrainingConfig, detect_env

dataset = load_dataset_for_model("lfm", data_dir="data/output")
env = detect_env()  # "colab" | "kaggle" | "local"
config = TrainingConfig.for_env(data_dir="data/output", output_dir="finetuning/output")
```

Reference scripts:
- `lfm2_5_sft_with_unsloth.py` — LFM2.5 fine-tuning with LoRA + ChatML format
- `gemma_4_finetuning_quide` — Gemma 4 fine-tuning with QLoRA + Unsloth chat template

## Evaluation & Chat & Export

```bash
# Evaluate fine-tuned model (merged 16-bit, no Unsloth needed)
python evaluation/evaluate.py \
  --checkpoint-dir finetuning/output/lfm/merged_16bit \
  --model-type lfm \
  --eval-path data/output/lfm/eval.jsonl

# Evaluate LoRA adapter (requires Unsloth, e.g. on training VM)
python evaluation/evaluate.py \
  --checkpoint-dir finetuning/output/lfm/lora_adapter \
  --model-type lfm \
  --eval-path data/output/lfm/eval.jsonl

# Interactive CLI chat (test model with tool execution)
python -m chat.chat_cli --model-type lfm

# Chat with custom settings and temperature
python -m chat.chat_cli --model-type lfm --settings-idx 2 --temp 0.9

# Export LFM → GGUF for llama.cpp Android runtime
python export/convert_lfm_gguf.py \
  --lora-dir finetuning/output/lfm/lora_adapter \
  --output-dir export/lfm --quant q4_k_m

# Export Gemma → LiteRT-LM for Google Edge Android runtime
python export/convert_gemma_litertlm.py \
  --merged-dir finetuning/output/gemma/merged_16bit \
  --output-dir export/gemma --quant int4
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GOOGLE_API_KEY` | For generation | — | Gemini API key for data generation |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model to use |
| `HF_TOKEN` | No | — | HuggingFace token (avoids anonymous rate limits; required for gated models) |
| `WANDB_API_KEY` | No | — | Enables wandb logging (auto-detected, or use `--report-to wandb`) |
| `WANDB_PROJECT` | No | `mininio-ai-finetuning` | WandB project name |

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
├── tests/                      # pytest unit tests (346 tests)
│   ├── conftest.py
│   ├── test_food_db_loader.py
│   ├── test_mock_harness.py
│   ├── test_validator.py
│   ├── test_scenarios.py
│   ├── test_generate.py
│   ├── test_formatters.py
│   ├── test_assemble.py
│   ├── test_masking.py
│   ├── test_data_loader.py
│   └── test_evaluate.py
│
├── finetuning/                 # Model fine-tuning
│   ├── common/
│   │   ├── config.py           # TrainingConfig, env detection (Colab/Kaggle/local)
│   │   ├── data_loader.py      # HuggingFace Dataset loader + token stats + BOS check
│   │   └── masking.py          # Custom loss masking (excludes tool results from loss)
│   ├── lfm/
│   │   ├── train_lfm.py        # LFM2.5 training script (Unsloth + TRL)
│   │   └── train_lfm.ipynb     # Colab wrapper notebook
│   └── gemma/
│       ├── train_gemma.py      # Gemma 4 training script (Unsloth + TRL)
│       └── train_gemma.ipynb   # Colab wrapper notebook
│
├── evaluation/                 # Model evaluation & selection
│   ├── criteria.py             # Weighted scoring (6 metrics, 40/25/15/10/5/5)
│   └── evaluate.py             # Harness-replay evaluator
│
├── chat/                       # Interactive chat CLI
│   └── chat_cli.py             # Test fine-tuned model interactively (GPU/CPU)
│
├── export/                     # Model export for Android
│   ├── convert_lfm_gguf.py     # LFM → GGUF (q4_k_m/q5_k_m/q8_0/f16)
│   └── convert_gemma_litertlm.py  # Gemma → LiteRT-LM (int4/int8)
│
├── docs/
│   ├── AI_INTEGRATION_PLAN.md  # Android AI integration plan
│   ├── DATA_GENERATION_PLAN.md # Synthetic data pipeline plan
│   └── TRAINING_PROCEDURE.md   # Training design: format, loss, optimizer, hyperparams, post-training
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
- [Training Procedure](docs/TRAINING_PROCEDURE.md) — Format choices, loss masking, optimizer, hyperparameter rationale, post-training
- [AGENTS.md](AGENTS.md) — Instructions for AI coding agents working in this repo
