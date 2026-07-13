# Mininio AI Fine-tuning

Fine-tuning pipeline for the Mininio Carb & Insulin Calc app's on-device AI assistant.

## What This Repo Does

- Generates 8,000 synthetic multi-turn training conversations in 10 languages via Gemini API
- Fine-tunes two candidate models in parallel (LFM2.5-1.2B and Gemma 4 E2B QAT)
- Evaluates both candidates and picks the winner based on weighted criteria
- Exports the winning model for Android deployment

## Structure

```
├── data/                  # Training data generation
│   ├── generate.py        # Gemini-powered conversation generator
│   ├── prompts/           # System prompt templates
│   ├── schemas/           # Tool JSON schemas (6 tools)
│   └── food_db/           # Food database CSVs (10 languages)
│
├── finetuning/            # Model fine-tuning
│   ├── common/            # Shared data loading utilities
│   ├── lfm/               # LFM2.5-1.2B fine-tuning
│   └── gemma/             # Gemma 4 E2B QAT fine-tuning
│
├── evaluation/            # Model evaluation & selection
│
└── export/                # Model export for Android deployment
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Related

- Main app repo: [mininio-carb-insulin-calc](../mininio-carb-insulin-calc)
- Full AI integration plan: [AI_INTEGRATION_PLAN.md](AI_INTEGRATION_PLAN.md)
