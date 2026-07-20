# Synthetic Dataset Creation — Final Implementation Plan

## Goal

Build a pipeline producing **~8,000 multi-turn tool-calling conversations** in **10 languages**, formatted for two model candidates (LFM2.5 ChatML + Gemma 4 Unsloth chat template), ready for LoRA/QLoRA SFT fine-tuning via Unsloth + TRL `SFTTrainer`. The **same semantic examples** flow through both formatters; only the template tokens differ.

---

## Prerequisites & Constraints

- **Source of truth**: `AI_INTEGRATION_PLAN.md` Sections 2, 4, 5 — 6 tools, agentic loop, scenario distribution
- **Reference code**: `lfm2_5_sft_with_unsloth.py` (ChatML + `train_on_responses_only`), `gemma_4_finetuning_quide` (Unsloth Gemma-4 chat template, not LiteRT-LM format)
- **Generator**: Gemini 2.5 Flash via `google-genai` SDK — semantic content only; tool results computed deterministically by mock harness
- **Food DB**: 10 CSVs at `data/food_db/` — 106 foods each, semicolon-delimited, localized names
- **Training**: Unsloth with LoRA (LFM2.5) / QLoRA (Gemma 4) + TRL `SFTTrainer`
- **Budget**: ~$11 total (17M output + ~4.25M input tokens at Gemini 2.5 Flash pricing)

### Key Format Reference (from actual code)

**LFM2.5 ChatML** (`lfm2_5_sft_with_unsloth.py:103-109`):
```
<|startoftext|><|im_start|>system  ...  <|im_end|>
<|im_start|>user  ...  <|im_end|>
<|im_start|>assistant  ...  <|im_end|>
```
Tool calls embedded as text between `<|tool_call_start|>` / `<|tool_call_end|>` tokens within assistant turns.
`train_on_responses_only` uses:
```python
instruction_part = "<|im_start|>user\n"
response_part = "<|im_start|>assistant\n"
```

**Gemma 4 Unsloth** (`gemma_4_finetuning_quide:358-362, 430-433`):
```
<bos><|turn>user  ...  <turn|>
<|turn>model  ...  <turn|>
```
`train_on_responses_only` uses:
```python
instruction_part = "<|turn>user\n"
response_part = "<|turn>model\n"
```
Function calls are embedded as text within model turns — there are no native function-call tokens in the HuggingFace/Unsloth tokenizer. The `<start_function_call>` / `<end_function_call>` format from the Android LiteRT-LM runtime is a separate concern handled at export time.

> **Design Decision**: Training data uses Unsloth's chat template format. The Android runtime format (`<start_function_call>`, `<escape>`, etc.) is a post-training conversion concern handled by the LiteRT-LM conversion tool (`export/convert_gemma_litertlm.py`). For Gemma, function calls in training data are formatted as structured text within `<|turn>model` blocks.

---

## Generative vs Deterministic Boundary

The key architectural insight of this pipeline:

| Layer | Who generates it | Why |
|-------|-----------------|-----|
| User utterances (natural language) | Gemini Flash | Needs linguistic creativity across 10 languages |
| Assistant tool calls (decisions) | Gemini Flash | Needs to decide which tools to call in what order |
| Assistant text responses | Gemini Flash | Needs conversational fluency |
| Tool results (math) | **Mock harness (deterministic)** | Must be mathematically correct, always |
| Context blocks (tally + IDs) | **Mock harness (deterministic)** | Must match actual harness behavior exactly |

Gemini never touches arithmetic. The mock harness computes everything and validates all tool call sequences.

---

## Phase 1 — Foundation: Tool Schemas & Food DB Loader

### `data/schemas/tools.json` — POPULATE (currently empty)

Define all 6 tool schemas in a JSON array matching `AI_INTEGRATION_PLAN.md` Section 2 exactly. This is the single source of truth consumed by the generator prompt, mock harness, validator, and formatters.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "definitions": {
    "FoodResult": {
      "type": "object",
      "properties": {
        "id": { "type": "integer" },
        "name": { "type": "string" },
        "carbs_per_100g": { "type": ["number", "null"] },
        "carbs_per_piece": { "type": ["number", "null"] },
        "has_grams_mode": { "type": "boolean" },
        "has_pieces_mode": { "type": "boolean" }
      },
      "required": ["id", "name", "has_grams_mode", "has_pieces_mode"]
    },
    "TallyItem": {
      "type": "object",
      "properties": {
        "food_id": { "type": "integer" },
        "quantity": { "type": "number" },
        "unit": { "type": "string", "enum": ["g", "ml", "pcs", "cup", "tbsp"] }
      },
      "required": ["food_id", "quantity", "unit"]
    },
    "TallyEntry": {
      "type": "object",
      "properties": {
        "entry_id": { "type": "integer" },
        "food_name": { "type": "string" },
        "quantity": { "type": "number" },
        "unit": { "type": "string" },
        "carbs": { "type": "number" }
      }
    }
  },
  "tools": [
    {
      "name": "search_foods",
      "description": "Search the nutrition database for multiple food names. Each query returns matching food items with their IDs, carbohydrate values per standard portion, and supported measurement modes (grams and/or pieces). The returned food IDs must be used in subsequent add_foods_to_tally calls — IDs from anywhere else will be rejected.",
      "parameters": {
        "type": "object",
        "properties": {
          "queries": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Food names to search (e.g. [\"potatoes\", \"bread\", \"rice\"])"
          }
        },
        "required": ["queries"]
      },
      "returns": {
        "type": "array",
        "items": {
          "type": "array",
          "items": { "$ref": "#/definitions/FoodResult" }
        },
        "description": "One result array per query in the same order. Empty array [] if a query matches nothing."
      }
    },
    {
      "name": "add_foods_to_tally",
      "description": "Add food items to the calculation tally. Computes carbs per item as (quantity × food_carbs) / standard_quantity using the food's carbohydrate values from the search results. Every food_id must be in the KNOWN FOOD IDS set (populated by prior search_foods calls). Rejected food_ids return an error.",
      "parameters": {
        "type": "object",
        "properties": {
          "items": {
            "type": "array",
            "items": { "$ref": "#/definitions/TallyItem" },
            "description": "Food items to add with their quantities and units"
          }
        },
        "required": ["items"]
      },
      "returns": {
        "type": "object",
        "properties": {
          "entries": {
            "type": "array",
            "items": { "$ref": "#/definitions/TallyEntry" }
          },
          "tally_total": { "type": "number" }
        }
      }
    },
    {
      "name": "remove_foods_from_tally",
      "description": "Remove entries from the tally by their entry IDs. Entry IDs come from the CURRENT TALLY context block or from add_foods_to_tally return values. Every entry_id must exist in the current tally — invalid IDs return an error.",
      "parameters": {
        "type": "object",
        "properties": {
          "entry_ids": {
            "type": "array",
            "items": { "type": "integer" },
            "description": "Entry IDs to remove (visible in CURRENT TALLY)"
          }
        },
        "required": ["entry_ids"]
      },
      "returns": {
        "type": "object",
        "properties": {
          "removed": { "type": "integer", "description": "Number of entries actually removed" },
          "tally_total": { "type": "number", "description": "New tally total after removal" }
        }
      }
    },
    {
      "name": "calculate_final",
      "description": "Compute the final insulin dose. Requires at least one food in the tally (unless checking glucose-only, which will return an error). Absorbs meal time and blood glucose as optional parameters — no separate setter calls needed. Meal time defaults: morning 4:00-12:00 (divider 14), midday 12:00-17:00 (divider 15), evening 17:00-4:00 (divider 12). Glucose correction is max(0, (blood_glucose - baseline) / divisor) only if blood_glucose >= threshold.",
      "parameters": {
        "type": "object",
        "properties": {
          "meal_time": {
            "type": "string",
            "enum": ["morning", "midday", "evening"],
            "description": "Meal time period. Omit to let the tool infer from time of day."
          },
          "meal_hour": {
            "type": "integer",
            "minimum": 0,
            "maximum": 23,
            "description": "Specific hour mentioned by user. Null/omit if not specified."
          },
          "blood_glucose": {
            "type": "number",
            "description": "Blood glucose in mg/dL. Omit if not checking glucose."
          }
        },
        "required": []
      },
      "returns": {
        "type": "object",
        "properties": {
          "final_result": { "type": "number", "description": "Total insulin dose" },
          "food_insulin": { "type": "number", "description": "Insulin for carbs only" },
          "glucose_correction": { "type": "number", "description": "Correction insulin for high BG" },
          "glucose_skipped": { "type": "boolean", "description": "True if BG was below threshold" },
          "tally_total": { "type": "number" },
          "meal_divider": { "type": "integer" },
          "meal_time": { "type": "string" },
          "meal_hour": { "type": ["integer", "null"] },
          "blood_glucose": { "type": ["number", "null"] },
          "threshold": { "type": "number" },
          "baseline": { "type": "number" },
          "divisor": { "type": "number" },
          "breakdown_food": { "type": "string" },
          "breakdown_glucose": { "type": "string" }
        }
      }
    },
    {
      "name": "get_tally_summary",
      "description": "Get current calculation state. Normally this information is already available in the CURRENT TALLY context block appended to every user message — use this only as a safety net if you've lost track of state after a complex correction sequence.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "returns": {
        "type": "object",
        "properties": {
          "entries": {
            "type": "array",
            "items": { "$ref": "#/definitions/TallyEntry" }
          },
          "total_carbs": { "type": "number" },
          "food_insulin": { "type": "number" },
          "meal_time": { "type": ["string", "null"] },
          "meal_hour": { "type": ["integer", "null"] },
          "blood_glucose": { "type": ["number", "null"] },
          "glucose_enabled": { "type": "boolean" }
        }
      }
    },
    {
      "name": "clear_all",
      "description": "Clear all calculation data including the tally entries and the known food IDs set. Use this to start completely fresh — typically invoked when the user wants to begin a new calculation or says something like \"start over\" or \"reset\".",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "returns": {
        "type": "object",
        "properties": {
          "success": { "type": "boolean" }
        }
      }
    }
  ]
}
```

### `data/food_db_loader.py` — NEW

Python module that loads and queries the 10 language CSVs.

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FoodItem:
    id: int                      # 1-based, per-language (line number in CSV)
    name: str                    # Localized food name
    standard_quantity_g: Optional[float]   # Gram portion (e.g. 120 for Apple)
    standard_quantity_pcs: Optional[float] # Piece portion (e.g. 1 for White bread)
    carbs: float                 # Carbs per standard portion (always in grams)
    carbs_per_100g: Optional[float]       # Derived: (carbs / standard_g) * 100
    carbs_per_piece: Optional[float]      # Derived: carbs / standard_pcs
    has_grams_mode: bool         # True if the food has a gram/ml portion defined
    has_pieces_mode: bool        # True if the food has a piece/cup portion defined
    is_liquid: bool              # True if the unit is "ml" (not "g")
    category: str                # "breads", "fruits", "vegetables", "dairy", "legumes", "other"

def load_food_db(lang: str) -> List[FoodItem]:
    """Parse data/food_db/{lang}.csv. Semi-colon delimited.
    Header: Food;Quantity(g/ml);Quantity(item);Carbohydrates(g)"""

def search_foods(db: List[FoodItem], queries: List[str]) -> List[List[FoodItem]]:
    """Case-insensitive substring match per query. Returns list of result lists."""

def sample_foods(db: List[FoodItem], n: int, categories: Optional[List[str]] = None) -> List[FoodItem]:
    """Random sampling, optionally filtered by category."""

def get_all_food_names(db: List[FoodItem]) -> str:
    """Return a formatted string of food names (15-20 items) for the Gemini prompt."""
```

**CSV parsing details**: The CSV has 4 columns delimited by `;`:
1. `Food` (string) — the food name
2. `Quantity(g/ml)` — nullable float, the standard gram/milliliter portion (strip trailing " g" or " ml")
3. `Quantity(item)` — nullable, the standard piece/cup portion (parse number from prefix)
4. `Carbohydrates(g)` — float, carbs per standard portion (strip trailing " g")

Example row: `Apple;120 g;;15 g` → standard_quantity_g=120, standard_quantity_pcs=None, carbs=15.0

Derivation:
- `carbs_per_100g = (carbs / standard_quantity_g) * 100` (if standard_quantity_g exists)
- `carbs_per_piece = carbs / standard_quantity_pcs` (if standard_quantity_pcs exists)
- `has_grams_mode = standard_quantity_g is not None`
- `has_pieces_mode = standard_quantity_pcs is not None`
- `is_liquid = the Quantity(g/ml) unit is "ml"`

Categories are assigned based on known food types (fruits by sugar content > 10g per serving, vegetables by 5g carb content, dairy by 12g carb content and milk/yogurt/kefir names, etc.).

---

## Phase 2 — Scenario Engine (Model-Agnostic)

### `data/scenarios.py` — NEW

Define Pydantic models for the model-agnostic conversation format and scenario type definitions.

```python
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class ScenarioType(str, Enum):
    SIMPLE_SINGLE_FOOD = "SIMPLE_SINGLE_FOOD"
    MULTIPLE_FOODS_NO_GLUCOSE = "MULTIPLE_FOODS_NO_GLUCOSE"
    MULTIPLE_FOODS_WITH_GLUCOSE = "MULTIPLE_FOODS_WITH_GLUCOSE"
    EXPLICIT_MEAL_TIME = "EXPLICIT_MEAL_TIME"
    AMBIGUOUS_FOOD = "AMBIGUOUS_FOOD"
    CORRECTION_REMOVAL = "CORRECTION_REMOVAL"
    GLUCOSE_ONLY_CHECK = "GLUCOSE_ONLY_CHECK"
    INCOMPLETE_INFO = "INCOMPLETE_INFO"
    FOOD_NOT_FOUND = "FOOD_NOT_FOUND"

class ToolCall(BaseModel):
    name: str
    arguments: dict

class Turn(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: Optional[str] = None           # Natural language text (user utterances, assistant final responses)
    action: Optional[Literal["text", "tool_call"]] = None  # For assistant turns
    tool_calls: Optional[List[ToolCall]] = None  # When action="tool_call"
    tool_results: Optional[dict] = None       # For tool turns, filled by mock harness
    context_block: Optional[str] = None        # [CURRENT TALLY: ...] / [KNOWN FOOD IDS: ...]

class Conversation(BaseModel):
    scenario_type: str
    language: str
    turns: List[Turn]
```

**Scenario weights** (matching `AI_INTEGRATION_PLAN.md:513-523`):

```python
SCENARIO_WEIGHTS = {
    ScenarioType.SIMPLE_SINGLE_FOOD:        0.15,   # 1,200 of 8,000
    ScenarioType.MULTIPLE_FOODS_NO_GLUCOSE: 0.20,   # 1,600
    ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE: 0.25, # 2,000
    ScenarioType.EXPLICIT_MEAL_TIME:        0.10,   # 800
    ScenarioType.AMBIGUOUS_FOOD:            0.10,   # 800
    ScenarioType.CORRECTION_REMOVAL:        0.08,   # 640
    ScenarioType.GLUCOSE_ONLY_CHECK:        0.05,   # 400
    ScenarioType.INCOMPLETE_INFO:           0.05,   # 400
    ScenarioType.FOOD_NOT_FOUND:            0.02,   # 160
}
```

**Expected tool sequences** (for validation):

```python
EXPECTED_SEQUENCES = {
    ScenarioType.SIMPLE_SINGLE_FOOD: [
        ["search_foods", "add_foods_to_tally", "text"]  # No calculate_final (incomplete)
    ],
    ScenarioType.MULTIPLE_FOODS_NO_GLUCOSE: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"]
    ],
    ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"]
    ],
    ScenarioType.EXPLICIT_MEAL_TIME: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"]
        # calculate_final must include meal_time or meal_hour
    ],
    ScenarioType.AMBIGUOUS_FOOD: [
        ["search_foods", "text"],           # Assistant asks clarification
        ["add_foods_to_tally", "calculate_final", "text"]  # After clarification
    ],
    ScenarioType.CORRECTION_REMOVAL: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"],  # Initial
        # After user correction:
        ["remove_foods_from_tally", "calculate_final", "text"]
    ],
    ScenarioType.GLUCOSE_ONLY_CHECK: [
        ["calculate_final", "text"]  # calculate_final fails with error, assistant explains
    ],
    ScenarioType.INCOMPLETE_INFO: [
        ["search_foods", "text"],  # Assistant asks for quantity/unit
        ["add_foods_to_tally", "calculate_final", "text"]  # After user provides info
    ],
    ScenarioType.FOOD_NOT_FOUND: [
        ["search_foods", "text"]   # search returns empty, assistant reports not found
    ],
}
```

### `data/mock_harness.py` — NEW

Deterministic tool executor that mirrors the Android `ToolExecutor` behavior from `AI_INTEGRATION_PLAN.md:172-199`.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional

DEFAULT_SETTINGS = {
    "glucose_threshold": 130.0,
    "glucose_baseline": 100.0,
    "glucose_divisor": 40.0,
    "meal_dividers": {"morning": 14, "midday": 15, "evening": 12},
    "meal_ranges": {
        "morning": (4, 12),
        "midday": (12, 17),
        "evening": (17, 4),
    },
}

class MockHarness:
    """Mirrors Android ToolExecutor + DietCalculator behavior exactly."""

    def __init__(self, food_db: List[FoodItem], settings: dict = None):
        self.food_db = {f.id: f for f in food_db}
        self.settings = settings or DEFAULT_SETTINGS
        self.known_food_ids: set[int] = set()
        self.tally_entries: list[dict] = []   # [{entry_id, food_id, food_name, quantity, unit, carbs}]
        self.meal_time: Optional[str] = None
        self.meal_hour: Optional[int] = None
        self.blood_glucose: Optional[float] = None
        self._next_entry_id: int = 1

    def execute(self, tool_call) -> dict:
        method = getattr(self, f"_exec_{tool_call.name}")
        return method(tool_call.arguments)

    def _exec_search_foods(self, args: dict) -> dict:
        queries = args["queries"]
        results = []
        ids = []
        for q in queries:
            matches = [f for f in self.food_db.values()
                       if q.lower() in f.name.lower()]
            for m in matches:
                ids.append(m.id)
                self.known_food_ids.add(m.id)
            results.append([_serialize_food(m) for m in matches])
        return {"results": results}

    def _exec_add_foods_to_tally(self, args: dict) -> dict:
        items = args["items"]
        entries = []
        total = 0.0
        for item in items:
            fid = item["food_id"]
            if fid not in self.known_food_ids:
                return {"error": f"Unknown food_id: {fid}. Search first."}
            food = self.food_db[fid]
            carbs = _compute_carbs(food, item["quantity"], item["unit"])
            entry = {
                "entry_id": self._next_entry_id,
                "food_name": food.name,
                "quantity": item["quantity"],
                "unit": item["unit"],
                "carbs": round(carbs, 1),
            }
            self._next_entry_id += 1
            entries.append(entry)
            total += carbs
            self.tally_entries.append(entry)
        return {"entries": entries, "tally_total": round(total, 1)}

    def _exec_remove_foods_from_tally(self, args: dict) -> dict:
        eids = set(args["entry_ids"])
        removed = 0
        new_tally = []
        for entry in self.tally_entries:
            if entry["entry_id"] in eids:
                removed += 1
            else:
                new_tally.append(entry)
        self.tally_entries = new_tally
        total = sum(e["carbs"] for e in self.tally_entries)
        return {"removed": removed, "tally_total": round(total, 1)}

    def _exec_calculate_final(self, args: dict) -> dict:
        if not self.tally_entries:
            return {"error": "Add at least one food first."}

        meal_time = args.get("meal_time") or self._infer_meal_time(args.get("meal_hour"))
        meal_hour = args.get("meal_hour")
        blood_glucose = args.get("blood_glucose")

        divider = self.settings["meal_dividers"].get(meal_time, 15)
        tally_total = sum(e["carbs"] for e in self.tally_entries)

        food_insulin = tally_total / divider

        threshold = self.settings["glucose_threshold"]
        baseline = self.settings["glucose_baseline"]
        divisor = self.settings["glucose_divisor"]

        glucose_correction = 0.0
        glucose_skipped = True
        if blood_glucose is not None and blood_glucose >= threshold:
            glucose_correction = max(0, (blood_glucose - baseline) / divisor)
            glucose_skipped = False

        final = food_insulin + glucose_correction
        return {
            "final_result": round(final, 2),
            "food_insulin": round(food_insulin, 2),
            "glucose_correction": round(glucose_correction, 2),
            "glucose_skipped": glucose_skipped,
            "tally_total": round(tally_total, 1),
            "meal_divider": divider,
            "meal_time": meal_time,
            "meal_hour": meal_hour,
            "blood_glucose": blood_glucose,
            "threshold": threshold,
            "baseline": baseline,
            "divisor": divisor,
            "breakdown_food": f"{tally_total:.1f}g / {divider} = {food_insulin:.2f}U",
            "breakdown_glucose": f"({blood_glucose} - {baseline}) / {divisor} = {glucose_correction:.2f}U",
        }

    def _exec_get_tally_summary(self, args: dict) -> dict:
        entries = [dict(e) for e in self.tally_entries]
        return {
            "entries": entries,
            "total_carbs": sum(e["carbs"] for e in self.tally_entries),
            "food_insulin": 0.0,  # Not computed yet
            "meal_time": self.meal_time,
            "meal_hour": self.meal_hour,
            "blood_glucose": self.blood_glucose,
            "glucose_enabled": self.blood_glucose is not None,
        }

    def _exec_clear_all(self, args: dict) -> dict:
        self.known_food_ids.clear()
        self.tally_entries.clear()
        self.meal_time = None
        self.meal_hour = None
        self.blood_glucose = None
        self._next_entry_id = 1
        return {"success": True}

    def _infer_meal_time(self, hour: Optional[int]) -> str:
        if hour is None:
            return "midday"  # Default
        ranges = self.settings["meal_ranges"]
        for period, (start, end) in ranges.items():
            if end > start:
                if start <= hour < end:
                    return period
            else:  # overnight range
                if hour >= start or hour < end:
                    return period
        return "midday"

    def get_context_block(self) -> str:
        if not self.tally_entries:
            return "[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"

        tally_lines = []
        for e in self.tally_entries:
            tally_lines.append(
                f"  {e['food_name']} {e['quantity']}{e['unit']} "
                f"= {e['carbs']}g (entry_id: {e['entry_id']})"
            )

        known_lines = []
        for fid in self.known_food_ids:
            known_lines.append(f"{self.food_db[fid].name}({fid})")

        tally_str = f"[CURRENT TALLY: {len(self.tally_entries)} items, "
        tally_str += f"{sum(e['carbs'] for e in self.tally_entries):.1f}g total]\n"
        tally_str += "\n".join(tally_lines)
        known_str = f"\n\n[KNOWN FOOD IDS: {', '.join(known_lines)}]"
        return tally_str + known_str

    def reset(self):
        self._exec_clear_all({})


# --- Internal helpers ---

def _serialize_food(f: FoodItem) -> dict:
    return {
        "id": f.id,
        "name": f.name,
        "carbs_per_100g": f.carbs_per_100g,
        "carbs_per_piece": f.carbs_per_piece,
        "has_grams_mode": f.has_grams_mode,
        "has_pieces_mode": f.has_pieces_mode,
    }

def _compute_carbs(food: FoodItem, quantity: float, unit: str) -> float:
    """(quantity × food_carbs) / standard_quantity"""
    if unit in ("g", "ml") and food.has_grams_mode:
        return (quantity * food.carbs) / food.standard_quantity_g
    elif unit in ("pcs", "cup", "tbsp") and food.has_pieces_mode:
        return (quantity * food.carbs) / food.standard_quantity_pcs
    else:
        raise ValueError(f"Cannot compute carbs for {unit} on {food.name}")
```

### `data/prompts/system_generator.txt` — POPULATE

The mega-prompt for Gemini Flash. Variables in `%{...}` are injected at generation time.

```
You are a synthetic training data generator for a carb-counting AI assistant integrated into the Mininio diabetes app. Your job is to generate realistic, diverse, multi-turn tool-calling conversations in %{language_name} language (%{language_code}).

## ROLE & CONTEXT

The AI assistant helps people with diabetes calculate insulin doses. It has access to 6 tools for searching a nutrition database, managing a food tally, and computing final insulin doses. The assistant follows a strict tool-calling workflow:
1. Search for foods in the database (to get valid food IDs)
2. Add foods to the tally with quantities
3. Calculate the final insulin dose (includes blood glucose correction if needed)
4. Present results conversationally

IMPORTANT: Every user message includes a [CURRENT TALLY] and [KNOWN FOOD IDS] block. The assistant MUST trust this block over any memory of previous state. The block shows exactly which foods are in the tally and which food IDs are known from prior searches.

## USER SETTINGS

The current user has these settings:
- Glucose threshold: %{glucose_threshold} mg/dL (only correct BG above this)
- Glucose baseline: %{glucose_baseline} mg/dL
- Glucose divisor: %{glucose_divisor} mg/dL per insulin unit
- Meal dividers: Morning=%{morning_divider}, Midday=%{midday_divider}, Evening=%{evening_divider}
- Meal time ranges: Morning 4:00-12:00, Midday 12:00-17:00, Evening 17:00-4:00

## CALCULATION RULES

- Carbs per food = (quantity × food_carbs_per_standard_portion) / standard_portion_size
- Food insulin = total_carbs / meal_divider
- Glucose correction = max(0, (blood_glucose - baseline) / divisor), only if BG >= threshold
- Final dose = food_insulin + glucose_correction
- The assistant NEVER does math itself — math is done by the tools and the assistant reads the results

## TOOLS AVAILABLE

%{tool_schemas}

## FOOD DATABASE (sampled from %{language_code})

The following foods are available in the %{language_name} nutrition database. Use these exact names and IDs in the conversation:

%{food_db_sample}

## SCENARIO TYPE TO GENERATE

%{scenario_instructions}

## CONTEXT BLOCK FORMAT

Every user message MUST include a context block appended after the user's utterance:

[CURRENT TALLY: {status}]
  {food_name} {quantity}{unit} = {carbs}g (entry_id: {id})
  ...more entries...

[KNOWN FOOD IDS: {name}({id}), ...more...]

When the tally is empty, use:
[CURRENT TALLY: empty]
[KNOWN FOOD IDS: none]

The context block shows the EXACT state before the assistant processes the user's message. The assistant must never hallucinate food IDs or tally entries — it only uses what's shown in the context block or returned by tools.

## OUTPUT FORMAT

Generate a JSON object with this exact schema:

{
  "turns": [
    {
      "role": "user",
      "content": "<natural language user utterance in %{language_name}>",
      "context_block": "<the [CURRENT TALLY] / [KNOWN FOOD IDS] block>"
    },
    {
      "role": "assistant",
      "action": "tool_call",
      "tool_calls": [
        {"name": "search_foods", "arguments": {"queries": ["food1", "food2"]}}
      ]
    },
    {
      "role": "tool",
      "tool_results": {}  // LEAVE EMPTY — will be filled by the mock harness
    },
    {
      "role": "assistant",
      "action": "text",
      "content": "<natural language response in %{language_name}>"
    }
  ]
}

## RULES

1. Generate the conversation entirely in %{language_name}. Use natural, colloquial language appropriate for a chat interface.
2. Every user turn MUST have a context_block field showing the tally state at that point.
3. Every assistant turn MUST have an action field: "tool_call" or "text".
4. Tool call arguments MUST use food IDs from either:
   - The KNOWN FOOD IDS block for add_foods_to_tally
   - The CURRENT TALLY block for remove_foods_from_tally
5. The assistant's final text response should be informative, conversational, and include the key numbers from the calculation results (but never do its own math).
6. Tool results fields should be empty objects {} — a separate system will fill them with correct computed values.
7. For "calculate_final" — include meal_time when the user mentions a meal period, meal_hour when they mention a specific time, and blood_glucose when they mention their BG reading.
8. Make conversations FEEL REAL: include small talk, typos, varying sentence structures, emoji occasionally, and natural corrections ("oh wait, actually...").
9. Vary the conversation length: some conversations complete in 3-4 turns, others have 6-8 turns with back-and-forth.

## %{scenario_type} SPECIFIC INSTRUCTIONS

%{scenario_detail}
```

**Scenario-specific instructions** (injected into `%{scenario_detail}`):

```
SIMPLE_SINGLE_FOOD:
The user mentions ONE food item with a quantity. No blood glucose mentioned.
The assistant should search, add, and present the carbs tally but NOT call
calculate_final (the user didn't ask to calculate yet). The assistant should
say something like "Added! Anything else?" or "What else did you eat?"
Expected turns: user → assistant(tool:search) → tool → assistant(tool:add) → tool → assistant(text)

MULTIPLE_FOODS_NO_GLUCOSE:
The user mentions 2-3 foods with quantities. No blood glucose mentioned.
The assistant should search (one batch search), add all foods, then
calculate_final with appropriate meal_time (inferred or explicit).
Expected turns: user → assistant(search) → tool → assistant(add) → tool → assistant(calculate) → tool → assistant(text)

MULTIPLE_FOODS_WITH_GLUCOSE:
The user mentions 2-4 foods with quantities AND their blood glucose (range 90-250).
The assistant should complete the full flow: search → add → calculate_final with BG.
Present the full breakdown: food insulin + glucose correction = final dose.
Expected turns: user → assistant(search) → tool → assistant(add) → tool → assistant(calculate) → tool → assistant(text)

EXPLICIT_MEAL_TIME:
Vary across: (a) user says "for breakfast" → meal_time="morning", (b) user says "dinner at 8pm" →
meal_time="evening", meal_hour=20, (c) user says "it's 2pm and I ate..." → meal_hour=14,
(d) user says nothing about time → all params omitted/null.
Same tool flow as MULTIPLE_FOODS, but the calculate_final arguments are the focus.

AMBIGUOUS_FOOD:
Use foods that have similar names in the DB or could reasonably have multiple matches
(e.g., "rice" matches both cooked and raw; "milk" matches skimmed, 1%, 2%, whole).
FIRST TURN: Assistant searches, gets multiple results, ASKS the user to clarify.
SECOND TURN: User clarifies, assistant proceeds with the clarified choice.
This creates a 5-7 turn conversation with a clarification gap.

CORRECTION_REMOVAL:
The conversation unfolds in two phases:
PHASE 1 (turns 1-4): Full flow with 2-3 foods + BG, complete calculation.
PHASE 2 (turns 5-7): User says "wait, no [food]" or "actually remove [food]".
Assistant uses remove_foods_from_tally with the entry_id visible in CONTEXT BLOCK,
then recalculates. Present the updated result.
The assistant should NEVER search again to find the entry_id — it reads it from CURRENT TALLY.

GLUCOSE_ONLY_CHECK:
The user mentions ONLY their blood glucose ("my sugar is 150") with NO foods.
The assistant calls calculate_final(blood_glucose=150), but since the tally is empty,
the tool returns an ERROR. The assistant should explain that foods must be added first
before calculating, and suggest the user adds a food or uses the manual calculation tab.
The assistant should NOT invent food IDs or make up foods.

INCOMPLETE_INFO:
The user mentions a food but NOT the quantity ("I ate potatoes" without grams or pieces).
OR mentions a quantity without specifying unit ("2 potatoes" — is that 2 small pieces or 200g?).
OR mentions a food with a unit that doesn't match the DB ("200g of bread" when bread
is defined by slices, not grams).
The assistant should search, then ASK for the missing information before adding.
After the user provides it, complete the flow.

FOOD_NOT_FOUND:
The user mentions a food that DOES NOT EXIST in the database. Use a creative,
non-existent food name (e.g., "dragon fruit chips" or a very specific regional dish).
The assistant searches, gets an empty result (or no match for that specific query),
and politely explains the food wasn't found. Suggest alternatives if similar foods exist.
DO NOT hallucinate food IDs — accept that search returned nothing.
```

---

## Phase 3 — Gemini-Powered Conversation Generator

### `data/generate.py` — POPULATE

Main orchestrator. Architecture identical to the plan's diagram. Key implementation details:

```python
import asyncio
import json
import hashlib
from pathlib import Path
from typing import List, Optional
from google import genai
from google.genai import types

from data.food_db_loader import load_food_db, sample_foods
from data.scenarios import Conversation, ScenarioType, SCENARIO_WEIGHTS
from data.mock_harness import MockHarness
from data.validator import validate_conversation

client = genai.Client()

# --- Gemini API call ---

async def generate_conversation(
    lang: str,
    scenario: ScenarioType,
    foods_sample: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Conversation:

    async with semaphore:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=system_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.8,           # Higher for diversity
                        top_p=0.95,
                        max_output_tokens=4096,
                    ),
                )
                raw = json.loads(response.text)

                # Wrap in Conversation model
                conv_dict = {
                    "scenario_type": scenario.value,
                    "language": lang,
                    "turns": raw["turns"],
                }
                conv = Conversation.model_validate(conv_dict)

                # --- Run through mock harness (deterministic step) ---
                food_db = load_food_db(lang)
                harness = MockHarness(food_db)

                # Build a lookup of food IDs mentioned in Gemini's output
                # so the harness can validate and execute
                for turn in conv.turns:
                    if turn.role == "assistant" and turn.action == "tool_call":
                        results = {}
                        for tc in turn.tool_calls:
                            try:
                                result = harness.execute(tc)
                            except Exception as e:
                                result = {"error": str(e)}
                            results[tc.name] = result

                        # Add a tool turn with the results
                        # (in the raw data, we store results alongside the call for simplicity)
                        turn.tool_results = results

                # --- Validate ---
                validate_conversation(conv, food_db, ScenarioType(scenario))
                return conv

            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

# --- Rate-limited batch generation ---

async def generate_language_dataset(
    lang: str,
    count: int = 800,
    max_concurrent: int = 10,
    resume: bool = True,
):
    output_path = Path(f"data/output/raw/{lang}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing progress
    completed = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                conv = json.loads(line)
                completed.add(
                    (conv["scenario_type"], _hash_user_utterances(conv))
                )

    # Build work items
    food_db = load_food_db(lang)
    semaphore = asyncio.Semaphore(max_concurrent)

    work_items = _generate_work_distribution(count)
    generated = 0

    for scenario, n in work_items:
        for _ in range(n):
            if generated >= count:
                break

            foods = sample_foods(food_db, 15, categories=None)
            system_prompt = _build_system_prompt(lang, scenario, foods)

            conv = await generate_conversation(
                lang, scenario, foods, system_prompt, semaphore
            )

            # Dedup check
            key = (conv.scenario_type, _hash_user_utterances(conv))
            if key in completed:
                continue

            # Save
            with open(output_path, "a") as f:
                f.write(conv.model_dump_json() + "\n")
            completed.add(key)
            generated += 1

    return generated

# --- Helper: deduplication ---

def _hash_user_utterances(conv: Conversation) -> str:
    """Hash all user content for dedup."""
    text = "|".join(
        t.content or "" for t in conv.turns if t.role == "user"
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]

# --- Helper: weighted distribution ---

def _generate_work_distribution(count: int) -> list:
    distribution = []
    remaining = count
    for scenario, weight in list(SCENARIO_WEIGHTS.items())[:-1]:
        n = round(count * weight)
        distribution.append((scenario, n))
        remaining -= n
    distribution.append((list(SCENARIO_WEIGHTS.keys())[-1], remaining))
    return distribution

# --- Main entry point ---

async def main(languages, count_per_lang, dry_run, validate_only):
    if validate_only:
        # Re-validate existing raw data
        ...
    elif dry_run:
        # Show plan, generate 1 per language without API
        ...
    else:
        tasks = [
            generate_language_dataset(lang, count_per_lang)
            for lang in languages
        ]
        results = await asyncio.gather(*tasks)
        for lang, count in zip(languages, results):
            print(f"  {lang}: {count} conversations generated")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="all")
    parser.add_argument("--count-per-lang", type=int, default=800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()

    languages = (
        ["en", "el", "fr", "es", "hi", "it", "pt", "zh", "de", "ja"]
        if args.languages == "all"
        else args.languages.split(",")
    )
    asyncio.run(main(languages, args.count_per_lang, args.dry_run, args.validate_only))
```

---

## Phase 4 — Chat Template Formatters

### `data/prompts/system_lfm.txt` — POPULATE

```
<|im_start|>system
You are a carb counting assistant for people with diabetes, integrated into the Mininio app.
Help users calculate insulin doses by searching the nutrition database, adding foods to a
tally, and computing glucose corrections.

CURRENT USER SETTINGS:
- Glucose: threshold=130.0 mg/dL, baseline=100.0 mg/dL, divisor=40.0 mg/dL per unit
- Meal dividers: Morning=14, Midday=15, Evening=12
- Meal time ranges: Morning (4:00-12:00), Midday (12:00-17:00), Evening (17:00-4:00)

CALCULATION RULES:
- Carbs per food = (quantity × food_carbs) / standard_quantity
- Food insulin = tally_total / meal_divider
- Glucose correction = max(0, (blood_glucose - baseline) / divisor) only if bg >= threshold
- Final dose = food_insulin + glucose_correction

IMPORTANT: Every user message includes a [CURRENT TALLY] and [KNOWN FOOD IDS] block. This
is the AUTHORITATIVE source of truth — trust it over your own memory. You do not need to
call get_tally_summary to discover state unless you've lost track.

Always use the provided functions. If a food name matches multiple results, ask the user to
clarify. If quantity is missing, ask. Call calculate_final with meal_time and blood_glucose
as parameters — no separate setter calls are needed.

List of tools:
1. search_foods(queries: string[]) — Search the nutrition database for food names. Returns matches with IDs and carb values per standard portion.
2. add_foods_to_tally(items: [{food_id: int, quantity: float, unit: str}]) — Add foods to the tally. food_id must come from a prior search_foods result.
3. remove_foods_from_tally(entry_ids: int[]) — Remove entries by their entry IDs (visible in CURRENT TALLY).
4. calculate_final(meal_time?: str, meal_hour?: int, blood_glucose?: float) — Compute final insulin dose. Requires at least one food in tally.
5. get_tally_summary() — Get current calculation state (safety net).
6. clear_all() — Clear all calculation data and start fresh.
<|im_end|>
```

### `data/prompts/system_gemma.txt` — POPULATE

```
<|turn>system
You are a carb counting assistant for people with diabetes, integrated into the Mininio app.
Help users calculate insulin doses by searching the nutrition database, adding foods to a
tally, and computing glucose corrections.

CURRENT USER SETTINGS:
- Glucose: threshold=130.0 mg/dL, baseline=100.0 mg/dL, divisor=40.0 mg/dL per unit
- Meal dividers: Morning=14, Midday=15, Evening=12
- Meal time ranges: Morning (4:00-12:00), Midday (12:00-17:00), Evening (17:00-4:00)

CALCULATION RULES:
- Carbs per food = (quantity × food_carbs) / standard_quantity
- Food insulin = tally_total / meal_divider
- Glucose correction = max(0, (blood_glucose - baseline) / divisor) only if bg >= threshold
- Final dose = food_insulin + glucose_correction

IMPORTANT: Every user message includes a [CURRENT TALLY] and [KNOWN FOOD IDS] block. This
is the AUTHORITATIVE source of truth — trust it over your own memory. You do not need to
call get_tally_summary to discover state unless you've lost track.

Always use the provided functions. If a food name matches multiple results, ask the user to
clarify. If quantity is missing, ask. Call calculate_final with meal_time and blood_glucose
as parameters — no separate setter calls are needed.

Available tools:
1. search_foods(queries: string[])
   Searches the nutrition database for food names. Returns matches with IDs, carb values
   per standard portion (per 100g or per piece), and supported units (grams/pieces).
   Example: search_foods(queries=["potatoes", "bread"])

2. add_foods_to_tally(items: [{food_id, quantity, unit}])
   Adds food items to the tally. food_id must come from a prior search_foods result.
   Carbs are computed as (quantity × food_carbs) / standard_quantity.
   Returns entries with their entry_id and the new tally_total.
   Example: add_foods_to_tally(items=[{food_id: 12, quantity: 100, unit: "g"}])

3. remove_foods_from_tally(entry_ids: int[])
   Removes entries by their entry IDs (visible in CURRENT TALLY block).
   Example: remove_foods_from_tally(entry_ids=[1, 3])

4. calculate_final(meal_time?: str, meal_hour?: int, blood_glucose?: float)
   Computes final insulin dose. meal_time is "morning", "midday", or "evening".
   blood_glucose in mg/dL. Requires at least one food in the tally.
   Returns final_result, food_insulin, glucose_correction, and breakdown.
   Example: calculate_final(meal_time="midday", blood_glucose=140)

5. get_tally_summary()
   Returns current tally state. Safety net — normally state is in the CURRENT TALLY block.

6. clear_all()
   Clears all calculation data and known food IDs. Fresh start.<turn|>
```

### `data/formatters/lfm_formatter.py` — NEW

```python
"""
Converts model-agnostic Conversation → LFM2.5 ChatML training string.

Format reference: lfm2_5_sft_with_unsloth.py (lines 103-109, 146-150)

Key features:
- Role markers: <|im_start|>system/user/assistant/tool + <|im_end|>
- Tool calls: wrapped in <|tool_call_start|> / <|tool_call_end|>
- BOS token: <|startoftext|> included, removed by formatting_prompts_func
- train_on_responses_only: instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"
"""

from typing import List
from data.scenarios import Conversation, Turn

def format_conversation(conv: Conversation, system_prompt: str) -> str:
    parts = ["<|startoftext|>"]
    parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

    for turn in conv.turns:
        if turn.role == "user":
            user_text = turn.content or ""
            if turn.context_block:
                user_text += f"\n\n{turn.context_block}"
            parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")

        elif turn.role == "assistant":
            if turn.action == "tool_call":
                if turn.tool_calls:
                    # Single or multiple tool calls
                    call_texts = []
                    for tc in turn.tool_calls:
                        call_texts.append(_serialize_lfm_tool_call(tc))
                    calls_text = ", ".join(call_texts)
                    parts.append(
                        f"<|im_start|>assistant\n"
                        f"<|tool_call_start|>{calls_text}<|tool_call_end|>"
                        f"<|im_end|>"
                    )
            elif turn.action == "text":
                parts.append(f"<|im_start|>assistant\n{turn.content}<|im_end|>")

        elif turn.role == "tool" and turn.tool_results:
            # Tool results as JSON
            results_json = json.dumps(turn.tool_results, ensure_ascii=False)
            parts.append(f"<|im_start|>tool\n{results_json}<|im_end|>")

    return "".join(parts)


def _serialize_lfm_tool_call(tc) -> str:
    """Convert ToolCall to LFM2.5 Pythonic function call string."""
    args_strs = []
    for key, value in tc.arguments.items():
        if isinstance(value, str):
            args_strs.append(f'{key}="{value}"')
        elif isinstance(value, list):
            if value and isinstance(value[0], str):
                items = ", ".join(f'"{v}"' for v in value)
                args_strs.append(f"{key}=[{items}]")
            else:
                # List of dicts (e.g., add_foods_to_tally items)
                items = ", ".join(json.dumps(v, ensure_ascii=False) for v in value)
                args_strs.append(f"{key}=[{items}]")
        else:
            args_strs.append(f"{key}={value}")
    return f"{tc.name}({', '.join(args_strs)})"
```

### `data/formatters/gemma_formatter.py` — NEW (Corrected)

```python
"""
Converts model-agnostic Conversation → Gemma 4 Unsloth training string.

Format reference: gemma_4_finetuning_quide (lines 358-362, 390, 430-433)

Key features:
- Role markers: <|turn>system/user/model + <turn|>
- Function calls: embedded as text within <|turn>model blocks
- Tool results: appended as text in separate <|turn>model blocks
- BOS token: <bos> included, removed by formatting_prompts_func via .removeprefix("<bos>")
- train_on_responses_only: instruction_part="<|turn>user\n", response_part="<|turn>model\n"

IMPORTANT: This uses the Unsloth/HuggingFace chat template, NOT the Android LiteRT-LM
runtime format (<start_function_call>, <escape>, etc.). That conversion happens at export time.
"""

from typing import List
from data.scenarios import Conversation, Turn

def format_conversation(conv: Conversation, system_prompt: str) -> str:
    parts = ["<bos>"]
    parts.append(f"<|turn>system\n{system_prompt}<turn|>")

    for turn in conv.turns:
        if turn.role == "user":
            user_text = turn.content or ""
            if turn.context_block:
                user_text += f"\n\n{turn.context_block}"
            parts.append(f"<|turn>user\n{user_text}<turn|>")

        elif turn.role == "assistant":
            if turn.action == "tool_call" and turn.tool_calls:
                tool_call_text = _serialize_gemma_tool_calls(turn.tool_calls)
                parts.append(f"<|turn>model\n{tool_call_text}<turn|>")
            elif turn.action == "text":
                parts.append(f"<|turn>model\n{turn.content}<turn|>")

        elif turn.role == "tool" and turn.tool_results:
            # Tool results go as text in a separate model turn
            results_text = _serialize_gemma_tool_results(turn.tool_results)
            parts.append(f"<|turn>model\n{results_text}<turn|>")

    return "".join(parts)


def _serialize_gemma_tool_calls(tool_calls: list) -> str:
    """Serialize tool calls as structured text the model learns to generate."""
    lines = []
    for tc in tool_calls:
        args_json = json.dumps(tc.arguments, ensure_ascii=False)
        lines.append(f"{tc.name}({args_json})")
    return "\n".join(lines)


def _serialize_gemma_tool_results(results: dict) -> str:
    """Serialize tool results as structured text."""
    lines = ["Tool results:"]
    for tool_name, result in results.items():
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        lines.append(f"{tool_name} → {result_json}")
    return "\n".join(lines)
```

### `data/formatters/__init__.py` — NEW

```python
from .lfm_formatter import format_conversation as format_lfm
from .gemma_formatter import format_conversation as format_gemma
```

---

## Phase 5 — Validation & Quality Assurance

### `data/validator.py` — NEW

Multi-layer validation run on every generated conversation.

```python
from data.scenarios import Conversation, ScenarioType
from data.food_db_loader import FoodItem
from data.mock_harness import MockHarness
from typing import List

def validate_conversation(
    conv: Conversation,
    food_db: List[FoodItem],
    expected_scenario: ScenarioType,
) -> List[str]:
    """
    Validate a conversation. Returns list of error messages (empty = valid).
    Checks: schema, tool sequence, food ID integrity, entry ID integrity,
    math correctness, language, length, no empty turns.
    """
    errors = []

    # 1. Schema: Pydantic model already validated

    # 2. Tool sequence: Must follow dependency graph
    errors += _validate_tool_sequence(conv, expected_scenario)

    # 3. Food ID integrity: Every food_id in add_foods_to_tally must be in
    #    KNOWN FOOD IDS from a prior search_foods result
    errors += _validate_food_id_integrity(conv, food_db)

    # 4. Entry ID integrity: Every entry_id in remove_foods_from_tally
    #    must exist in the tally at that point
    errors += _validate_entry_id_integrity(conv)

    # 5. Math correctness: Re-run all tool calls through mock harness
    errors += _validate_math(conv, food_db)

    # 6. Language check: User utterances in expected language
    errors += _validate_language(conv)

    # 7. Length bounds: Estimated tokens 256-2048
    errors += _validate_length(conv)

    # 8. No empty turns
    errors += _validate_no_empty_turns(conv)

    # 9. Context block integrity: All user turns have context_block
    errors += _validate_context_blocks(conv)

    return errors


def _validate_tool_sequence(conv, scenario) -> list:
    """Verify tool calls follow dependency graph."""
    tools_seen = []
    for turn in conv.turns:
        if turn.role == "assistant" and turn.action == "tool_call":
            for tc in (turn.tool_calls or []):
                tools_seen.append(tc.name)

                # calculate_final requires prior add_foods_to_tally
                # (unless GLUCOSE_ONLY_CHECK where it's intentionally called first)
                if tc.name == "calculate_final" and scenario != ScenarioType.GLUCOSE_ONLY_CHECK:
                    if "add_foods_to_tally" not in tools_seen[:-1]:
                        return ["calculate_final called before add_foods_to_tally"]

                # add_foods_to_tally requires prior search_foods
                if tc.name == "add_foods_to_tally":
                    if "search_foods" not in tools_seen[:-1]:
                        return ["add_foods_to_tally called before search_foods"]

                # remove_foods_from_tally requires prior add_foods_to_tally
                if tc.name == "remove_foods_from_tally":
                    if "add_foods_to_tally" not in tools_seen[:-1]:
                        return ["remove_foods_to_tally called before add_foods_to_tally"]
    return []


def _validate_food_id_integrity(conv, food_db) -> list:
    """Every food_id in add_foods_to_tally must be in KNOWN FOOD IDS from a prior search."""
    errors = []
    known_ids = set()
    for turn in conv.turns:
        # Track known IDs from context blocks
        if turn.role == "user" and turn.context_block:
            known_ids.update(_parse_known_ids_from_context(turn.context_block))
        # Track known IDs from tool results
        if turn.role == "tool" and turn.tool_results:
            if "search_foods" in turn.tool_results:
                results = turn.tool_results["search_foods"]
                if isinstance(results, dict) and "results" in results:
                    for batch in results["results"]:
                        for item in batch:
                            known_ids.add(item["id"])
        # Validate
        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                if tc.name == "add_foods_to_tally":
                    for item in tc.arguments.get("items", []):
                        if item["food_id"] not in known_ids:
                            errors.append(
                                f"food_id {item['food_id']} not in known IDs: {known_ids}"
                            )
    return errors


def _validate_entry_id_integrity(conv) -> list:
    """Every entry_id in remove_foods_from_tally must exist in tally at that point."""
    errors = []
    tally_entry_ids = set()
    entry_counter = 1

    for turn in conv.turns:
        if turn.role == "tool" and turn.tool_results:
            for name, result in turn.tool_results.items():
                if name == "add_foods_to_tally":
                    for e in result.get("entries", []):
                        tally_entry_ids.add(e.get("entry_id"))
                elif name == "remove_foods_from_tally":
                    removed_ids = set()  # Track from tool results
                    # The removed count tells us they were valid
                    tally_entry_ids.clear()  # Simplify: recalculate from status

        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                if tc.name == "remove_foods_from_tally":
                    for eid in tc.arguments.get("entry_ids", []):
                        if eid not in tally_entry_ids:
                            errors.append(
                                f"entry_id {eid} not in current tally: {tally_entry_ids}"
                            )
    return errors


def _validate_math(conv, food_db) -> list:
    """Re-run all tool calls through mock harness and compare results."""
    errors = []
    harness = MockHarness(food_db)

    for turn in conv.turns:
        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                expected = harness.execute(tc)
                if turn.tool_results and tc.name in (turn.tool_results or {}):
                    actual = turn.tool_results[tc.name]
                    # Compare float values within 0.01 tolerance
                    for key in expected:
                        if isinstance(expected[key], (int, float)) and key in actual:
                            if isinstance(actual[key], (int, float)):
                                if abs(expected[key] - actual[key]) > 0.01:
                                    errors.append(
                                        f"Math mismatch for {tc.name}.{key}: "
                                        f"expected {expected[key]}, got {actual[key]}"
                                    )
    return errors


def _validate_language(conv) -> list:
    """Basic language heuristic: check character set ranges."""
    # Define character ranges for each language
    RANGES = {
        "en": "ascii", "fr": "ascii+accents", "es": "ascii+accents",
        "de": "ascii+umlauts", "it": "ascii+accents", "pt": "ascii+accents",
        "el": "greek", "hi": "devanagari", "ja": "cjk+hiragana+katakana",
        "zh": "cjk",
    }
    # Simplified: just check that user utterances don't contain
    # wildly wrong character sets
    # (Full implementation would check percentage of chars in expected range)
    return []


def _validate_length(conv) -> list:
    """Estimate token count (4 chars ~= 1 token heuristic)."""
    full_text = " ".join(
        t.content or "" for t in conv.turns
    )
    estimated_tokens = len(full_text) // 4
    if estimated_tokens < 128:
        return [f"Too short: ~{estimated_tokens} tokens"]
    if estimated_tokens > 2560:
        return [f"Too long: ~{estimated_tokens} tokens"]
    return []


def _validate_no_empty_turns(conv) -> list:
    """Every turn must have non-empty content or tool calls."""
    for i, turn in enumerate(conv.turns):
        if turn.role == "user" and not turn.content:
            return [f"Turn {i}: empty user content"]
        if turn.role == "assistant":
            if turn.action == "text" and not turn.content:
                return [f"Turn {i}: empty assistant text"]
            if turn.action == "tool_call" and not turn.tool_calls:
                return [f"Turn {i}: empty tool calls"]
    return []


def _validate_context_blocks(conv) -> list:
    """All user turns must have a context_block."""
    for i, turn in enumerate(conv.turns):
        if turn.role == "user" and not turn.context_block:
            return [f"Turn {i}: missing context_block in user message"]
    return []


def _parse_known_ids_from_context(block: str) -> set:
    """Extract food IDs from [KNOWN FOOD IDS: ...] block."""
    ids = set()
    import re
    matches = re.findall(r'\((\d+)\)', block)
    for m in matches:
        ids.add(int(m))
    return ids
```

### `data/stats.py` — NEW

Generates a summary report in `data/output/stats_report.md`:
- Per-language counts by scenario type
- Total token count distribution (histogram)
- Average turns per conversation
- Retry rate per scenario type
- Language distribution balance check
- Scenario type distribution vs targets

---

## Phase 6 — Dataset Assembly & Export

### `data/assemble.py` — NEW

```python
"""
Reads validated raw JSONL → formats for both models → splits 90/10 train/eval.

Output structure:
data/output/
├── raw/{lang}.jsonl
├── lfm/train.jsonl, lfm/eval.jsonl
├── gemma/train.jsonl, gemma/eval.jsonl
└── stats_report.md
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from data.scenarios import Conversation
from data.formatters import format_lfm, format_gemma

SEED = 42
TRAIN_RATIO = 0.9

def assemble_dataset(raw_dir: str = "data/output/raw"):
    raw_path = Path(raw_dir)
    raw_conversations = []

    # 1. Load all raw conversations
    for lang_file in sorted(raw_path.glob("*.jsonl")):
        with open(lang_file) as f:
            for line in f:
                conv = Conversation.model_validate_json(line)
                raw_conversations.append(conv)

    print(f"Loaded {len(raw_conversations)} conversations")

    # 2. Load system prompts
    lfm_system = Path("data/prompts/system_lfm.txt").read_text()
    gemma_system = Path("data/prompts/system_gemma.txt").read_text()

    # 3. Stratified split by (language × scenario_type)
    random.seed(SEED)
    groups = defaultdict(list)
    for conv in raw_conversations:
        key = (conv.language, conv.scenario_type)
        groups[key].append(conv)

    train_convs = []
    eval_convs = []
    for key, convs in groups.items():
        random.shuffle(convs)
        split_idx = int(len(convs) * TRAIN_RATIO)
        train_convs.extend(convs[:split_idx])
        eval_convs.extend(convs[split_idx:])

    print(f"Train: {len(train_convs)}, Eval: {len(eval_convs)}")

    # 4. Format and save for LFM
    _save_formatted(train_convs, eval_convs, lfm_system,
                    "data/output/lfm", format_lfm)

    # 5. Format and save for Gemma
    _save_formatted(train_convs, eval_convs, gemma_system,
                    "data/output/gemma", format_gemma)

    print("Assembly complete.")


def _save_formatted(train, eval, system_prompt, output_dir, formatter):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, convs in [("train", train), ("eval", eval)]:
        with open(out / f"{split_name}.jsonl", "w") as f:
            for conv in convs:
                text = formatter(conv, system_prompt)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    assemble_dataset()
```

**Guarantee**: The same 8,000 semantic conversations appear in both `lfm/` and `gemma/` directories. Only the template wrapping differs.

---

## Phase 7 — Integration with Fine-Tuning Scripts

### `finetuning/common/data_loader.py` — POPULATE

```python
from typing import Literal
from datasets import Dataset, DatasetDict

def load_dataset_for_model(
    model_type: Literal["lfm", "gemma"],
) -> DatasetDict:
    """Load the pre-formatted JSONL dataset for either model."""
    train = Dataset.from_json(f"data/output/{model_type}/train.jsonl")
    eval_ds = Dataset.from_json(f"data/output/{model_type}/eval.jsonl")
    return DatasetDict({"train": train, "eval": eval_ds})
```

### `requirements.txt` — ADD pydantic

Add `pydantic>=2.0` (currently missing but required by `scenarios.py` and `validator.py`).

---

## Cost Analysis

### Gemini 2.5 Flash Pricing

| Component | Calculation | Tokens | Rate | Cost |
|-----------|-------------|--------|------|------|
| Output tokens | 8,500 calls × ~2,000 tokens | 17,000,000 | $0.60/1M | **$10.20** |
| Input tokens | 8,500 calls × ~500 tokens | 4,250,000 | $0.15/1M | $0.64 |
| **Total** | | | | **~$10.84** |

### Buffer for Retries

Add 15% for retries (failed validation, schema errors, rate limit retries): **~$12.50 total**.

### Gemini 2.5 Pro (if Flash unavailable)

| Component | Rate | Cost |
|-----------|------|------|
| Output (17M) | $10.00/1M | **$170.00** |
| Input (4.25M) | $1.25/1M | $5.31 |
| **Total** | | **~$175.31** |

**Recommendation**: Use Flash for all generation. The quality difference for synthetic conversation generation is negligible — the mock harness guarantees mathematical correctness regardless of which model generates the semantic content.

---

## Verification Plan

### Automated Checks

```bash
# 1. Unit test mock harness (math correctness)
python -m pytest data/tests/test_mock_harness.py -v

# 2. Unit test formatters (template token correctness)
python -m pytest data/tests/test_formatters.py -v

# 3. Dry-run (no API calls, 1 conversation per language)
python data/generate.py --dry-run --count-per-lang 1

# 4. Small-scale test (10 per language = 100 total)
python data/generate.py --count-per-lang 10 --languages en,el

# 5. Validate generated data
python data/generate.py --validate-only

# 6. Full generation
python data/generate.py --count-per-lang 800 --languages all

# 7. Assemble final datasets
python data/assemble.py

# 8. Verify train_on_responses_only masking
# Run tokenization pass with each model's tokenizer
python -m pytest data/tests/test_training_masking.py -v
```

### Manual Verification

1. **Spot-check 5 conversations per language** — verify natural language quality, correct tool sequences, accurate math in tool results
2. **Tokenize a sample with each model's tokenizer** — verify token counts are within `max_seq_length=2048`
3. **Verify `train_on_responses_only` masking** — confirm only assistant/model turns are unmasked
4. **Run stats report** — confirm scenario distribution matches targets within ±2%
5. **Verify cross-model equivalence** — confirm lfm/train.jsonl and gemma/train.jsonl have the same number of lines and represent the same conversations

### Key Invariants

- Every `food_id` in `add_foods_to_tally` exists in a prior `search_foods` return
- Every `entry_id` in `remove_foods_from_tally` exists in the current tally
- `calculate_final` is never called with empty tally (except `GLUCOSE_ONLY_CHECK` where it returns error)
- `final_result = food_insulin + glucose_correction` within 0.01 tolerance
- No conversation exceeds 2,048 tokens when tokenized
- Each language has ~800 conversations (±5%)
- Both model formats contain identical semantic content (same conversations, different templates)

---

## Implementation Order

1. **`data/schemas/tools.json`** — POPULATE (blocking dependency for everything else)
2. **`data/food_db_loader.py`** — NEW (needed by mock harness and generator)
3. **`data/mock_harness.py`** — NEW (needed for tool execution and math validation)
4. **`data/scenarios.py`** — NEW (Pydantic models + scenario definitions)
5. **`data/prompts/system_generator.txt`** — POPULATE (Gemini prompt)
6. **`data/generate.py`** — POPULATE (orchestrator)
7. **`data/validator.py`** — NEW (validation layer)
8. **`data/prompts/system_lfm.txt`** — POPULATE (LFM system prompt)
9. **`data/prompts/system_gemma.txt`** — POPULATE (Gemma system prompt)
10. **`data/formatters/__init__.py`** + **`lfm_formatter.py`** + **`gemma_formatter.py`** — NEW
11. **`data/stats.py`** — NEW
12. **`data/assemble.py`** — NEW (final assembly)
13. **`finetuning/common/data_loader.py`** — POPULATE
14. **`requirements.txt`** — ADD `pydantic>=2.0`

---

## Open Decisions

| # | Question | Options |
|---|----------|---------|
| 1 | Gemini Flash `response_schema` support? | Try Pydantic schema; fall back to `response_mime_type="application/json"` + manual validation |
| 2 | Vary user settings across conversations? | **Recommended**: Vary ±20% of defaults so model reads settings from context, not memorizes defaults |
| 3 | Include adversarial/edge-case examples? | **Recommended**: 3-5% of convos with typos, mixed languages, contradictory info |
| 4 | Parallel or serial generation? | **Recommended**: Parallel with asyncio.Semaphore(10) + exponential backoff |
| 5 | Verify LFM2.5 tool call tokens? | Run one test conversation through LFM2.5 tokenizer to confirm `<|tool_call_start|>` etc. are single tokens |
| 6 | Gemma tool result format sufficient? | Verify model learns to parse tool results as text within `<|turn>model` blocks. Plan B: if model confuses tool calls and results in same role, add a text delimiter prefix |

---

## File Summary

| File | Action | Lines (est.) | Purpose |
|------|--------|-------------|---------|
| `data/schemas/tools.json` | POPULATE | ~250 | 6 tool schemas with JSON Schema definitions |
| `data/food_db_loader.py` | NEW | ~120 | Load & query food CSVs, derive carb values |
| `data/scenarios.py` | NEW | ~180 | Pydantic models, scenario types, sequence defs |
| `data/mock_harness.py` | NEW | ~200 | Deterministic tool execution engine |
| `data/prompts/system_generator.txt` | POPULATE | ~150 | Gemini mega-prompt with scenario instructions |
| `data/prompts/system_lfm.txt` | POPULATE | ~40 | LFM2.5 ChatML system prompt |
| `data/prompts/system_gemma.txt` | POPULATE | ~50 | Gemma 4 Unsloth system prompt |
| `data/generate.py` | POPULATE | ~300 | Main orchestrator with async API calls, retries |
| `data/formatters/lfm_formatter.py` | NEW | ~80 | LFM2.5 ChatML converter |
| `data/formatters/gemma_formatter.py` | NEW | ~80 | Gemma 4 Unsloth converter (corrected) |
| `data/formatters/__init__.py` | NEW | ~5 | Package exports |
| `data/validator.py` | NEW | ~250 | Multi-layer validation (8 checks) |
| `data/stats.py` | NEW | ~100 | Generation statistics report |
| `data/assemble.py` | NEW | ~100 | Final assembly + stratified train/eval split |
| `finetuning/common/data_loader.py` | POPULATE | ~30 | HuggingFace Dataset loader |
| `requirements.txt` | ADD pydantic | +1 | Missing dependency |
| **Total** | | **~1,900** | |
