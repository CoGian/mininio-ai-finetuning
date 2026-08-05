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
    content: Optional[str] = None
    action: Optional[Literal["text", "tool_call"]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[dict] = None
    context_block: Optional[str] = None


class Conversation(BaseModel):
    scenario_type: str
    language: str
    turns: List[Turn]
    user_settings_idx: Optional[int] = None


SCENARIO_WEIGHTS = {
    ScenarioType.SIMPLE_SINGLE_FOOD: 0.15,
    ScenarioType.MULTIPLE_FOODS_NO_GLUCOSE: 0.20,
    ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE: 0.25,
    ScenarioType.EXPLICIT_MEAL_TIME: 0.10,
    ScenarioType.AMBIGUOUS_FOOD: 0.10,
    ScenarioType.CORRECTION_REMOVAL: 0.08,
    ScenarioType.GLUCOSE_ONLY_CHECK: 0.05,
    ScenarioType.INCOMPLETE_INFO: 0.05,
    ScenarioType.FOOD_NOT_FOUND: 0.02,
}

EXPECTED_SEQUENCES = {
    ScenarioType.SIMPLE_SINGLE_FOOD: [
        ["search_foods", "add_foods_to_tally", "text"],
    ],
    ScenarioType.MULTIPLE_FOODS_NO_GLUCOSE: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"],
    ],
    ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"],
    ],
    ScenarioType.EXPLICIT_MEAL_TIME: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"],
    ],
    ScenarioType.AMBIGUOUS_FOOD: [
        ["search_foods", "text"],
        ["add_foods_to_tally", "calculate_final", "text"],
    ],
    ScenarioType.CORRECTION_REMOVAL: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"],
        ["remove_foods_from_tally", "calculate_final", "text"],
    ],
    ScenarioType.GLUCOSE_ONLY_CHECK: [
        ["calculate_final", "text"],
    ],
    ScenarioType.INCOMPLETE_INFO: [
        ["search_foods", "text"],
        ["add_foods_to_tally", "calculate_final", "text"],
    ],
    ScenarioType.FOOD_NOT_FOUND: [
        ["search_foods", "text"],
    ],
}

LANGUAGE_NAMES = {
    "en": "English",
    "el": "Greek",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese (Simplified)",
    "de": "German",
    "ja": "Japanese",
}

SCENARIO_INSTRUCTIONS = {
    ScenarioType.SIMPLE_SINGLE_FOOD: (
        "The user mentions ONE food item with a quantity. No blood glucose mentioned. "
        "The assistant should search, add, and present the carbs tally but NOT call "
        "calculate_final (the user didn't ask to calculate yet). The assistant should "
        "say something like \"Added! Anything else?\" or \"What else did you eat?\" "
        "Expected turns: user -> assistant(tool:search) -> tool -> assistant(tool:add) -> tool -> assistant(text)"
    ),
    ScenarioType.MULTIPLE_FOODS_NO_GLUCOSE: (
        "The user mentions 2-3 foods with quantities. No blood glucose mentioned. "
        "The assistant should search (one batch search), add all foods, then "
        "calculate_final with appropriate meal_time (inferred or explicit). "
        "Expected turns: user -> assistant(search) -> tool -> assistant(add) -> tool -> assistant(calculate) -> tool -> assistant(text)"
    ),
    ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE: (
        "The user mentions 2-4 foods with quantities AND their blood glucose (range 90-250). "
        "The assistant should complete the full flow: search -> add -> calculate_final with BG. "
        "Present the full breakdown: food insulin + glucose correction = final dose. "
        "Expected turns: user -> assistant(search) -> tool -> assistant(add) -> tool -> assistant(calculate) -> tool -> assistant(text)"
    ),
    ScenarioType.EXPLICIT_MEAL_TIME: (
        "Vary across: (a) user says \"for breakfast\" -> meal_time=\"morning\", (b) user says \"dinner at 8pm\" -> "
        "meal_time=\"evening\", meal_hour=20, (c) user says \"it's 2pm and I ate...\" -> meal_hour=14, "
        "(d) user says nothing about time -> all params omitted/null. "
        "Same tool flow as MULTIPLE_FOODS, but the calculate_final arguments are the focus."
    ),
    ScenarioType.AMBIGUOUS_FOOD: (
        "Use foods that have similar names in the DB or could reasonably have multiple matches "
        "(e.g., \"rice\" matches both cooked and raw; \"milk\" matches skimmed, 1%, 2%, whole). "
        "FIRST TURN: Assistant searches, gets multiple results, ASKS the user to clarify. "
        "SECOND TURN: User clarifies, assistant proceeds with the clarified choice. "
        "This creates a 5-7 turn conversation with a clarification gap."
    ),
    ScenarioType.CORRECTION_REMOVAL: (
        "The conversation unfolds in two phases: "
        "PHASE 1 (turns 1-4): Full flow with 2-3 foods + BG, complete calculation. "
        "PHASE 2 (turns 5-7): User says \"wait, no [food]\" or \"actually remove [food]\". "
        "Assistant uses remove_foods_from_tally with the entry_id visible in CONTEXT BLOCK, "
        "then recalculates. Present the updated result. "
        "The assistant should NEVER search again to find the entry_id - it reads it from CURRENT TALLY."
    ),
    ScenarioType.GLUCOSE_ONLY_CHECK: (
        "The user mentions ONLY their blood glucose (\"my sugar is 150\") with NO foods. "
        "The assistant calls calculate_final(blood_glucose=150), but since the tally is empty, "
        "the tool returns an ERROR. The assistant should explain that foods must be added first "
        "before calculating, and suggest the user adds a food or uses the manual calculation tab. "
        "The assistant should NOT invent food IDs or make up foods."
    ),
    ScenarioType.INCOMPLETE_INFO: (
        "The user mentions a food but NOT the quantity (\"I ate potatoes\" without grams or pieces). "
        "OR mentions a quantity without specifying unit (\"2 potatoes\" - is that 2 small pieces or 200g?). "
        "OR mentions a food with a unit that doesn't match the DB (\"200g of bread\" when bread "
        "is defined by slices, not grams). "
        "The assistant should search, then ASK for the missing information before adding. "
        "After the user provides it, complete the flow."
    ),
    ScenarioType.FOOD_NOT_FOUND: (
        "The user mentions a food that DOES NOT EXIST in the database. Use a creative, "
        "non-existent food name (e.g., \"dragon fruit chips\" or a very specific regional dish). "
        "The assistant searches, gets an empty result (or no match for that specific query), "
        "and politely explains the food wasn't found. Suggest alternatives if similar foods exist. "
        "DO NOT hallucinate food IDs - accept that search returned nothing."
    ),
}
