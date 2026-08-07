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
    CORRECTION_QUANTITY = "CORRECTION_QUANTITY"
    GLUCOSE_QUANTITY_CORRECTION = "GLUCOSE_QUANTITY_CORRECTION"
    GLUCOSE_ONLY_CHECK = "GLUCOSE_ONLY_CHECK"
    INCOMPLETE_INFO = "INCOMPLETE_INFO"
    FOOD_NOT_FOUND = "FOOD_NOT_FOUND"
    TALLY_SUMMARY = "TALLY_SUMMARY"


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
    ScenarioType.SIMPLE_SINGLE_FOOD: 0.11,
    ScenarioType.MULTIPLE_FOODS_NO_GLUCOSE: 0.20,
    ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE: 0.25,
    ScenarioType.EXPLICIT_MEAL_TIME: 0.10,
    ScenarioType.AMBIGUOUS_FOOD: 0.08,
    ScenarioType.CORRECTION_REMOVAL: 0.07,
    ScenarioType.CORRECTION_QUANTITY: 0.05,
    ScenarioType.GLUCOSE_QUANTITY_CORRECTION: 0.02,
    ScenarioType.GLUCOSE_ONLY_CHECK: 0.04,
    ScenarioType.INCOMPLETE_INFO: 0.04,
    ScenarioType.FOOD_NOT_FOUND: 0.01,
    ScenarioType.TALLY_SUMMARY: 0.03,
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
    ScenarioType.CORRECTION_QUANTITY: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"],
        ["remove_foods_from_tally", "add_foods_to_tally", "calculate_final", "text"],
    ],
    ScenarioType.TALLY_SUMMARY: [
        ["search_foods", "add_foods_to_tally", "text"],
        ["get_tally_summary", "text"],
    ],
    ScenarioType.GLUCOSE_QUANTITY_CORRECTION: [
        ["search_foods", "add_foods_to_tally", "calculate_final", "text"],
        ["calculate_final", "text"],
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
    ScenarioType.CORRECTION_QUANTITY: (
        "Two-phase conversation: "
        "PHASE 1 (turns 1-4): Full flow with 1-2 foods + optional BG, complete calculation. "
        "PHASE 2 (turns 5-8): User corrects a quantity: \"correction, it was 100g, not 1000g\" "
        "or \"actually I ate 200g of [food]\" or \"sorry, the potatoes were only 150g\". "
        "The assistant MUST: (1) read entry_id from CURRENT TALLY, (2) call remove_foods_from_tally "
        "with that entry_id, (3) call add_foods_to_tally with the corrected quantity using food_id "
        "from KNOWN FOOD IDS (do NOT search again), (4) call calculate_final to get updated dose. "
        "Present the corrected result showing the new total and dose."
    ),
    ScenarioType.TALLY_SUMMARY: (
        "Two-phase conversation or standalone: "
        "PHASE 1 (turns 1-3): User mentions 1-2 foods, assistant searches and adds, "
        "then asks \"Anything else?\" (no calculate_final unless requested). "
        "PHASE 2 (turns 4-5): User asks \"what's in my tally?\" or \"show me what I've added\" "
        "or \"what foods are in my tally?\". Assistant calls get_tally_summary(), then reads "
        "the entries from the result and reports them back in natural language WITHOUT "
        "mentioning entry_id numbers: "
        "\"Your tally has: Grapes 200g = 35.30g carbs, Potatoes 150g = 26.47g carbs. "
        "Total: 61.77g carbs.\" Do NOT expose internal entry IDs to the user."
    ),
    ScenarioType.GLUCOSE_QUANTITY_CORRECTION: (
        "Two-phase conversation: "
        "PHASE 1 (turns 1-4): Full flow with 1-2 foods + blood glucose, complete calculation "
        "with a specific BG value (e.g., 150). Assistant presents full breakdown. "
        "PHASE 2 (turns 5-6): User corrects the glucose reading: \"sorry, my sugar was 180, not 150\" "
        "or \"actually my glucose was 200\". The assistant simply calls calculate_final again with "
        "the corrected blood_glucose value — NO tally changes needed (no remove/re-add). "
        "Present the updated dose showing how the glucose correction changed."
    ),
}
