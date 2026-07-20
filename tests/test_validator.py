import pytest

from data.scenarios import Conversation, ScenarioType, Turn, ToolCall
from data.validator import (
    validate_conversation,
    _validate_tool_sequence,
    _validate_food_id_integrity,
    _validate_math,
    _validate_length,
    _validate_no_empty_turns,
    _validate_context_blocks,
    _validate_units,
    _validate_text_math,
)


class TestValidateToolSequence:
    def test_correct_sequence(self, valid_simple_conversation: Conversation):
        errors = _validate_tool_sequence(
            valid_simple_conversation, ScenarioType.SIMPLE_SINGLE_FOOD
        )
        assert errors == []

    def test_add_before_search(self, invalid_wrong_order: Conversation):
        errors = _validate_tool_sequence(
            invalid_wrong_order, ScenarioType.SIMPLE_SINGLE_FOOD
        )
        assert len(errors) >= 1
        assert any("before search" in e.lower() for e in errors)

    def test_correction_multi_phase(
        self, sample_gram_food, sample_piece_food
    ):
        from data.mock_harness import _make_search_result
        search_result = _make_search_result(sample_gram_food)
        conv = Conversation(
            scenario_type=ScenarioType.CORRECTION_REMOVAL.value,
            language="en",
            turns=[
                Turn(
                    role="user",
                    content="I ate 100g potatoes and 2 slices bread, BG 180. Pad text for length. "
                            "Pad text for length. Pad text for length. Pad text for length. "
                            "Pad text for length. Pad text for length.",
                    context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
                ),
                Turn(role="assistant", action="tool_call",
                     tool_calls=[ToolCall(name="search_foods", arguments={"queries": ["potatoes", "bread"]})],
                     tool_results={"search_foods": {"results": [[search_result], []]}}),
                Turn(role="tool",
                     tool_results={"search_foods": {"results": [[search_result], []]}}),
                Turn(role="assistant", action="tool_call",
                     tool_calls=[ToolCall(name="add_foods_to_tally", arguments={
                         "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
                     })],
                     tool_results={"add_foods_to_tally": {"entries": [
                         {"entry_id": 1, "food_name": "Potatoes", "quantity": 100, "unit": "g", "carbs": 20.0}
                     ], "tally_total": 20.0}}),
                Turn(role="tool",
                     tool_results={"add_foods_to_tally": {"entries": [
                         {"entry_id": 1, "food_name": "Potatoes", "quantity": 100, "unit": "g", "carbs": 20.0}
                     ], "tally_total": 20.0}}),
                Turn(role="user",
                     content="Wait, remove the potatoes please. Pad text for length. "
                             "Pad text for length. Pad text for length.",
                     context_block="[CURRENT TALLY: 1 items, 20.0g total]\n"
                                   "  Potatoes 100g = 20.0g (entry_id: 1)\n"
                                   "[KNOWN FOOD IDS: Potatoes(1)]"),
                Turn(role="assistant", action="tool_call",
                     tool_calls=[ToolCall(name="remove_foods_from_tally", arguments={"entry_ids": [1]})],
                     tool_results={"remove_foods_from_tally": {"removed": 1, "tally_total": 0.0}}),
                Turn(role="tool",
                     tool_results={"remove_foods_from_tally": {"removed": 1, "tally_total": 0.0}}),
                Turn(role="assistant", action="text",
                     content="Removed. Pad text for length. Pad text for length. "
                             "Pad text for length."),
            ],
        )
        errors = _validate_tool_sequence(
            conv, ScenarioType.CORRECTION_REMOVAL
        )
        assert errors == []


class TestValidateFoodIdIntegrity:
    def test_known_from_context(self, valid_simple_conversation: Conversation, sample_food_db):
        errors = _validate_food_id_integrity(valid_simple_conversation, sample_food_db)
        assert errors == []

    def test_unknown_food_id(self, sample_food_db):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(
                    role="user",
                    content="I ate 100g potatoes. Pad text for length. Pad text for length. "
                            "Pad text for length. Pad text for length.",
                    context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
                ),
                Turn(
                    role="assistant",
                    action="tool_call",
                    tool_calls=[
                        ToolCall(
                            name="add_foods_to_tally",
                            arguments={"items": [{"food_id": 99, "quantity": 100, "unit": "g"}]},
                        )
                    ],
                    tool_results={
                        "add_foods_to_tally": {
                            "entries": [],
                            "tally_total": 0.0,
                        }
                    },
                ),
            ],
        )
        errors = _validate_food_id_integrity(conv, sample_food_db)
        assert len(errors) >= 1
        assert any("99" in e for e in errors)

    def test_known_from_search_result(self, sample_food_db, sample_gram_food):
        from data.mock_harness import _make_search_result
        search_result = _make_search_result(sample_gram_food)
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(
                    role="user",
                    content="I ate 100g potatoes. Pad text. Pad text. Pad text. Pad text.",
                    context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
                ),
                Turn(
                    role="tool",
                    tool_results={"search_foods": {"results": [[search_result]]}},
                ),
                Turn(
                    role="assistant",
                    action="tool_call",
                    tool_calls=[
                        ToolCall(
                            name="add_foods_to_tally",
                            arguments={"items": [{"food_id": 1, "quantity": 100, "unit": "g"}]},
                        )
                    ],
                    tool_results={
                        "add_foods_to_tally": {
                            "entries": [
                                {"entry_id": 1, "food_name": "Potatoes", "quantity": 100, "unit": "g", "carbs": 20.0}
                            ],
                            "tally_total": 20.0,
                        }
                    },
                ),
            ],
        )
        errors = _validate_food_id_integrity(conv, sample_food_db)
        assert errors == []


class TestValidateMath:
    def test_correct_math(self, valid_simple_conversation: Conversation, sample_food_db):
        errors = _validate_math(valid_simple_conversation, sample_food_db)
        assert errors == []

    def test_mismatch(self, sample_food_db, sample_gram_food):
        from data.mock_harness import _make_search_result
        search_result = _make_search_result(sample_gram_food)
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(
                    role="user",
                    content="I ate 100g potatoes. Pad text. Pad text. Pad text. Pad text. Pad text.",
                    context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
                ),
                Turn(role="assistant", action="tool_call",
                     tool_calls=[ToolCall(name="search_foods", arguments={"queries": ["potatoes"]})],
                     tool_results={"search_foods": {"results": [[search_result]]}}),
                Turn(role="tool",
                     tool_results={"search_foods": {"results": [[search_result]]}}),
                Turn(role="assistant", action="tool_call",
                     tool_calls=[ToolCall(name="add_foods_to_tally", arguments={
                         "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
                     })],
                     tool_results={"add_foods_to_tally": {
                         "entries": [{"entry_id": 1, "food_name": "Potatoes", "quantity": 100, "unit": "g", "carbs": 20.0}],
                         "tally_total": 999.0,
                     }}),
            ],
        )
        errors = _validate_math(conv, sample_food_db)
        assert len(errors) >= 1
        assert any("tally_total" in e for e in errors)


class TestValidateLength:
    def test_too_short(self):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content="hi", context_block="[CURRENT TALLY: empty]"),
                Turn(role="assistant", action="text", content="hello"),
            ],
        )
        errors = _validate_length(conv)
        assert len(errors) >= 1
        assert "short" in errors[0].lower()

    def test_valid_min(self):
        padding = "X " * 300
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content=padding, context_block="[CURRENT TALLY: empty]"),
                Turn(role="assistant", action="text", content=padding),
            ],
        )
        errors = _validate_length(conv)
        assert errors == []

    def test_too_long(self):
        padding = "X " * 6000
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content=padding, context_block="[CURRENT TALLY: empty]"),
            ],
        )
        errors = _validate_length(conv)
        assert len(errors) >= 1
        assert "long" in errors[0].lower()


class TestValidateNoEmptyTurns:
    def test_empty_user_content(self):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content=None, context_block="[CURRENT TALLY: empty]"),
            ],
        )
        errors = _validate_no_empty_turns(conv)
        assert len(errors) >= 1

    def test_empty_assistant_text(self):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content="hi", context_block="[CURRENT TALLY: empty]"),
                Turn(role="assistant", action="text", content=None),
            ],
        )
        errors = _validate_no_empty_turns(conv)
        assert len(errors) >= 1

    def test_empty_tool_calls(self):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content="hi", context_block="[CURRENT TALLY: empty]"),
                Turn(role="assistant", action="tool_call", tool_calls=[]),
            ],
        )
        errors = _validate_no_empty_turns(conv)
        assert len(errors) >= 1

    def test_valid_turns(self, valid_simple_conversation: Conversation):
        errors = _validate_no_empty_turns(valid_simple_conversation)
        assert errors == []


class TestValidateContextBlocks:
    def test_missing_context_block(self, invalid_no_context: Conversation):
        errors = _validate_context_blocks(invalid_no_context)
        assert len(errors) >= 1
        assert "context_block" in errors[0].lower()

    def test_all_have_context(self, valid_simple_conversation: Conversation):
        errors = _validate_context_blocks(valid_simple_conversation)
        assert errors == []


class TestValidateUnits:
    def test_correct_unit(self, valid_simple_conversation: Conversation, sample_food_db):
        errors = _validate_units(valid_simple_conversation, sample_food_db)
        assert errors == []

    def test_slice_on_gram_only(self, invalid_bad_unit: Conversation, sample_food_db):
        errors = _validate_units(invalid_bad_unit, sample_food_db)
        assert len(errors) >= 1

    def test_gram_on_piece_only(self, sample_food_db):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(
                    role="user",
                    content="I ate bread. Pad text. Pad text. Pad text. Pad text. "
                            "Pad text. Pad text.",
                    context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: Bread(2)]",
                ),
                Turn(
                    role="assistant",
                    action="tool_call",
                    tool_calls=[
                        ToolCall(
                            name="add_foods_to_tally",
                            arguments={"items": [{"food_id": 2, "quantity": 100, "unit": "g"}]},
                        )
                    ],
                    tool_results={
                        "add_foods_to_tally": {
                            "entries": [],
                            "tally_total": 0.0,
                        }
                    },
                ),
            ],
        )
        errors = _validate_units(conv, sample_food_db)
        assert len(errors) >= 1

    def test_unknown_food_id_skips(self, sample_food_db):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(
                    role="user",
                    content="I ate. Pad text. Pad text. Pad text. Pad text.",
                    context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
                ),
                Turn(
                    role="assistant",
                    action="tool_call",
                    tool_calls=[
                        ToolCall(
                            name="add_foods_to_tally",
                            arguments={"items": [{"food_id": 999, "quantity": 1, "unit": "g"}]},
                        )
                    ],
                    tool_results={
                        "add_foods_to_tally": {
                            "entries": [],
                            "tally_total": 0.0,
                        }
                    },
                ),
            ],
        )
        errors = _validate_units(conv, sample_food_db)
        assert errors == []


class TestValidateConversationIntegration:
    def test_valid_passes(self, valid_simple_conversation: Conversation, sample_food_db):
        errors = validate_conversation(
            valid_simple_conversation, sample_food_db, ScenarioType.SIMPLE_SINGLE_FOOD
        )
        assert errors == []

    def test_invalid_returns_multiple(self, invalid_no_context: Conversation, sample_food_db):
        errors = validate_conversation(
            invalid_no_context, sample_food_db, ScenarioType.SIMPLE_SINGLE_FOOD
        )
        assert len(errors) >= 1

    def test_valid_multi_food(self, valid_multi_food_with_bg: Conversation, sample_food_db):
        errors = validate_conversation(
            valid_multi_food_with_bg, sample_food_db, ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE
        )
        assert errors == []


class TestValidateTextMath:
    def test_exact_match(self):
        conv = Conversation(
            scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="calculate_final", arguments={"blood_glucose": 150.0})
                ], tool_results={"calculate_final": {"final_result": 3.5, "tally_total": 45.0}}),
                Turn(role="tool", tool_results={"calculate_final": {"final_result": 3.5, "tally_total": 45.0}}),
                Turn(role="assistant", action="text", content="Your dose is 3.5 units for 45.0g carbs."),
            ],
        )
        assert _validate_text_math(conv) == []

    def test_rounded_match(self):
        conv = Conversation(
            scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="calculate_final", arguments={})
                ], tool_results={"calculate_final": {"final_result": 3.88}}),
                Turn(role="tool", tool_results={"calculate_final": {"final_result": 3.88}}),
                Turn(role="assistant", action="text", content="Your dose is 4 units."),
            ],
        )
        assert _validate_text_math(conv) == []

    def test_hallucinated_number(self):
        conv = Conversation(
            scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="calculate_final", arguments={})
                ], tool_results={"calculate_final": {"final_result": 3.5}}),
                Turn(role="tool", tool_results={"calculate_final": {"final_result": 3.5}}),
                Turn(role="assistant", action="text", content="Your dose is 10.0 units."),
            ],
        )
        errors = _validate_text_math(conv)
        assert len(errors) == 1
        assert "10.0" in errors[0]

    def test_user_quantity_from_args(self):
        conv = Conversation(
            scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="add_foods_to_tally", arguments={"items": [
                        {"food_id": 1, "quantity": 2, "unit": "pcs"}
                    ]})
                ], tool_results={"add_foods_to_tally": {"entries": [{"entry_id": 1, "food_name": "Bread", "quantity": 2, "unit": "pcs", "carbs": 30.0}], "tally_total": 30.0}}),
                Turn(role="tool", tool_results={"add_foods_to_tally": {"entries": [{"entry_id": 1, "food_name": "Bread", "quantity": 2, "unit": "pcs", "carbs": 30.0}], "tally_total": 30.0}}),
                Turn(role="assistant", action="text", content="Added 2 slices, 30g carbs."),
            ],
        )
        assert _validate_text_math(conv) == []

    def test_no_numbers_in_text(self):
        conv = Conversation(
            scenario_type=ScenarioType.FOOD_NOT_FOUND.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="search_foods", arguments={"queries": ["xyz"]})
                ], tool_results={"search_foods": {"results": [[]]}}),
                Turn(role="tool", tool_results={"search_foods": {"results": [[]]}}),
                Turn(role="assistant", action="text", content="Sorry, I could not find that food. Can you try another name?"),
            ],
        )
        assert _validate_text_math(conv) == []

    def test_no_tool_results(self):
        conv = Conversation(
            scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="text", content="How can I help you?"),
            ],
        )
        assert _validate_text_math(conv) == []

    def test_multiple_numbers_all_match(self):
        conv = Conversation(
            scenario_type=ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="calculate_final", arguments={"blood_glucose": 180.0})
                ], tool_results={
                    "calculate_final": {
                        "final_result": 5.33,
                        "food_insulin": 3.33,
                        "glucose_correction": 2.0,
                        "tally_total": 50.0,
                    }
                }),
                Turn(role="tool", tool_results={
                    "calculate_final": {
                        "final_result": 5.33,
                        "food_insulin": 3.33,
                        "glucose_correction": 2.0,
                        "tally_total": 50.0,
                    }
                }),
                Turn(role="assistant", action="text", content="Total carbs 50g, food insulin 3.33U, correction 2U, final dose 5.33U."),
            ],
        )
        assert _validate_text_math(conv) == []

    def test_partial_hallucination(self):
        conv = Conversation(
            scenario_type=ScenarioType.MULTIPLE_FOODS_NO_GLUCOSE.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="calculate_final", arguments={})
                ], tool_results={"calculate_final": {"final_result": 4.0, "tally_total": 60.0}}),
                Turn(role="tool", tool_results={"calculate_final": {"final_result": 4.0, "tally_total": 60.0}}),
                Turn(role="assistant", action="text", content="60g carbs, your dose is 7.5 units."),
            ],
        )
        errors = _validate_text_math(conv)
        assert len(errors) == 1
        assert "7.5" in errors[0]
        assert "60" not in " ".join(errors)

    def test_blood_glucose_from_args(self):
        conv = Conversation(
            scenario_type=ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE.value,
            language="en",
            turns=[
                Turn(role="user", content="test", context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"),
                Turn(role="assistant", action="tool_call", tool_calls=[
                    ToolCall(name="calculate_final", arguments={"blood_glucose": 150.0})
                ], tool_results={"calculate_final": {"final_result": 4.0, "blood_glucose": 150.0}}),
                Turn(role="tool", tool_results={"calculate_final": {"final_result": 4.0, "blood_glucose": 150.0}}),
                Turn(role="assistant", action="text", content="Your BG is 150 and your dose is 4 units."),
            ],
        )
        assert _validate_text_math(conv) == []

