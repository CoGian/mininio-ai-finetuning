import pytest
from pydantic import ValidationError

from data.scenarios import (
    Conversation,
    Turn,
    ToolCall,
    ScenarioType,
    SCENARIO_WEIGHTS,
    LANGUAGE_NAMES,
    EXPECTED_SEQUENCES,
)


class TestConversationModel:
    def test_valid_conversation(self, valid_simple_conversation: Conversation):
        assert valid_simple_conversation.scenario_type == ScenarioType.SIMPLE_SINGLE_FOOD.value
        assert valid_simple_conversation.language == "en"
        assert len(valid_simple_conversation.turns) == 6

    def test_missing_turns(self):
        with pytest.raises(ValidationError):
            Conversation.model_validate({"scenario_type": "SIMPLE_SINGLE_FOOD", "language": "en"})

    def test_wrong_role(self):
        with pytest.raises(ValidationError):
            Turn.model_validate({"role": "system", "content": "hello"})

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            Turn.model_validate({"role": "assistant", "action": "invalid"})

    def test_tool_call_required_name(self):
        with pytest.raises(ValidationError):
            ToolCall.model_validate({"arguments": {}})

    def test_conversation_roundtrip(self, valid_simple_conversation: Conversation):
        json_str = valid_simple_conversation.model_dump_json()
        reloaded = Conversation.model_validate_json(json_str)
        assert reloaded.scenario_type == valid_simple_conversation.scenario_type
        assert reloaded.language == valid_simple_conversation.language
        assert len(reloaded.turns) == len(valid_simple_conversation.turns)
        assert reloaded.turns[0].content == valid_simple_conversation.turns[0].content


class TestScenarioWeights:
    def test_weights_sum_to_one(self):
        total = sum(SCENARIO_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_all_scenarios_have_weights(self):
        for st in ScenarioType:
            assert st in SCENARIO_WEIGHTS


class TestLanguageNames:
    def test_all_ten_languages_present(self):
        assert len(LANGUAGE_NAMES) == 10
        for code in ["en", "el", "fr", "es", "hi", "it", "pt", "zh", "de", "ja"]:
            assert code in LANGUAGE_NAMES


class TestExpectedSequences:
    def test_all_scenarios_have_sequence(self):
        for st in ScenarioType:
            assert st in EXPECTED_SEQUENCES

    def test_sequences_are_list_of_lists(self):
        for st, seq in EXPECTED_SEQUENCES.items():
            assert isinstance(seq, list)
            for branch in seq:
                assert isinstance(branch, list)
                for tool_name in branch:
                    assert isinstance(tool_name, str)
