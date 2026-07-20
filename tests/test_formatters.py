import json

import pytest

from data.scenarios import Conversation, Turn, ToolCall, ScenarioType
from data.formatters import format_lfm, format_gemma

SYSTEM_PROMPT = "You are a helpful carb counting assistant."


class TestLfmFormatter:
    def test_starts_with_bos(self, valid_simple_conversation: Conversation):
        result = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        assert result.startswith("<|startoftext|>")

    def test_system_prompt_embedded(self, valid_simple_conversation: Conversation):
        result = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        assert "<|im_start|>system\n" in result
        assert SYSTEM_PROMPT in result
        assert "<|im_end|>" in result

    def test_user_with_context_block(self, valid_simple_conversation: Conversation):
        result = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        assert "<|im_start|>user\n" in result
        assert "CURRENT TALLY" in result
        assert "<|im_end|>" in result

    def test_tool_call_wrapped(self, valid_simple_conversation: Conversation):
        result = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        assert "<|tool_call_start|>" in result
        assert "<|tool_call_end|>" in result
        assert "search_foods" in result

    def test_assistant_text(self, valid_simple_conversation: Conversation):
        result = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        assert "<|im_start|>assistant\n" in result
        assert "potatoes" in result.lower()
        assert "20" in result

    def test_tool_results_json(self, valid_simple_conversation: Conversation):
        result = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        assert "<|im_start|>tool\n" in result
        assert "search_foods" in result

    def test_serialize_tool_call_string_args(self):
        from data.formatters.lfm_formatter import _serialize_lfm_tool_call

        tc = ToolCall(name="search_foods", arguments={"queries": ["potatoes"]})
        result = _serialize_lfm_tool_call(tc)
        assert 'queries=["potatoes"]' in result

    def test_serialize_tool_call_numeric_args(self):
        from data.formatters.lfm_formatter import _serialize_lfm_tool_call

        tc = ToolCall(name="add_foods_to_tally", arguments={
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        result = _serialize_lfm_tool_call(tc)
        assert "food_id" in result
        assert "100" in result

    def test_serialize_tool_call_list_of_objects(self):
        from data.formatters.lfm_formatter import _serialize_lfm_tool_call

        tc = ToolCall(name="add_foods_to_tally", arguments={
            "items": [
                {"food_id": 1, "quantity": 100, "unit": "g"},
                {"food_id": 2, "quantity": 1, "unit": "slice"},
            ]
        })
        result = _serialize_lfm_tool_call(tc)
        assert "food_id" in result
        assert "unit" in result


class TestGemmaFormatter:
    def test_starts_with_bos(self, valid_simple_conversation: Conversation):
        result = format_gemma(valid_simple_conversation, SYSTEM_PROMPT)
        assert result.startswith("<bos>")

    def test_system_prompt_embedded(self, valid_simple_conversation: Conversation):
        result = format_gemma(valid_simple_conversation, SYSTEM_PROMPT)
        assert "<|turn>system\n" in result
        assert SYSTEM_PROMPT in result
        assert "<turn|>" in result

    def test_tool_call_inline(self, valid_simple_conversation: Conversation):
        result = format_gemma(valid_simple_conversation, SYSTEM_PROMPT)
        assert "search_foods(" in result
        assert "<|turn>model\n" in result

    def test_tool_results_section(self, valid_simple_conversation: Conversation):
        result = format_gemma(valid_simple_conversation, SYSTEM_PROMPT)
        assert "Tool results:" in result

    def test_user_with_context(self, valid_simple_conversation: Conversation):
        result = format_gemma(valid_simple_conversation, SYSTEM_PROMPT)
        assert "<|turn>user\n" in result
        assert "CURRENT TALLY" in result


class TestFormattersIntegration:
    def test_both_nonempty(self, valid_simple_conversation: Conversation):
        lfm = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        gemma = format_gemma(valid_simple_conversation, SYSTEM_PROMPT)
        assert len(lfm) > 0
        assert len(gemma) > 0

    def test_different_output(self, valid_simple_conversation: Conversation):
        lfm = format_lfm(valid_simple_conversation, SYSTEM_PROMPT)
        gemma = format_gemma(valid_simple_conversation, SYSTEM_PROMPT)
        assert lfm != gemma

    def test_format_contains_all_tool_calls(self, valid_multi_food_with_bg: Conversation):
        lfm = format_lfm(valid_multi_food_with_bg, SYSTEM_PROMPT)
        gemma = format_gemma(valid_multi_food_with_bg, SYSTEM_PROMPT)
        assert "search_foods" in lfm
        assert "add_foods_to_tally" in lfm
        assert "calculate_final" in lfm
        assert "search_foods" in gemma
        assert "add_foods_to_tally" in gemma
        assert "calculate_final" in gemma
