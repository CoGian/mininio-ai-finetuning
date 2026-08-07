import pytest
from evaluation.evaluate import (
    _args_match,
    _extract_final_result,
    _format_turn,
    _is_clarification,
    parse_gemma_turns,
    parse_lfm_turns,
    parse_tool_call_gemma,
    parse_tool_call_lfm,
)


class TestParseLfmTurns:
    def test_complete_conversation(self) -> None:
        text = (
            "<|im_start|>system\nYou are a helper<|im_end|>"
            "<|im_start|>user\nHello<|im_end|>"
            "<|im_start|>assistant\nHi there!<|im_end|>"
            "<|im_start|>tool\n{\"result\": 1}<|im_end|>"
        )
        turns = parse_lfm_turns(text)
        assert len(turns) == 4
        assert turns[0] == {"role": "system", "content": "You are a helper"}
        assert turns[1] == {"role": "user", "content": "Hello"}
        assert turns[2] == {"role": "assistant", "content": "Hi there!"}
        assert turns[3] == {"role": "tool", "content": '{"result": 1}'}

    def test_empty_string(self) -> None:
        assert parse_lfm_turns("") == []

    def test_no_markers(self) -> None:
        assert parse_lfm_turns("just some text") == []

    def test_multiline_user_content(self) -> None:
        text = "<|im_start|>user\nline1\nline2\n\nline3<|im_end|>"
        turns = parse_lfm_turns(text)
        assert len(turns) == 1
        assert turns[0]["content"] == "line1\nline2\n\nline3"

    def test_tool_call_content(self) -> None:
        text = (
            "<|im_start|>assistant\n"
            "<|tool_call_start|>search_foods(queries=[\"Apple\"])<|tool_call_end|>"
            "<|im_end|>"
        )
        turns = parse_lfm_turns(text)
        assert len(turns) == 1
        assert (
            turns[0]["content"]
            == '<|tool_call_start|>search_foods(queries=["Apple"])<|tool_call_end|>'
        )


class TestParseGemmaTurns:
    def test_complete_conversation(self) -> None:
        text = (
            "<bos><|turn>system\nYou are a helper<turn|>"
            "<|turn>user\nHello<turn|>"
            "<|turn>model\nHi there!<turn|>"
        )
        turns = parse_gemma_turns(text)
        assert len(turns) == 3
        assert turns[0] == {"role": "system", "content": "You are a helper"}
        assert turns[1] == {"role": "user", "content": "Hello"}
        assert turns[2] == {"role": "assistant", "content": "Hi there!"}

    def test_model_role_converted_to_assistant(self) -> None:
        text = "<bos><|turn>model\nsome output<turn|>"
        turns = parse_gemma_turns(text)
        assert len(turns) == 1
        assert turns[0]["role"] == "assistant"

    def test_no_bos_prefix(self) -> None:
        text = "<|turn>user\nHello<turn|>"
        turns = parse_gemma_turns(text)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"

    def test_empty_string(self) -> None:
        assert parse_gemma_turns("") == []


class TestParseToolCallLfm:
    def test_search_foods(self) -> None:
        content = '<|tool_call_start|>search_foods(queries=["Apple"])<|tool_call_end|>'
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["name"] == "search_foods"
        assert result["arguments"] == {"queries": ["Apple"]}

    def test_add_foods_to_tally(self) -> None:
        content = (
            '<|tool_call_start|>'
            'add_foods_to_tally(items=[{"food_id": 49, "quantity": 150, "unit": "g"}])'
            '<|tool_call_end|>'
        )
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["name"] == "add_foods_to_tally"
        assert result["arguments"] == {
            "items": [{"food_id": 49, "quantity": 150, "unit": "g"}],
        }

    def test_multiple_calls_comma_separated(self) -> None:
        content = (
            '<|tool_call_start|>'
            'search_foods(queries=["A"]), add_foods_to_tally(items=[{"food_id": 1, "quantity": 100, "unit": "g"}])'
            '<|tool_call_end|>'
        )
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["name"] == "search_foods"

    def test_no_tool_call(self) -> None:
        assert parse_tool_call_lfm("regular text response") is None

    def test_malformed_tool_call_returns_none(self) -> None:
        content = "<|tool_call_start|>not_a_tool()<|tool_call_end|>"
        result = parse_tool_call_lfm(content)
        assert result is not None

    def test_calculate_final(self) -> None:
        content = (
            '<|tool_call_start|>'
            'calculate_final(meal_time="morning", blood_glucose=120.5)'
            '<|tool_call_end|>'
        )
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["name"] == "calculate_final"
        assert result["arguments"] == {"meal_time": "morning", "blood_glucose": 120.5}

    def test_clear_all_no_args(self) -> None:
        content = "<|tool_call_start|>clear_all()<|tool_call_end|>"
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["name"] == "clear_all"

    def test_remove_foods_from_tally_int_list(self) -> None:
        content = "<|tool_call_start|>remove_foods_from_tally(entry_ids=[1])<|tool_call_end|>"
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["name"] == "remove_foods_from_tally"
        assert result["arguments"] == {"entry_ids": [1]}

    def test_remove_foods_from_tally_multi_ids(self) -> None:
        content = "<|tool_call_start|>remove_foods_from_tally(entry_ids=[1, 3, 5])<|tool_call_end|>"
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["arguments"] == {"entry_ids": [1, 3, 5]}

    def test_string_list_still_works(self) -> None:
        content = '<|tool_call_start|>search_foods(queries=["Apple", "Banana"])<|tool_call_end|>'
        result = parse_tool_call_lfm(content)
        assert result is not None
        assert result["arguments"] == {"queries": ["Apple", "Banana"]}


class TestParseToolCallGemma:
    def test_search_foods_json_args(self) -> None:
        content = 'search_foods({"queries": ["Apple"]})'
        result = parse_tool_call_gemma(content)
        assert result is not None
        assert result["name"] == "search_foods"
        assert result["arguments"] == {"queries": ["Apple"]}

    def test_no_tool_call(self) -> None:
        assert parse_tool_call_gemma("regular text response") is None

    def test_malformed_json_returns_none(self) -> None:
        assert parse_tool_call_gemma("search_foods({broken json)") is None


class TestExtractFinalResult:
    def test_valid(self) -> None:
        assert _extract_final_result('{"final_result": 3.5}') == pytest.approx(3.5)

    def test_integer_result(self) -> None:
        assert _extract_final_result('{"final_result": 4}') == 4.0

    def test_missing(self) -> None:
        assert _extract_final_result("no result here") is None

    def test_embedded_in_text(self) -> None:
        text = 'Here is your dose: {"final_result": 2.75} units'
        assert _extract_final_result(text) == pytest.approx(2.75)

    def test_multiple_matches_first_wins(self) -> None:
        text = '{"final_result": 1.0} then {"final_result": 2.0}'
        assert _extract_final_result(text) == 1.0


class TestIsClarification:
    def test_question_without_tool_call(self) -> None:
        assert _is_clarification("Which food did you mean?") is True

    def test_tool_call_block_not_clarification(self) -> None:
        assert (
            _is_clarification(
                '<|tool_call_start|>search_foods(queries=["Apple"])<|tool_call_end|>'
            )
            is False
        )

    def test_plain_text_not_clarification(self) -> None:
        assert _is_clarification("Here is your result") is False

    def test_gemma_tool_call_not_clarification(self) -> None:
        assert _is_clarification('search_foods({"queries": ["Apple"]})') is False

    def test_empty_string(self) -> None:
        assert _is_clarification("") is False


class TestArgsMatch:
    def test_exact_match(self) -> None:
        a = {"queries": ["Apple"]}
        b = {"queries": ["Apple"]}
        assert _args_match(a, b) is True

    def test_different_keys(self) -> None:
        a = {"queries": ["Apple"]}
        b = {"query": ["Apple"]}
        assert _args_match(a, b) is False

    def test_different_values(self) -> None:
        a = {"queries": ["Apple"]}
        b = {"queries": ["Banana"]}
        assert _args_match(a, b) is False

    def test_extra_key(self) -> None:
        a = {"queries": ["Apple"]}
        b = {"queries": ["Apple"], "limit": 5}
        assert _args_match(a, b) is False

    def test_empty_dicts(self) -> None:
        assert _args_match({}, {}) is True


class TestFormatTurn:
    def test_lfm_user(self) -> None:
        result = _format_turn({"role": "user", "content": "Hello"}, "lfm")
        assert result == "<|im_start|>user\nHello<|im_end|>"

    def test_lfm_assistant(self) -> None:
        result = _format_turn({"role": "assistant", "content": "Hi there"}, "lfm")
        assert result == "<|im_start|>assistant\nHi there<|im_end|>"

    def test_lfm_tool(self) -> None:
        result = _format_turn({"role": "tool", "content": '{"result": 1}'}, "lfm")
        assert result == '<|im_start|>tool\n{"result": 1}<|im_end|>'

    def test_gemma_user(self) -> None:
        result = _format_turn({"role": "user", "content": "Hello"}, "gemma")
        assert result == "<|turn>user\nHello<turn|>"

    def test_gemma_model_role_mapped(self) -> None:
        result = _format_turn({"role": "assistant", "content": "Hi"}, "gemma")
        assert result == "<|turn>model\nHi<turn|>"
