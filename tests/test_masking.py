from finetuning.common.masking import (
    compute_masked_labels,
    find_assistant_spans_gemma,
    find_assistant_spans_lfm,
    map_char_spans_to_token_indices,
)

_LFM_SAMPLE = (
    "<|startoftext|><|im_start|>system\nSystem prompt<|im_end|>"
    "<|im_start|>user\nUser message<|im_end|>"
    "<|im_start|>assistant\n<|tool_call_start|>search_foods(queries=[\"food\"])<|tool_call_end|><|im_end|>"
    "<|im_start|>tool\n{\"results\": []}<|im_end|>"
    "<|im_start|>assistant\n"
    "<|tool_call_start|>add_foods_to_tally(items=[{\"food_id\": 1, \"quantity\": 100, \"unit\": \"g\"}])<|tool_call_end|>"
    "<|im_end|>"
    "<|im_start|>tool\n{\"results\": []}<|im_end|>"
    "<|im_start|>assistant\nDone!<|im_end|>"
)

_GEMMA_SAMPLE = (
    "<bos><|turn>system\nSystem prompt<turn|>"
    "<|turn>user\nUser message<turn|>"
    "<|turn>model\nsearch_foods({\"queries\": [\"food\"]})<turn|>"
    "<|turn>model\nTool results:\nsearch_foods -> {\"results\": []}<turn|>"
    "<|turn>model\nadd_foods_to_tally({\"items\": [{\"food_id\": 1}]})<turn|>"
    "<|turn>model\nTool results:\nadd_foods_to_tally -> {\"entry_id\": 1}<turn|>"
    "<|turn>model\nDone!<turn|>"
)


class TestFindAssistantSpansLfm:
    def test_finds_tool_call_and_text_assistant_turns(self):
        spans = find_assistant_spans_lfm(_LFM_SAMPLE)
        assert len(spans) == 3

    def test_excludes_tool_turns(self):
        spans = find_assistant_spans_lfm(_LFM_SAMPLE)
        for start, end in spans:
            chunk = _LFM_SAMPLE[start:end]
            assert "<|im_start|>tool" not in chunk

    def test_assistant_span_contains_tool_call(self):
        spans = find_assistant_spans_lfm(_LFM_SAMPLE)
        first_span = _LFM_SAMPLE[spans[0][0] : spans[0][1]]
        assert "<|tool_call_start|>" in first_span
        assert "<|tool_call_end|>" in first_span
        assert "search_foods" in first_span

    def test_assistant_span_contains_im_end(self):
        spans = find_assistant_spans_lfm(_LFM_SAMPLE)
        for start, end in spans:
            chunk = _LFM_SAMPLE[start:end]
            assert chunk.endswith("<|im_end|>")

    def test_no_assistant_turns_returns_empty(self):
        spans = find_assistant_spans_lfm("just some text without markers")
        assert spans == []

    def test_multi_turn_conversation(self):
        text = (
            "<|im_start|>user\nQ1<|im_end|>"
            "<|im_start|>assistant\nA1<|im_end|>"
            "<|im_start|>tool\nR1<|im_end|>"
            "<|im_start|>user\nQ2<|im_end|>"
            "<|im_start|>assistant\nA2<|im_end|>"
        )
        spans = find_assistant_spans_lfm(text)
        assert len(spans) == 2
        assert text[spans[0][0] : spans[0][1]] == "A1<|im_end|>"
        assert text[spans[1][0] : spans[1][1]] == "A2<|im_end|>"


class TestFindAssistantSpansGemma:
    def test_finds_model_tool_call_turns(self):
        spans = find_assistant_spans_gemma(_GEMMA_SAMPLE)
        assert len(spans) == 3

    def test_excludes_tool_result_turns(self):
        spans = find_assistant_spans_gemma(_GEMMA_SAMPLE)
        for start, end in spans:
            chunk = _GEMMA_SAMPLE[start:end]
            assert not chunk.startswith("Tool results:")

    def test_includes_text_model_turns(self):
        spans = find_assistant_spans_gemma(_GEMMA_SAMPLE)
        last_span = _GEMMA_SAMPLE[spans[-1][0] : spans[-1][1]]
        assert last_span == "Done!<turn|>"

    def test_includes_tool_call_model_turns(self):
        spans = find_assistant_spans_gemma(_GEMMA_SAMPLE)
        first_span = _GEMMA_SAMPLE[spans[0][0] : spans[0][1]]
        assert "search_foods" in first_span
        assert first_span.endswith("<turn|>")

    def test_no_model_turns_returns_empty(self):
        spans = find_assistant_spans_gemma("just some text")
        assert spans == []

    def test_multi_turn_conversation(self):
        text = (
            "<|turn>user\nQ1<turn|>"
            "<|turn>model\nA1<turn|>"
            "<|turn>model\nTool results:\nR1<turn|>"
            "<|turn>user\nQ2<turn|>"
            "<|turn>model\nA2<turn|>"
        )
        spans = find_assistant_spans_gemma(text)
        assert len(spans) == 2
        assert text[spans[0][0] : spans[0][1]] == "A1<turn|>"
        assert text[spans[1][0] : spans[1][1]] == "A2<turn|>"


class TestMapCharSpansToTokenIndices:
    def test_simple_overlap(self):
        offset_mapping = [(0, 3), (3, 6), (6, 9), (9, 12)]
        char_spans = [(3, 9)]
        result = map_char_spans_to_token_indices(char_spans, offset_mapping)
        assert result == {1, 2}

    def test_full_span(self):
        offset_mapping = [(0, 5), (5, 10)]
        char_spans = [(0, 10)]
        result = map_char_spans_to_token_indices(char_spans, offset_mapping)
        assert result == {0, 1}

    def test_partial_overlap_start(self):
        offset_mapping = [(0, 5), (5, 10), (10, 15)]
        char_spans = [(2, 7)]
        result = map_char_spans_to_token_indices(char_spans, offset_mapping)
        assert result == {0, 1}

    def test_multiple_spans(self):
        offset_mapping = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
        char_spans = [(0, 3), (9, 15)]
        result = map_char_spans_to_token_indices(char_spans, offset_mapping)
        assert result == {0, 3, 4}

    def test_no_overlap(self):
        offset_mapping = [(0, 3), (3, 6)]
        char_spans = [(10, 12)]
        result = map_char_spans_to_token_indices(char_spans, offset_mapping)
        assert result == set()

    def test_empty_spans(self):
        offset_mapping = [(0, 3)]
        result = map_char_spans_to_token_indices([], offset_mapping)
        assert result == set()

    def test_skips_special_tokens_zero_width(self):
        offset_mapping = [(0, 0), (0, 5), (5, 10)]
        char_spans = [(0, 10)]
        result = map_char_spans_to_token_indices(char_spans, offset_mapping)
        assert result == {1, 2}


class _BatchedTokenizer:
    def __call__(self, **kw):
        t = kw.get("text", [""])
        t = t[0] if isinstance(t, list) else t
        n = len(t)
        ids = list(range(1, n + 1))
        om = [(i, i + 1) for i in range(n)]
        return {"input_ids": [ids], "offset_mapping": [om]}


class TestComputeMaskedLabels:
    _text = "<|im_start|>assistant\ntext<|im_end|>"

    def test_lfm_batched_output(self):
        tok = _BatchedTokenizer()
        input_ids, labels = compute_masked_labels(self._text, tok, "lfm", max_length=128)
        assert len(input_ids) == len(self._text)
        assert len(labels) == len(self._text)
        assert any(l != -100 for l in labels)

    def test_gemma_batched_output(self):
        tok = _BatchedTokenizer()
        input_ids, labels = compute_masked_labels(
            "<|turn>model\ntext<turn|>", tok, "gemma", max_length=128
        )
        assert len(input_ids) == 24
        assert any(l != -100 for l in labels)

    def test_empty_assistant_spans_all_masked(self):
        tok = _BatchedTokenizer()
        input_ids, labels = compute_masked_labels(
            "just user text no assistant", tok, "lfm", max_length=128
        )
        assert all(l == -100 for l in labels)
