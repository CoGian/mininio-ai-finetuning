import re
from collections.abc import Callable
from typing import Literal

_ASSISTANT_LFM_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?(?:<\|tool_call_start\|>.*?<\|tool_call_end\|>)?.*?<\|im_end\|>)",
)

_TOOL_RESULT_GEMMA_RE = re.compile(r"^Tool results:")

_MODEL_GEMMA_RE = re.compile(r"<\|turn>model\n(.*?<turn\|>)")


def find_assistant_spans_lfm(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _ASSISTANT_LFM_RE.finditer(text):
        spans.append((match.start(1), match.end(1)))
    return spans


def find_assistant_spans_gemma(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _MODEL_GEMMA_RE.finditer(text):
        content_start = match.start(1)
        content = text[content_start : match.end(1)]
        if _TOOL_RESULT_GEMMA_RE.match(content):
            continue
        spans.append((content_start, match.end(1)))
    return spans


def map_char_spans_to_token_indices(
    char_spans: list[tuple[int, int]],
    offset_mapping: list[tuple[int, int]],
) -> set[int]:
    token_indices: set[int] = set()
    span_idx = 0
    n_spans = len(char_spans)
    if n_spans == 0:
        return token_indices

    current_start, current_end = char_spans[span_idx]

    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue

        while span_idx < n_spans and tok_start >= current_end:
            span_idx += 1
            if span_idx < n_spans:
                current_start, current_end = char_spans[span_idx]

        if span_idx >= n_spans:
            break

        if tok_end > current_start and tok_start < current_end:
            token_indices.add(tok_idx)

    return token_indices


def compute_masked_labels(
    text: str,
    tokenizer,
    model_type: Literal["lfm", "gemma"],
    max_length: int = 4096,
) -> tuple[list[int], list[int]]:
    encoding = tokenizer(
        text=[text],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids: list[int] = encoding["input_ids"][0]
    offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"][0]

    if model_type == "lfm":
        char_spans = find_assistant_spans_lfm(text)
    else:
        char_spans = find_assistant_spans_gemma(text)

    unmasked = map_char_spans_to_token_indices(char_spans, offset_mapping)
    labels = [-100] * len(input_ids)
    for idx in unmasked:
        labels[idx] = input_ids[idx]

    return input_ids, labels


def make_masking_fn(
    tokenizer,
    model_type: Literal["lfm", "gemma"],
    max_length: int = 4096,
) -> Callable[[dict], dict]:
    def mask_fn(examples: dict) -> dict:
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        for text in examples["text"]:
            input_ids, labels = compute_masked_labels(
                text, tokenizer, model_type, max_length
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        return {"input_ids": all_input_ids, "labels": all_labels}

    return mask_fn
