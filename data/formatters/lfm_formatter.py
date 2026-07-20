"""
Converts model-agnostic Conversation -> LFM2.5 ChatML training string.

Format reference: lfm2_5_sft_with_unsloth.py (lines 103-109, 146-150)

Key features:
- Role markers: <|im_start|>system/user/assistant/tool + <|im_end|>
- Tool calls: wrapped in <|tool_call_start|> / <|tool_call_end|>
- BOS token: <|startoftext|> included, removed by formatting_prompts_func
- train_on_responses_only: instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"
"""

import json

from data.scenarios import Conversation


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

        elif turn.role == "tool":
            results = turn.tool_results
            if results:
                results_json = json.dumps(results, ensure_ascii=False)
                parts.append(f"<|im_start|>tool\n{results_json}<|im_end|>")

    return "".join(parts)


def _serialize_lfm_tool_call(tc) -> str:
    args_strs = []
    for key, value in tc.arguments.items():
        if isinstance(value, str):
            args_strs.append(f'{key}="{value}"')
        elif isinstance(value, list):
            if value and isinstance(value[0], str):
                items = ", ".join(f'"{v}"' for v in value)
                args_strs.append(f"{key}=[{items}]")
            else:
                items = ", ".join(json.dumps(v, ensure_ascii=False) for v in value)
                args_strs.append(f"{key}=[{items}]")
        else:
            args_strs.append(f"{key}={value}")
    return f"{tc.name}({', '.join(args_strs)})"
