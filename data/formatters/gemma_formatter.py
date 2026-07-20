"""
Converts model-agnostic Conversation -> Gemma 4 Unsloth training string.

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

import json

from data.scenarios import Conversation


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

        elif turn.role == "tool":
            results = turn.tool_results
            if results:
                results_text = _serialize_gemma_tool_results(results)
                parts.append(f"<|turn>model\n{results_text}<turn|>")

    return "".join(parts)


def _serialize_gemma_tool_calls(tool_calls: list) -> str:
    lines = []
    for tc in tool_calls:
        args_json = json.dumps(tc.arguments, ensure_ascii=False)
        lines.append(f"{tc.name}({args_json})")
    return "\n".join(lines)


def _serialize_gemma_tool_results(results: dict) -> str:
    lines = ["Tool results:"]
    for tool_name, result in results.items():
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        lines.append(f"{tool_name} -> {result_json}")
    return "\n".join(lines)
