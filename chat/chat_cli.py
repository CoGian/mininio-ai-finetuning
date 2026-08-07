import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.food_db_loader import load_food_db
from data.log_config import setup_logging
from data.mock_harness import (
    USER_SETTINGS_POOL,
    MockHarness,
    format_user_settings_training,
)
from evaluation.evaluate import parse_tool_call_gemma, parse_tool_call_lfm


def _serialize_gemma_tool_results(result: dict) -> str:
    lines = ["Tool results:"]
    for tool_name, sub_result in result.items():
        lines.append(
            f"{tool_name} -> "
            f"{json.dumps(sub_result, ensure_ascii=False, indent=2)}",
        )
    return "\n".join(lines)


MODEL_FORMATS = {
    "lfm": {
        "bos": "<|startoftext|>",
        "system_open": "<|im_start|>system\n",
        "system_close": "<|im_end|>",
        "user_open": "<|im_start|>user\n",
        "user_close": "<|im_end|>",
        "tool_open": "<|im_start|>tool\n",
        "tool_close": "<|im_end|>",
        "assistant_marker": "<|im_start|>assistant\n",
        "template_file": Path("data/prompts/system_lfm.txt"),
        "parse_tc": parse_tool_call_lfm,
        "result_formatter": lambda j: j,
    },
    "gemma": {
        "bos": "<bos>",
        "system_open": "<|turn>system\n",
        "system_close": "<turn|>",
        "user_open": "<|turn>user\n",
        "user_close": "<turn|>",
        "tool_open": "<|turn>model\n",
        "tool_close": "<turn|>",
        "assistant_marker": "<|turn>model\n",
        "template_file": Path("data/prompts/system_gemma.txt"),
        "parse_tc": parse_tool_call_gemma,
        "result_formatter": _serialize_gemma_tool_results,
    },
}


def _build_system_prefix(model_type: str, settings_idx: int, lang: str) -> str:
    cfg = MODEL_FORMATS[model_type]
    settings = USER_SETTINGS_POOL[min(settings_idx, len(USER_SETTINGS_POOL) - 1)]
    template = cfg["template_file"].read_text(encoding="utf-8")
    template = template.replace("%{user_settings}", format_user_settings_training(settings))
    template = template.replace("%{user_language}", lang)
    return (
        f"{cfg['bos']}"
        f"{cfg['system_open']}{template}{cfg['system_close']}"
    )


def _append_user_turn(conversation: str, user_input: str, harness: MockHarness, model_type: str) -> str:
    cfg = MODEL_FORMATS[model_type]
    ctx = harness.get_context_block()
    return f"{conversation}\n{cfg['user_open']}{user_input}\n\n{ctx}{cfg['user_close']}"


def _append_assistant_prefix(conversation: str, model_type: str) -> str:
    return conversation + MODEL_FORMATS[model_type]["assistant_marker"]


def _append_tool_result(conversation: str, result: dict, model_type: str) -> str:
    cfg = MODEL_FORMATS[model_type]
    text = cfg["result_formatter"](result)
    return f"{conversation}{cfg['tool_open']}{text}{cfg['tool_close']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mininio AI chat — test the fine-tuned model interactively")
    parser.add_argument("--model-dir", default="finetuning/output/lfm/merged_16bit")
    parser.add_argument("--model-type", default="lfm", choices=["lfm", "gemma"])
    parser.add_argument("--settings-idx", type=int, default=0)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    setup_logging()

    settings_idx = min(args.settings_idx, len(USER_SETTINGS_POOL) - 1)
    settings = USER_SETTINGS_POOL[settings_idx]
    cfg = MODEL_FORMATS[args.model_type]

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if device == "cpu" else torch.float16
    logger.info(f"Loading model from {args.model_dir} ({device}, {dtype})")

    model_dir = str(Path(args.model_dir).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if args.model_type == "gemma":
        try:
            from unsloth.chat_templates import get_chat_template

            tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
            logger.info("Applied Gemma-4 chat template via Unsloth")
        except ImportError:
            logger.warning("Unsloth not available — using raw tokenizer. "
                           "Gemma may not produce correct output without the chat template.")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    logger.info("Model loaded.")

    food_db = load_food_db(args.lang)
    harness = MockHarness(food_db, settings=settings)
    conversation = _build_system_prefix(args.model_type, settings_idx, args.lang)

    print("\nMininio CLI Chat")
    print("  Type your food intake messages (e.g. 'I ate 150g of grapes with lunch')")
    print("  Prefix commands: /clear, /settings, /help, /quit")
    print("  Ctrl+C to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/q"):
                break
            elif cmd == "/clear":
                harness.reset()
                conversation = _build_system_prefix(args.model_type, settings_idx, args.lang)
                print("  [harness and conversation reset]\n")
                continue
            elif cmd == "/settings":
                td = settings["glucose_threshold"]
                bl = settings["glucose_baseline"]
                dv = settings["glucose_divisor"]
                md = settings["meal_dividers"]
                print(f"  Glucose: threshold={td} baseline={bl} divisor={dv}")
                print(f"  Meal dividers: morning={md['morning']} midday={md['midday']} evening={md['evening']}")
                print(f"  Settings pool index: {settings_idx}\n")
                continue
            elif cmd == "/help":
                print("  /clear     Reset harness and start fresh")
                print("  /settings  Show current user settings")
                print("  /quit      Exit\n")
                continue
            else:
                print(f"  Unknown command: {user_input}\n")
                continue

        conversation = _append_user_turn(conversation, user_input, harness, args.model_type)
        context = harness.get_context_block()
        print(f"\n[context block]\n{context}")
        conversation = _append_assistant_prefix(conversation, args.model_type)

        for _ in range(args.max_turns):
            inputs = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temp,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            conversation += new_text

            print("\n[assistant output]")
            print(new_text)

            tc = cfg["parse_tc"](new_text)
            if tc:
                result = harness.execute(tc)
                result_json = json.dumps(result, ensure_ascii=False, indent=2)
                print("\n[tool result]")
                print(result_json)
                conversation = _append_tool_result(conversation, result, args.model_type)
                conversation = _append_assistant_prefix(conversation, args.model_type)
            else:
                break

        print()

    print("Goodbye.")


if __name__ == "__main__":
    main()
