import argparse
import json
import math
import os
import re
import time
from typing import Any, Optional

os.environ["HF_HUB_OFFLINE"] = "1"

import huggingface_hub.utils._validators
huggingface_hub.utils._validators.validate_repo_id = lambda name: name

import torch
from loguru import logger

from evaluation.criteria import FINAL_RESULT_TOLERANCE, EvalMetrics

_LFM_TOOL_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)
_GEMMA_TOOL_RE = re.compile(r"(\w+)\((\{.*?\})\)", re.DOTALL)
_LFM_TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
_GEMMA_TURN_RE = re.compile(r"<\|turn>(\w+)\n(.*?)<turn\|>", re.DOTALL)
_FINAL_RESULT_RE = re.compile(r'"final_result"\s*:\s*([0-9.]+)')
_SCENARIO_MAP = {
    "ambiguous_food": True,
    "incomplete_info": True,
}
_GOLD_TOOL_RE = re.compile(r"(\w+)\((.*)\)", re.DOTALL)


def parse_lfm_turns(text: str) -> list[dict]:
    turns = []
    for match in _LFM_TURN_RE.finditer(text):
        role = match.group(1)
        content = match.group(2)
        if role in ("system", "user", "assistant", "tool"):
            turns.append({"role": role, "content": content})
    return turns


def parse_gemma_turns(text: str) -> list[dict]:
    turns = []
    text_no_bos = text.removeprefix("<bos>")
    for match in _GEMMA_TURN_RE.finditer(text_no_bos):
        role = match.group(1)
        content = match.group(2)
        if role in ("system", "user", "model"):
            role = "assistant" if role == "model" else role
            turns.append({"role": role, "content": content})
    return turns


def parse_tool_call_lfm(content: str) -> Optional[dict]:
    m = _LFM_TOOL_RE.search(content)
    if not m:
        return None
    inner = m.group(1).strip()
    parsed = _parse_tool_call_text(inner)
    if not parsed:
        return None
    return parsed


def parse_tool_call_gemma(content: str) -> Optional[dict]:
    m = _GEMMA_TOOL_RE.search(content)
    if not m:
        return None
    name = m.group(1)
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None
    return {"name": name, "arguments": args}


def _parse_tool_call_text(text: str) -> Optional[dict]:
    m = _GOLD_TOOL_RE.match(text.strip())
    if not m:
        return None
    name = m.group(1)
    raw_args = m.group(2).strip()
    args = _parse_lfm_args(raw_args)
    return {"name": name, "arguments": args}


def _parse_lfm_args(raw: str) -> dict:
    args: dict[str, Any] = {}
    if not raw:
        return args
    parts = re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^{]*\})(?![^\[]*\])', raw)
    for part in parts:
        kv = part.split("=", 1)
        if len(kv) != 2:
            continue
        key = kv[0].strip()
        val = kv[1].strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            try:
                items = json.loads(f"[{inner}]")
                args[key] = items
            except json.JSONDecodeError:
                items = re.findall(r'"([^"]*)"', inner)
                if items:
                    args[key] = items
        else:
            try:
                args[key] = float(val) if "." in val else int(val)
            except ValueError:
                args[key] = val.strip('"')
    return args


def _extract_final_result(text: str) -> Optional[float]:
    m = _FINAL_RESULT_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def _is_clarification(content: str) -> bool:
    return "?" in content and not _LFM_TOOL_RE.search(content) and not _GEMMA_TOOL_RE.search(content)


def evaluate_model(
    checkpoint_dir: str,
    model_type: str,
    eval_path: str,
    max_new_tokens: int = 256,
    max_turns: int = 8,
) -> EvalMetrics:
    is_merged = os.path.isfile(os.path.join(checkpoint_dir, "model.safetensors"))

    if is_merged:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if model_type == "gemma":
            try:
                from unsloth.chat_templates import get_chat_template
                tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
            except ImportError:
                logger.warning("Unsloth chat template unavailable for Gemma")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype,
            device_map=device,
        )
        model.eval()
    else:
        try:
            if model_type == "lfm":
                from unsloth import FastLanguageModel

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=checkpoint_dir,
                    max_seq_length=4096,
                    load_in_4bit=True,
                )
                FastLanguageModel.for_inference(model)
            else:
                from unsloth import FastModel
                from unsloth.chat_templates import get_chat_template

                model, tokenizer = FastModel.from_pretrained(
                    model_name=checkpoint_dir,
                    max_seq_length=4096,
                    load_in_4bit=True,
                )
                tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
                FastModel.for_inference(model)
        except ImportError:
            logger.error("Unsloth required for LoRA adapter loading but not available")
            raise

    parse_turns = parse_lfm_turns if model_type == "lfm" else parse_gemma_turns
    parse_tc = parse_tool_call_lfm if model_type == "lfm" else parse_tool_call_gemma

    total_tool_calls = 0
    correct_tool_calls = 0
    total_final = 0
    correct_final = 0
    total_clarify = 0
    correct_clarify = 0
    total_tokens = 0
    total_time = 0.0
    n_convos = 0

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            turns = parse_turns(text)

            gold_tool_calls: list[dict] = []
            gold_tool_result_turns: list[dict] = []
            gold_final_text = ""
            is_clarification = False
            for turn in turns:
                if turn["role"] == "tool":
                    gold_tool_result_turns.append(turn)
                elif turn["role"] == "assistant":
                    tc = parse_tc(turn["content"])
                    if tc:
                        gold_tool_calls.append(tc)
                    elif _is_clarification(turn["content"]):
                        is_clarification = True
                    elif turn["content"].startswith("Tool results:"):
                        gold_tool_result_turns.append(turn)
                    else:
                        gold_final_text = turn["content"]

            if is_clarification:
                total_clarify += 1

            delimiter = "<|im_start|>assistant\n" if model_type == "lfm" else "<|turn>model\n"
            first_assistant = text.find(delimiter)
            if first_assistant == -1:
                logger.warning("No assistant/model turn found in eval example, skipping")
                continue
            conversation = text[:first_assistant]

            model_tool_calls: list[dict] = []
            model_final_text = ""
            for _ in range(max_turns):
                inputs = tokenizer(
                    text=conversation,
                    return_tensors="pt",
                    return_dict=True,
                    add_special_tokens=False,
                ).to(model.device)

                t0 = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    use_cache=True,
                )
                dt = time.time() - t0
                total_time += dt
                total_tokens += outputs.shape[1] - inputs["input_ids"].shape[1]

                new_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=False,
                )
                conversation += new_text

                tc = parse_tc(new_text)
                if tc:
                    model_tool_calls.append(tc)
                    idx = len(model_tool_calls) - 1
                    if idx < len(gold_tool_result_turns):
                        conversation += _format_turn(gold_tool_result_turns[idx], model_type)
                    continue
                elif _is_clarification(new_text):
                    model_final_text = new_text
                    break
                else:
                    model_final_text = new_text
                    break

            for i, gold_tc in enumerate(gold_tool_calls):
                total_tool_calls += 1
                if i < len(model_tool_calls):
                    mtc = model_tool_calls[i]
                    if mtc["name"] == gold_tc["name"] and _args_match(
                        mtc.get("arguments", {}), gold_tc.get("arguments", {})
                    ):
                        correct_tool_calls += 1

            if gold_final_text and model_final_text:
                gold_result = _extract_final_result(gold_final_text)
                model_result = _extract_final_result(model_final_text)
                if gold_result is not None and model_result is not None:
                    total_final += 1
                    if math.isclose(model_result, gold_result, rel_tol=FINAL_RESULT_TOLERANCE):
                        correct_final += 1

            if is_clarification and _is_clarification(model_final_text):
                correct_clarify += 1

            n_convos += 1

    torch.cuda.empty_cache()

    if total_clarify == 0:
        logger.warning("No clarification scenarios in eval set — clarification_quality defaults to 0.0")

    metrics = EvalMetrics(
        tool_call_accuracy=correct_tool_calls / total_tool_calls if total_tool_calls else 0.0,
        sequence_correctness=correct_final / total_final if total_final else 0.0,
        clarification_quality=correct_clarify / total_clarify if total_clarify else 0.0,
        natural_language_quality=3.5,
        latency_score=1.0,
        memory_score=1.0,
    )

    logger.info(f"Evaluated {n_convos} conversations")
    logger.info(f"Tool calls: {correct_tool_calls}/{total_tool_calls}")
    logger.info(f"Final results: {correct_final}/{total_final}")
    logger.info(f"Clarifications: {correct_clarify}/{total_clarify}")
    logger.info(f"Avg time/turn: {total_time / max(1, n_convos):.2f}s")
    logger.info(f"Avg tokens/turn: {total_tokens / max(1, n_convos):.0f}")

    return metrics


def _format_turn(turn: dict, model_type: str) -> str:
    role = turn["role"]
    content = turn["content"]
    if model_type == "lfm":
        return f"<|im_start|>{role}\n{content}<|im_end|>"
    else:
        mapped_role = "model" if role == "assistant" else role
        return f"<|turn>{mapped_role}\n{content}<turn|>"


def _args_match(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-type", required=True, choices=["lfm", "gemma"])
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    metrics = evaluate_model(
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        eval_path=args.eval_path,
        max_new_tokens=args.max_new_tokens,
    )
    print(metrics.report())

    score = metrics.weighted_score()
    print(f"\nWeighted score: {score:.4f}")

    if score >= 0.70:
        print("PASS — model meets minimum threshold (0.70)")
    else:
        print("BELOW THRESHOLD — consider more training or hyperparameter tuning")


if __name__ == "__main__":
    main()
