import asyncio
import json
import hashlib
import os
import argparse
import warnings
import collections
import time as _time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

from data.log_config import (
    logger, setup_logging, StepTimer, ProgressTracker,
    GlobalProgressTracker, log_conversation_result,
)
from data.food_db_loader import load_food_db, get_all_food_names_and_ids, get_all_food_names
from data.scenarios import (
    Conversation, ScenarioType, SCENARIO_WEIGHTS,
    LANGUAGE_NAMES, SCENARIO_INSTRUCTIONS,
)
import re
from data.mock_harness import MockHarness, format_user_settings, USER_SETTINGS_POOL
from data.validator import validate_conversation


def _format_number(value) -> str:
    """Format number for natural text display, accurate to 2nd decimal."""
    if isinstance(value, int):
        return str(value)
    f = float(value)
    if f == int(f):
        return str(int(f))
    formatted = f"{f:.2f}".rstrip("0").rstrip(".")
    return formatted


def _replace_placeholders(conv: Conversation) -> list[str]:
    """Replace {{placeholder}} tokens in assistant text with actual tool result values.
    Returns list of errors (unreplaced placeholders)."""
    placeholders = {}
    errors = []
    
    for i, turn in enumerate(conv.turns):
        if turn.role == "tool" and turn.tool_results:
            if "add_foods_to_tally" in turn.tool_results:
                result = turn.tool_results["add_foods_to_tally"]
                if isinstance(result, dict) and "entries" in result:
                    for j, entry in enumerate(result.get("entries", [])):
                        placeholders[f"carbs_{j+1}"] = entry.get("carbs")
                    placeholders["tally_total"] = result.get("tally_total")
            
            if "calculate_final" in turn.tool_results:
                result = turn.tool_results["calculate_final"]
                if isinstance(result, dict):
                    placeholders["total_carbs"] = result.get("tally_total")
                    placeholders["food_insulin"] = result.get("food_insulin")
                    placeholders["glucose_correction"] = result.get("glucose_correction")
                    placeholders["final_dose"] = result.get("final_result")
                    placeholders["meal_divider"] = result.get("meal_divider")
            
            if "remove_foods_from_tally" in turn.tool_results:
                result = turn.tool_results["remove_foods_from_tally"]
                if isinstance(result, dict):
                    placeholders["remaining_total"] = result.get("tally_total")
                    placeholders["tally_total"] = result.get("tally_total")
        
        if turn.role == "assistant" and turn.action == "text" and turn.content:
            def replacer(match):
                key = match.group(1)
                if key in placeholders and placeholders[key] is not None:
                    return _format_number(placeholders[key])
                return match.group(0)
            
            turn.content = re.sub(r"\{\{(\w+)\}\}", replacer, turn.content)
            
            remaining = re.findall(r"\{\{(\w+)\}\}", turn.content)
            for p in remaining:
                errors.append(f"Turn {i}: unreplaced placeholder {{{{{p}}}}}")
                
    return errors


GENERATOR_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_generator.txt"
TOOLS_SCHEMA_PATH = Path(__file__).parent / "schemas" / "tools.json"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_client = None

def _get_client():
    global _client
    if _client is None:
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Create a key at https://aistudio.google.com/apikey "
                "and add it to your .env file (see .env.example)."
            )
        langfuse = get_client()
        if not langfuse.auth_check():
            logger.warning("Langfuse auth failed - traces will not be recorded. Check your keys.")
        GoogleGenAIInstrumentor().instrument()
        _client = genai.Client()
    return _client


class RateLimiter:
    def __init__(self, rpm: int = 1000, tpm: int = 1_000_000):
        self._rpm_limit = int(rpm * 0.8)
        self._tpm_limit = int(tpm * 0.8)
        self._records: collections.deque = collections.deque()
        self._lock = asyncio.Lock()

    async def wait_if_needed(self, estimated_tokens: int = 5000) -> None:
        while True:
            async with self._lock:
                now = _time.time()
                cutoff = now - 60
                while self._records and self._records[0][0] < cutoff:
                    self._records.popleft()

                rpm_ok = len(self._records) < self._rpm_limit
                tpm_ok = sum(r[1] for r in self._records) + estimated_tokens < self._tpm_limit

                if rpm_ok and tpm_ok:
                    self._records.append((now, estimated_tokens))
                    return

                sleep_for = self._records[0][0] - cutoff + 0.5

            await asyncio.sleep(max(0, sleep_for))


_rate_limiter = RateLimiter()

with open(TOOLS_SCHEMA_PATH, encoding="utf-8") as f:
    TOOL_SCHEMAS_JSON = json.dumps(json.load(f)["tools"], indent=2, ensure_ascii=False)


def _build_system_prompt(
    lang: str,
    scenario: ScenarioType,
    foods_sample_str: str,
    conversation_meta: str = "",
    user_settings_str: str = "",
) -> str:
    base = GENERATOR_PROMPT_PATH.read_text(encoding="utf-8")

    lang_name = LANGUAGE_NAMES.get(lang, lang)
    instructions = SCENARIO_INSTRUCTIONS.get(scenario, "")
    detail = instructions

    replacements = {
        "%{language_name}": lang_name,
        "%{language_code}": lang,
        "%{user_settings}": user_settings_str,
        "%{tool_schemas}": TOOL_SCHEMAS_JSON,
        "%{food_db_sample}": foods_sample_str,
        "%{scenario_instructions}": instructions,
        "%{scenario_type}": scenario.value,
        "%{scenario_detail}": detail,
        "%{conversation_meta}": conversation_meta,
    }

    for placeholder, value in replacements.items():
        base = base.replace(placeholder, value)

    return base


def _sanitize_raw_turns(turns: list) -> list:
    for turn in turns:
        if turn.get("role") == "assistant" and turn.get("action") == "tool_call":
            if "tool_results" not in turn:
                turn["tool_results"] = {}
        if turn.get("role") == "tool":
            if not isinstance(turn.get("tool_results"), dict):
                turn["tool_results"] = {}
    return turns


def _copy_results_to_tool_turns(turns: list) -> None:
    for i, turn in enumerate(turns):
        if turn.role == "assistant" and turn.tool_results:
            for j in range(i + 1, len(turns)):
                if turns[j].role == "tool":
                    if turns[j].tool_results is None:
                        turns[j].tool_results = {}
                    for k, v in turn.tool_results.items():
                        turns[j].tool_results[k] = v
                    break


def _build_retry_message(error: Exception, tool_results: str | None = None) -> str:
    lines = ["Your previous response had errors. Please fix them and regenerate:\n"]

    if isinstance(error, json.JSONDecodeError):
        lines.append("- Output was NOT valid JSON. Generate ONLY valid JSON, nothing else.")
        return "\n".join(lines)

    msg = str(error)

    if "Validation errors:" in msg:
        errs = msg.replace("Validation errors: ", "")
        if len(errs) > 500:
            errs = errs[:500] + "..."
        lines.append(f"- {errs}")
    elif error.__class__.__name__ == "ValidationError":
        err_list = error.errors()
        for e in err_list[:3]:
            loc = " -> ".join(str(x) for x in e.get("loc", []))
            msg_text = e.get("msg", "invalid")
            lines.append(f"- {loc}: {msg_text}")
        if len(err_list) > 3:
            lines.append(f"- ... and {len(err_list) - 3} more structural issues")
    else:
        excerpt = msg[:300]
        if excerpt:
            lines.append(f"- {excerpt}")
        if len(msg) > 300:
            lines.append("- ... (truncated)")

    if tool_results:
        lines.append("\nTool results from your previous attempt (reference these numbers when writing assistant text):")
        lines.append(tool_results)

    return "\n".join(lines)


def _format_tool_results_for_retry(turns: list) -> str:
    parts = []
    for i, turn in enumerate(turns):
        if turn.role == "tool" and turn.tool_results:
            results_str = json.dumps(turn.tool_results, ensure_ascii=False, indent=2)
            if len(results_str) > 400:
                results_str = results_str[:400] + "..."
            parts.append(f"Turn {i}: {results_str}")
    return "\n".join(parts) if parts else ""


class FoodUsageTracker:
    """Per-food-ID usage counter. Injects AVOID/PREFER guidance into the prompt."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def record(self, food_ids: set[str]) -> None:
        for fid in food_ids:
            self._counts[fid] = self._counts.get(fid, 0) + 1

    def get_stats_prompt(
        self,
        food_db: list,
        sampled_ids: list[str],
        max_overused: int = 8,
        max_underused: int = 12,
    ) -> str:
        total = sum(self._counts.values()) or 1
        threshold = max(1, int(total * 0.04))
        id_to_name = {f.id: f.name for f in food_db}

        overused = [
            (fid, self._counts.get(fid, 0), id_to_name.get(fid, fid))
            for fid in sampled_ids
            if self._counts.get(fid, 0) >= threshold
        ]
        overused.sort(key=lambda x: -x[1])

        underused = [
            (fid, self._counts.get(fid, 0), id_to_name.get(fid, fid))
            for fid in sampled_ids
            if self._counts.get(fid, 0) <= 1
        ]

        if not (len(overused) >= 2 or len(underused) >= 2):
            return ""

        parts = ["## FOOD USAGE TRACKER"]
        if len(overused) >= 2:
            parts.append("FOODS TO AVOID (already used frequently in this batch):")
            for fid, cnt, name in overused[:max_overused]:
                parts.append(f"  AVOID {name} ({fid}) - used {cnt}x already")
        if len(underused) >= 2:
            if len(overused) >= 2:
                parts.append("")
            parts.append("FOODS TO PREFER (rarely or never used — pick these):")
            for fid, cnt, name in underused[:max_underused]:
                parts.append(f"  PREFER {name} ({fid}) - used only {cnt}x")
        return "\n".join(parts)


class ConversationMetaTracker:
    """Tracks persona rotation, short/long conversation bias, and user settings."""

    _PERSONAS = ["terse", "verbose", "confused", "slang", "formal"]
    _PERSONA_DESCRIPTIONS = {
        "terse": "short abbreviations, minimal words ('add banana 1 pcs thx')",
        "verbose": "polite but keeps it brief, adds a bit of context without rambling",
        "confused": "hesitant, uncertain, 'wait how many carbs?', 'is that right?'",
        "slang": "casual abbreviations, emojis (pls, u, idk, btw, np)",
        "formal": "proper grammar, polite, precise ('Please add one banana, thank you')",
    }
    _SHORT_THRESHOLD = 5

    def __init__(self) -> None:
        self._total = 0
        self._short_count = 0
        self._long_count = 0

    def record(self, turn_count: int) -> None:
        self._total += 1
        if turn_count <= self._SHORT_THRESHOLD:
            self._short_count += 1
        else:
            self._long_count += 1

    def get_user_settings(self) -> dict:
        idx = self._total % len(USER_SETTINGS_POOL)
        return USER_SETTINGS_POOL[idx]

    def get_user_settings_idx(self) -> int:
        return self._total % len(USER_SETTINGS_POOL)

    def format_user_settings_prompt(self) -> str:
        return format_user_settings(self.get_user_settings())

    def get_stats_prompt(self) -> str:
        persona_idx = self._total % len(self._PERSONAS)
        persona = self._PERSONAS[persona_idx]
        desc = self._PERSONA_DESCRIPTIONS[persona]

        parts = ["## CONVERSATION META TRACKER"]
        parts.append(f"Persona for this conversation: {persona.upper()} USER")
        parts.append(f"  ({desc})")

        if self._short_count + self._long_count >= 3:
            ratio = self._short_count / max(self._short_count + self._long_count, 1)
            if ratio > 0.6:
                parts.append(
                    f"Length bias: PREFER LONG conversation "
                    f"(short:{self._short_count}, long:{self._long_count} — too many short)"
                )
            elif ratio < 0.4:
                parts.append(
                    f"Length bias: PREFER SHORT conversation "
                    f"(short:{self._short_count}, long:{self._long_count} — too many long)"
                )
            else:
                parts.append(
                    f"Length: balanced (short:{self._short_count}, long:{self._long_count}) — no bias"
                )

        return "\n".join(parts)


def _extract_food_ids_from_raw(conv_data: dict) -> set[str]:
    ids: set[str] = set()
    for turn in conv_data.get("turns", []):
        if turn.get("role") != "assistant":
            continue
        for tc in turn.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            if tc.get("name") == "search_foods":
                for q in args.get("queries") or []:
                    ids.add(q)
            elif tc.get("name") == "add_foods_to_tally":
                fid = args.get("food_id")
                if fid:
                    ids.add(fid)
    return ids


def _extract_food_ids_from_conversation(conv: Conversation) -> set[str]:
    ids: set[str] = set()
    for turn in conv.turns:
        if turn.role != "assistant" or not turn.tool_calls:
            continue
        for tc in turn.tool_calls:
            args = tc.arguments or {}
            if tc.name == "search_foods":
                for q in args.get("queries") or []:
                    ids.add(q)
            elif tc.name == "add_foods_to_tally":
                fid = args.get("food_id")
                if fid:
                    ids.add(fid)
    return ids


def _replay_existing(
    file_path: Path,
    food_tracker: FoodUsageTracker,
    meta_tracker: ConversationMetaTracker,
) -> None:
    if not file_path.exists():
        return
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            try:
                conv_data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            food_ids = _extract_food_ids_from_raw(conv_data)
            food_tracker.record(food_ids)
            turn_count = len(conv_data.get("turns", []))
            meta_tracker.record(turn_count)


async def generate_conversation(
    lang: str,
    scenario: ScenarioType,
    foods_sample_str: str,
    semaphore: asyncio.Semaphore,
    global_tracker: Optional[GlobalProgressTracker] = None,
    conversation_meta: str = "",
    user_settings: Optional[dict] = None,
    user_settings_str: str = "",
) -> Conversation:
    system_prompt = _build_system_prompt(lang, scenario, foods_sample_str, conversation_meta, user_settings_str)
    t_start = _time.perf_counter()

    config_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.8,
        "top_p": 0.95,
        "max_output_tokens": 4096,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }

    async with semaphore:
        last_error: Optional[Exception] = None
        last_tool_results_str = ""
        for attempt in range(3):
            try:
                if attempt == 0:
                    prompt = system_prompt
                else:
                    prompt = system_prompt + "\n\n" + _build_retry_message(last_error, last_tool_results_str)

                await _rate_limiter.wait_if_needed()

                with propagate_attributes(
                    tags=["mininio", "data-generation"],
                    metadata={
                        "project": "mininio-ai-finetuning",
                        "language": lang,
                        "scenario": scenario.value
                    }
                ):
                    response = _get_client().models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )

                if global_tracker is not None and hasattr(response, 'usage_metadata'):
                    um = response.usage_metadata
                    if um:
                        global_tracker.add_tokens(
                            getattr(um, 'prompt_token_count', 0) or 0,
                            getattr(um, 'candidates_token_count', 0) or 0,
                        )

                raw = json.loads(response.text)
                raw["turns"] = _sanitize_raw_turns(raw.get("turns", []))

                conv_dict = {
                    "scenario_type": scenario.value,
                    "language": lang,
                    "turns": raw["turns"],
                }
                conv = Conversation.model_validate(conv_dict)

                food_db = load_food_db(lang)
                harness = MockHarness(food_db, user_settings)

                for turn in conv.turns:
                    if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
                        for tc in turn.tool_calls:
                            try:
                                result = harness.execute(tc)
                            except Exception:
                                result = {"error": "execution failed"}

                            if turn.tool_results is None:
                                turn.tool_results = {}
                            turn.tool_results[tc.name] = result

                _copy_results_to_tool_turns(conv.turns)
                placeholder_errors = _replace_placeholders(conv)
                last_tool_results_str = _format_tool_results_for_retry(conv.turns)

                errors = validate_conversation(conv, food_db, scenario, user_settings)
                errors.extend(placeholder_errors)
                if errors:
                    raise ValueError(f"Validation errors: {'; '.join(errors)}")

                elapsed = _time.perf_counter() - t_start
                log_conversation_result(lang, scenario.value, True, elapsed, attempt + 1)
                return conv

            except Exception as e:
                last_error = e
                if attempt < 2:
                    delay = 2 ** attempt
                    logger.warning(
                        f"[{lang}] {scenario.value} -- retry {attempt + 1}/3: {e} "
                        f"(backoff {delay}s)"
                    )
                    await asyncio.sleep(delay)
                else:
                    elapsed = _time.perf_counter() - t_start
                    log_conversation_result(lang, scenario.value, False, elapsed, 3, str(e))
                    raise


def _hash_user_utterances(conv: Conversation) -> str:
    text = "|".join(
        t.content or "" for t in conv.turns if t.role == "user"
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


import collections

def _generate_work_distribution(count: int, existing_counts: dict = None) -> list:
    if existing_counts is None:
        existing_counts = {}
        
    distribution = []
    remaining_target = count
    scenarios = list(SCENARIO_WEIGHTS.keys())
    
    for scenario in scenarios[:-1]:
        target = round(count * SCENARIO_WEIGHTS[scenario])
        existing = existing_counts.get(scenario.value, 0)
        n_to_generate = max(0, target - existing)
        distribution.append((scenario, n_to_generate))
        remaining_target -= target
        
    last_scenario = scenarios[-1]
    existing_last = existing_counts.get(last_scenario.value, 0)
    n_to_generate = max(0, remaining_target - existing_last)
    distribution.append((last_scenario, n_to_generate))
    
    return distribution


async def generate_language_dataset(
    lang: str,
    count: int = 800,
    max_concurrent: int = 10,
    resume: bool = True,
    global_tracker: Optional[GlobalProgressTracker] = None,
):
    output_path = Path(f"data/output/raw/{lang}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    existing_count = 0
    existing_counts_by_scenario = collections.defaultdict(int)
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    conv_data = json.loads(line)
                    scenario_type = conv_data["scenario_type"]
                    key = (scenario_type, _hash_user_utterances_raw(conv_data))
                    completed.add(key)
                    existing_counts_by_scenario[scenario_type] += 1
                    existing_count += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    food_db = load_food_db(lang)
    food_tracker = FoodUsageTracker()
    meta_tracker = ConversationMetaTracker()
    semaphore = asyncio.Semaphore(max_concurrent)

    work_items = _generate_work_distribution(count, existing_counts_by_scenario)
    generated = 0
    failed = 0
    skipped = 0

    _replay_existing(output_path, food_tracker, meta_tracker)

    tracker = ProgressTracker(count, f"[{lang}]")
    logger.info(f"[{lang}] Starting ({count} conversations, {len(work_items)} scenarios)")
    if existing_count:
        logger.info(f"[{lang}] Resumed {existing_count} existing")
        tracker.bulk_resume(existing_count)

    for scenario, n in work_items:
        logger.debug(f"[{lang}] Scenario {scenario.value}: {n} items to generate")
        for _ in range(n):
            if generated + existing_count >= count:
                break

            sampled_str, sampled_ids = get_all_food_names_and_ids(food_db, max_items=20)
            food_stats = food_tracker.get_stats_prompt(food_db, sampled_ids)
            augmented_sample = (food_stats + "\n\n" + sampled_str) if food_stats else sampled_str
            conv_meta = meta_tracker.get_stats_prompt()
            user_settings = meta_tracker.get_user_settings()
            user_settings_idx = meta_tracker.get_user_settings_idx()
            user_settings_str = meta_tracker.format_user_settings_prompt()

            try:
                conv = await generate_conversation(
                    lang, scenario, augmented_sample, semaphore, global_tracker,
                    conv_meta, user_settings, user_settings_str,
                )
            except Exception:
                failed += 1
                continue

            key = (conv.scenario_type, _hash_user_utterances(conv))
            if key in completed:
                skipped += 1
                logger.debug(f"[{lang}] Skipped duplicate: {scenario.value}")
                continue

            food_tracker.record(_extract_food_ids_from_conversation(conv))
            meta_tracker.record(len(conv.turns))

            conv.user_settings_idx = user_settings_idx

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(conv.model_dump_json() + "\n")
            completed.add(key)
            generated += 1
            tracker.update()
            if global_tracker:
                global_tracker.update()

    tracker.finish()
    if failed or skipped:
        logger.info(f"[{lang}] Failed: {failed}, Skipped (dupes): {skipped}")
    return generated


def _hash_user_utterances_raw(conv_data: dict) -> str:
    text = "|".join(
        t.get("content") or "" for t in conv_data.get("turns", []) if t.get("role") == "user"
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def main(languages, count_per_lang, dry_run, validate_only, max_concurrent,
               verbose, log_file, debug_harness):
    log_path = None
    if log_file:
        if log_file == "auto":
            ts = _time.strftime("%Y%m%d_%H%M%S")
            log_path = Path(f"data/output/logs/generation_{ts}.log")
        else:
            log_path = Path(log_file)
    elif not dry_run and not validate_only:
        logger.info("Tip: use --log-file to save full logs to disk")

    setup_logging(verbose=verbose, log_file=log_path, debug_harness=debug_harness)

    if validate_only:
        logger.info("Validating existing raw data...")
        for lang in languages:
            raw_path = Path(f"data/output/raw/{lang}.jsonl")
            if not raw_path.exists():
                logger.warning(f"[{lang}] No raw data found at {raw_path}")
                continue
            food_db = load_food_db(lang)
            errors_total = 0
            valid = 0
            with open(raw_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    conv = Conversation.model_validate_json(line)
                    errors = validate_conversation(conv, food_db, ScenarioType(conv.scenario_type))
                    if errors:
                        errors_total += 1
                        if errors_total <= 5:
                            logger.warning(f"[{lang}] Validation errors in {conv.scenario_type}: {errors}")
                    else:
                        valid += 1
            logger.info(f"[{lang}]: {valid} valid, {errors_total} with errors")
        return

    if dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"Would generate {count_per_lang} conversations per language")
        logger.info(f"Languages: {', '.join(languages)} ({len(languages)})")
        logger.info(f"Total: {len(languages) * count_per_lang} conversations")
        for scenario, weight in SCENARIO_WEIGHTS.items():
            n = round(count_per_lang * weight)
            logger.info(f"  {scenario.value}: {n} per language")
        return

    total_items = len(languages) * count_per_lang
    global_tracker = GlobalProgressTracker(total_items, len(languages))

    logger.info("=== DATA GENERATION STARTED ===")
    logger.info(f"Model: {GEMINI_MODEL} | Concurrent: {max_concurrent}")
    logger.info(f"Languages: {len(languages)} x {count_per_lang} = {total_items} total")
    if log_path:
        logger.info(f"Log file: {log_path}")

    with StepTimer("Full generation run"):
        results = []
        for lang in languages:
            result = await generate_language_dataset(lang, count_per_lang, max_concurrent, global_tracker)
            results.append(result)

    total_gen = sum(results)
    logger.success("=== GENERATION COMPLETE ===")
    logger.success(f"Generated: {total_gen}/{total_items} conversations")
    logger.success(f"Elapsed: {global_tracker.elapsed_str}")
    logger.success(f"Avg rate: {global_tracker.rate_per_hour:.1f}/h")
    if global_tracker.total_cost > 0:
        logger.success(f"Est. cost: ${global_tracker.total_cost:.2f}")

    try:
        get_client().flush()
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="all")
    parser.add_argument("--count-per-lang", type=int, default=800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-file", nargs="?", const="auto", default=None)
    parser.add_argument("--debug-harness", action="store_true")
    args = parser.parse_args()

    languages = (
        ["en", "el", "fr", "es", "hi", "it", "pt", "zh", "de", "ja"]
        if args.languages == "all"
        else args.languages.split(",")
    )
    asyncio.run(main(
        languages, args.count_per_lang, args.dry_run, args.validate_only,
        args.max_concurrent, args.verbose, args.log_file, args.debug_harness,
    ))
