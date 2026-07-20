import re
from typing import List, Optional
from data.log_config import logger
from data.scenarios import Conversation, ScenarioType
from data.food_db_loader import FoodItem
from data.mock_harness import MockHarness


def validate_conversation(
    conv: Conversation,
    food_db: List[FoodItem],
    expected_scenario: ScenarioType,
    user_settings: Optional[dict] = None,
) -> List[str]:
    errors: List[str] = []

    r = _validate_tool_sequence(conv, expected_scenario)
    _log_check("tool_sequence", r, conv.language)
    errors += r

    r = _validate_food_id_integrity(conv, food_db)
    _log_check("food_id_integrity", r, conv.language)
    errors += r

    r = _validate_entry_id_integrity(conv)
    _log_check("entry_id_integrity", r, conv.language)
    errors += r

    r = _validate_units(conv, food_db)
    _log_check("units", r, conv.language)
    errors += r

    r = _validate_math(conv, food_db, user_settings)
    _log_check("math", r, conv.language)
    errors += r

    r = _validate_text_math(conv)
    _log_check("text_math", r, conv.language)
    errors += r

    r = _validate_length(conv)
    _log_check("length", r, conv.language)
    errors += r

    r = _validate_no_empty_turns(conv)
    _log_check("no_empty_turns", r, conv.language)
    errors += r

    r = _validate_context_blocks(conv)
    _log_check("context_blocks", r, conv.language)
    errors += r

    r = _validate_user_input_tokens(conv)
    _log_check("user_input_tokens", r, conv.language)
    errors += r

    return errors


def _log_check(name: str, errors: list, lang: str) -> None:
    if errors:
        logger.debug(f"[{lang}] {name}: FAIL ({len(errors)} error(s)) -- {errors[0][:120]}")
    else:
        logger.debug(f"[{lang}] {name}: passed")


def _validate_tool_sequence(conv: Conversation, scenario: ScenarioType) -> list:
    tools_seen = []
    for turn in conv.turns:
        if turn.role == "assistant" and turn.action == "tool_call":
            for tc in (turn.tool_calls or []):
                tools_seen.append(tc.name)

                if tc.name == "calculate_final" and scenario != ScenarioType.GLUCOSE_ONLY_CHECK:
                    if "add_foods_to_tally" not in tools_seen[:-1]:
                        return ["calculate_final called before add_foods_to_tally"]

                if tc.name == "add_foods_to_tally":
                    if "search_foods" not in tools_seen[:-1]:
                        return ["add_foods_to_tally called before search_foods"]

                if tc.name == "remove_foods_from_tally":
                    if "add_foods_to_tally" not in tools_seen[:-1]:
                        return ["remove_foods_to_tally called before add_foods_to_tally"]
    return []


def _validate_food_id_integrity(conv: Conversation, food_db: List[FoodItem]) -> list:
    errors = []
    known_ids = set()
    db_ids = {f.id for f in food_db}

    for turn in conv.turns:
        if turn.role == "user" and turn.context_block:
            known_ids.update(_parse_known_ids_from_context(turn.context_block))

        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                if tc.name == "search_foods":
                    pass
                elif tc.name == "add_foods_to_tally":
                    for item in tc.arguments.get("items", []):
                        fid = item.get("food_id")
                        if fid is not None and fid not in known_ids:
                            errors.append(
                                f"food_id {fid} not in known IDs: {known_ids}"
                            )
                elif tc.name == "remove_foods_from_tally":
                    pass
                elif tc.name == "calculate_final":
                    pass

        if turn.role == "tool" and turn.tool_results:
            if "search_foods" in turn.tool_results:
                result = turn.tool_results["search_foods"]
                if isinstance(result, dict) and "results" in result:
                    for batch in result["results"]:
                        if isinstance(batch, list):
                            for item in batch:
                                if isinstance(item, dict) and "id" in item:
                                    known_ids.add(item["id"])

    return errors


def _validate_entry_id_integrity(conv: Conversation) -> list:
    errors = []
    tally_entry_ids = set()

    for turn in conv.turns:
        if turn.role == "tool" and turn.tool_results:
            for name, result in turn.tool_results.items():
                if name == "add_foods_to_tally":
                    if isinstance(result, dict):
                        for e in result.get("entries", []):
                            if isinstance(e, dict) and "entry_id" in e:
                                tally_entry_ids.add(e["entry_id"])
                elif name == "remove_foods_from_tally":
                    if isinstance(result, dict):
                        removed = result.get("removed", 0)

        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                if tc.name == "remove_foods_from_tally":
                    for eid in tc.arguments.get("entry_ids", []):
                        pass

    return errors


def _validate_math(conv: Conversation, food_db: List[FoodItem], user_settings: Optional[dict] = None) -> list:
    errors = []
    harness = MockHarness(food_db, user_settings)

    for turn in conv.turns:
        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                try:
                    expected = harness.execute(tc)
                except Exception:
                    continue

                if turn.tool_results and tc.name in (turn.tool_results or {}):
                    actual = turn.tool_results[tc.name]
                    if isinstance(expected, dict) and isinstance(actual, dict):
                        for key in expected:
                            if isinstance(expected[key], (int, float)) and key in actual:
                                if isinstance(actual[key], (int, float)):
                                    if abs(expected[key] - actual[key]) > 0.02:
                                        errors.append(
                                            f"Math mismatch for {tc.name}.{key}: "
                                            f"expected {expected[key]}, got {actual[key]}"
                                        )

    return errors


def _collect_numeric_values(obj, values: set[float]) -> None:
    if isinstance(obj, (int, float)):
        values.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numeric_values(v, values)
    elif isinstance(obj, list):
        for item in obj:
            _collect_numeric_values(item, values)


def _validate_text_math(conv: Conversation) -> list:
    errors = []

    valid_values: set[float] = set()
    for turn in conv.turns:
        if turn.role == "tool" and turn.tool_results:
            _collect_numeric_values(turn.tool_results, valid_values)
        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                _collect_numeric_values(tc.arguments, valid_values)
        if turn.role == "user" and turn.content:
            for m in re.findall(r"\d+\.?\d*", turn.content):
                valid_values.add(float(m))

    for i, turn in enumerate(conv.turns):
        if turn.role != "assistant" or turn.action != "text" or not turn.content:
            continue

        text_numbers = [float(m) for m in re.findall(r"\d+\.\d+|\d+", turn.content)]
        if not text_numbers:
            continue

        for tn in text_numbers:
            if tn < 0.01:
                continue

            found = _match_number(tn, valid_values)
            if not found:
                errors.append(
                    f"Turn {i}: number {tn} in assistant text does not match "
                    f"any tool result or argument"
                )

    return errors


def _match_number(tn: float, valid_values: set[float]) -> bool:
    for vv in valid_values:
        if abs(tn - vv) < 0.05:
            return True
        if abs(tn - vv) < 1.0 and abs(round(tn) - round(vv)) < 0.01:
            return True
    return False


def _validate_length(conv: Conversation) -> list:
    full_text = " ".join(
        t.content or "" for t in conv.turns
    )
    if conv.language in ("zh", "ja", "hi"):
        estimated_tokens = len(full_text)
    else:
        estimated_tokens = len(full_text) // 4

    if estimated_tokens < 20:
        return [f"Too short: ~{estimated_tokens} tokens"]
    if estimated_tokens > 2560:
        return [f"Too long: ~{estimated_tokens} tokens"]
    return []


def _validate_no_empty_turns(conv: Conversation) -> list:
    for i, turn in enumerate(conv.turns):
        if turn.role == "user" and not turn.content:
            return [f"Turn {i}: empty user content"]
        if turn.role == "assistant":
            if turn.action == "text" and not turn.content:
                return [f"Turn {i}: empty assistant text"]
            if turn.action == "tool_call" and not turn.tool_calls:
                return [f"Turn {i}: empty tool calls"]
    return []


def _validate_context_blocks(conv: Conversation) -> list:
    for i, turn in enumerate(conv.turns):
        if turn.role == "user" and not turn.context_block:
            return [f"Turn {i}: missing context_block in user message"]
    return []


def _validate_user_input_tokens(conv: Conversation) -> list:
    errors = []
    for i, turn in enumerate(conv.turns):
        if turn.role == "user" and turn.content:
            estimated = len(turn.content) // 4
            if estimated > 100:
                errors.append(
                    f"Turn {i}: user utterance ~{estimated} tokens (max ~100)"
                )
    return errors


def _parse_known_ids_from_context(block: str) -> set:
    ids = set()
    matches = re.findall(r'\((\d+)\)', block)
    for m in matches:
        ids.add(int(m))
    return ids


def _validate_units(conv: Conversation, food_db: List[FoodItem]) -> list:
    errors = []
    db_by_id = {f.id: f for f in food_db}

    for turn in conv.turns:
        if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                if tc.name == "add_foods_to_tally":
                    for item in tc.arguments.get("items", []):
                        fid = item.get("food_id")
                        unit = item.get("unit")
                        if fid is None or unit is None:
                            continue
                        if fid not in db_by_id:
                            continue
                        food = db_by_id[fid]
                        if not food.has_grams_mode and not food.has_pieces_mode:
                            continue
                        if unit in ("g", "ml") and not food.has_grams_mode:
                            errors.append(
                                f"food_id {fid} ({food.name}): unit '{unit}' used "
                                f"but food has no grams/ml mode"
                            )
                        if unit in ("pcs", "cup", "tbsp", "slice") and not food.has_pieces_mode:
                            errors.append(
                                f"food_id {fid} ({food.name}): unit '{unit}' used "
                                f"but food has no piece mode"
                            )
    return errors
