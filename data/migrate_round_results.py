import json
from pathlib import Path
from typing import Optional

from data.food_db_loader import load_food_db
from data.log_config import logger, setup_logging
from data.mock_harness import USER_SETTINGS_POOL, MockHarness
from data.scenarios import Conversation

RAW_DIR = Path("data/output/raw")


def remigrate_results(raw_dir: str | Path = RAW_DIR) -> None:
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.jsonl"))

    for lang_file in files:
        lang = lang_file.stem
        logger.info(f"Processing {lang_file.name}...")
        food_db = load_food_db(lang)
        updated_lines: list[str] = []

        with open(lang_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                conv = Conversation.model_validate_json(line)

                idx = conv.user_settings_idx if conv.user_settings_idx is not None else 0
                settings = USER_SETTINGS_POOL[idx]
                harness = MockHarness(food_db, settings=settings)

                for turn in conv.turns:
                    if turn.role == "assistant" and turn.action == "tool_call" and turn.tool_calls:
                        if turn.tool_results is None:
                            turn.tool_results = {}
                        for tc in turn.tool_calls:
                            try:
                                result = harness.execute(tc)
                            except Exception:
                                result = {"error": "execution failed"}
                            turn.tool_results[tc.name] = result

                _copy_results_to_tool_turns(conv.turns)
                updated_lines.append(
                    conv.model_dump_json(exclude_none=True) + "\n",
                )

        with open(lang_file, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        logger.info(f"  Done ({len(updated_lines)} conversations)")

    logger.success("Migration complete.")


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


if __name__ == "__main__":
    setup_logging()
    remigrate_results()
