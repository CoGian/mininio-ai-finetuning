import json
from pathlib import Path

from data.log_config import logger, setup_logging, StepTimer
from data.mock_harness import USER_SETTINGS_POOL

N_POOL = len(USER_SETTINGS_POOL)
SRC_DIR = Path("data/output/raw")
DST_DIR = Path("data/output/raw_migrated")


def migrate_raw_files() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    with StepTimer("Migrate raw conversations"):
        for src_path in sorted(SRC_DIR.glob("*.jsonl")):
            dst_path = DST_DIR / src_path.name
            valid = 0
            skipped_empty = 0

            with open(dst_path, "w", encoding="utf-8") as out_f:
                line_no = -1
                with open(src_path, encoding="utf-8") as in_f:
                    for raw_line in in_f:
                        stripped = raw_line.strip()
                        if not stripped:
                            skipped_empty += 1
                            continue
                        line_no += 1
                        idx = line_no % N_POOL

                        try:
                            conv = json.loads(stripped)
                        except json.JSONDecodeError:
                            line_no -= 1
                            logger.warning(f"{src_path.name}:{line_no + 2} — invalid JSON, skipped")
                            continue

                        conv["user_settings_idx"] = idx
                        out_f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                        valid += 1

            logger.info(
                f"  {src_path.name} → {dst_path.name}: "
                f"{valid} conversations (skipped {skipped_empty} empty lines)"
            )

    logger.success(f"Migration complete. Output: {DST_DIR}")


if __name__ == "__main__":
    setup_logging()
    migrate_raw_files()
