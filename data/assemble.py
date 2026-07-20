import json
import random
from pathlib import Path
from collections import defaultdict

from data.log_config import logger, StepTimer, setup_logging
from data.scenarios import Conversation
from data.formatters import format_lfm, format_gemma

SEED = 42
TRAIN_RATIO = 0.9


def assemble_dataset(raw_dir: str = "data/output/raw"):
    logger.info("=== ASSEMBLING DATASETS ===")

    raw_path = Path(raw_dir)
    raw_conversations = []
    broken = 0

    with StepTimer("Load raw conversations"):
        files = sorted(raw_path.glob("*.jsonl"))
        for lang_file in files:
            with open(lang_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        conv = Conversation.model_validate_json(line)
                        raw_conversations.append(conv)
                    except Exception as e:
                        broken += 1
                        logger.warning(f"Skipping invalid conversation: {e}")

    if broken:
        logger.warning(f"Skipped {broken} invalid conversations")
    logger.info(f"Loaded {len(raw_conversations)} conversations from {len(files)} language files")

    lfm_system = Path("data/prompts/system_lfm.txt").read_text(encoding="utf-8")
    gemma_system = Path("data/prompts/system_gemma.txt").read_text(encoding="utf-8")

    with StepTimer("Stratified split"):
        random.seed(SEED)
        groups = defaultdict(list)
        for conv in raw_conversations:
            key = (conv.language, conv.scenario_type)
            groups[key].append(conv)

        train_convs = []
        eval_convs = []
        for key, convs in groups.items():
            random.shuffle(convs)
            split_idx = int(len(convs) * TRAIN_RATIO)
            train_convs.extend(convs[:split_idx])
            eval_convs.extend(convs[split_idx:])

    total = len(train_convs) + len(eval_convs)
    logger.info(
        f"Split: {len(train_convs)} train, {len(eval_convs)} eval "
        f"({len(train_convs) / max(1, total) * 100:.0f}/"
        f"{len(eval_convs) / max(1, total) * 100:.0f})"
    )
    logger.info(f"Stratified across {len(groups)} (language, scenario) groups")

    with StepTimer("Format LFM2.5 ChatML"):
        _save_formatted(train_convs, eval_convs, lfm_system,
                        "data/output/lfm", format_lfm)

    with StepTimer("Format Gemma 4 Unsloth"):
        _save_formatted(train_convs, eval_convs, gemma_system,
                        "data/output/gemma", format_gemma)

    logger.success("Assembly complete.")


def _save_formatted(train, eval_ds, system_prompt, output_dir, formatter):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, convs in [("train", train), ("eval", eval_ds)]:
        path = out / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for conv in convs:
                text = formatter(conv, system_prompt)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        logger.info(f"  Saved {path} ({len(convs)} examples)")


if __name__ == "__main__":
    setup_logging()
    assemble_dataset()
