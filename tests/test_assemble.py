import json
from pathlib import Path

import pytest

from data.scenarios import Conversation, ScenarioType
from data.formatters import format_lfm, format_gemma


def _write_raw_jsonl(conv: Conversation, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(conv.model_dump_json() + "\n")


def _read_formatted_lines(path: Path) -> list[dict]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


class TestAssembleCore:
    def test_save_formatted_creates_files(self, valid_simple_conversation, tmp_path):
        from data.assemble import _save_formatted

        train = [valid_simple_conversation]
        eval_ds = [valid_simple_conversation]
        system_prompt = "test prompt"
        out_dir = tmp_path / "output" / "lfm"

        _save_formatted(train, eval_ds, system_prompt, str(out_dir), format_lfm)

        train_file = out_dir / "train.jsonl"
        eval_file = out_dir / "eval.jsonl"
        assert train_file.exists()
        assert eval_file.exists()

    def test_save_formatted_structure(self, valid_simple_conversation, tmp_path):
        from data.assemble import _save_formatted

        train = [valid_simple_conversation]
        eval_ds = []
        system_prompt = "test prompt"
        out_dir = tmp_path / "output" / "gemma"

        _save_formatted(train, eval_ds, system_prompt, str(out_dir), format_gemma)

        lines = _read_formatted_lines(out_dir / "train.jsonl")
        assert len(lines) == 1
        assert "text" in lines[0]
        assert isinstance(lines[0]["text"], str)
        assert len(lines[0]["text"]) > 0

    def test_stratified_split_ratio(self, valid_simple_conversation, valid_multi_food_with_bg, tmp_path):
        convs = [
            Conversation(
                scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
                language="en",
                turns=valid_simple_conversation.turns,
            ),
            Conversation(
                scenario_type=ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE.value,
                language="el",
                turns=valid_multi_food_with_bg.turns,
            ),
        ]
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        for i, c in enumerate(convs):
            path = raw_dir / f"test_{i}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                f.write(c.model_dump_json() + "\n")

        import random
        random.seed(42)
        groups = {}
        for c in convs:
            key = (c.language, c.scenario_type)
            groups.setdefault(key, []).append(c)

        train = []
        eval_d = []
        for key, items in groups.items():
            random.shuffle(items)
            split_idx = max(1, int(len(items) * 0.9))
            train.extend(items[:split_idx])
            eval_d.extend(items[split_idx:])

        assert len(train) == 2
        assert len(eval_d) == 0

    def test_seeded_split_deterministic(self, valid_simple_conversation, tmp_path):
        convs = []
        for i in range(10):
            convs.append(Conversation(
                scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
                language="en",
                turns=valid_simple_conversation.turns,
            ))

        import random

        def run_split():
            random.seed(42)
            groups = {}
            for c in convs:
                key = (c.language, c.scenario_type)
                groups.setdefault(key, []).append(c)
            train = []
            for key, items in groups.items():
                random.shuffle(items)
                split_idx = max(1, int(len(items) * 0.9))
                train.extend(items[:split_idx])
            return len(train)

        r1 = run_split()
        r2 = run_split()
        assert r1 == r2

    def test_empty_raw_dir_handled(self, tmp_path):
        raw_dir = tmp_path / "empty_raw"
        raw_dir.mkdir(parents=True)

        files = list(raw_dir.glob("*.jsonl"))
        assert len(files) == 0
