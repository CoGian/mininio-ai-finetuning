import os

import pytest

from data.scenarios import Conversation, ScenarioType, Turn
from data.mock_harness import DEFAULT_SETTINGS, format_user_settings
from data.generate import (
    _build_system_prompt,
    _generate_work_distribution,
    _hash_user_utterances,
    _hash_user_utterances_raw,
    GEMINI_MODEL,
    _get_client,
)


class TestBuildSystemPrompt:
    def test_no_placeholders_left(self):
        result = _build_system_prompt("en", ScenarioType.SIMPLE_SINGLE_FOOD, "sample foods")
        assert "%{" not in result

    def test_settings_substituted(self):
        settings_str = format_user_settings(DEFAULT_SETTINGS)
        result = _build_system_prompt("en", ScenarioType.SIMPLE_SINGLE_FOOD, "sample foods", user_settings_str=settings_str)
        assert "130" in result
        assert "40" in result
        assert "14" in result

    def test_language_substituted(self):
        result = _build_system_prompt("el", ScenarioType.SIMPLE_SINGLE_FOOD, "sample foods")
        assert "Greek" in result
        assert "(el)" in result

    def test_food_sample_in_output(self):
        result = _build_system_prompt("en", ScenarioType.SIMPLE_SINGLE_FOOD, "FOOD_SAMPLE_HERE")
        assert "FOOD_SAMPLE_HERE" in result

    def test_scenario_type_in_output(self):
        result = _build_system_prompt("en", ScenarioType.GLUCOSE_ONLY_CHECK, "sample foods")
        assert "GLUCOSE_ONLY_CHECK" in result


class TestGenerateWorkDistribution:
    def test_sums_to_count(self):
        dist = _generate_work_distribution(100)
        total = sum(n for _, n in dist)
        assert total == 100

    def test_single_item(self):
        dist = _generate_work_distribution(1)
        assert len(dist) == len(ScenarioType)
        total = sum(n for _, n in dist)
        assert total == 1

    def test_large_count(self):
        dist = _generate_work_distribution(10000)
        total = sum(n for _, n in dist)
        assert total == 10000

    def test_all_scenarios_present(self):
        dist = _generate_work_distribution(100)
        scenarios = {s for s, _ in dist}
        assert scenarios == set(ScenarioType)


class TestHashUserUtterances:
    def test_deterministic(self):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content="hello", context_block="[CURRENT TALLY: empty]"),
                Turn(role="assistant", action="text", content="hi"),
                Turn(role="user", content="world", context_block="[CURRENT TALLY: empty]"),
            ],
        )
        h1 = _hash_user_utterances(conv)
        h2 = _hash_user_utterances(conv)
        assert h1 == h2

    def test_different_yields_different_hash(self):
        conv1 = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content="hello", context_block="[CURRENT TALLY: empty]"),
            ],
        )
        conv2 = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content="goodbye", context_block="[CURRENT TALLY: empty]"),
            ],
        )
        assert _hash_user_utterances(conv1) != _hash_user_utterances(conv2)

    def test_no_user_turns(self):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="assistant", action="text", content="hi"),
            ],
        )
        result = _hash_user_utterances(conv)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_hash_raw_matches(self):
        conv = Conversation(
            scenario_type="SIMPLE_SINGLE_FOOD",
            language="en",
            turns=[
                Turn(role="user", content="hello world", context_block="[CURRENT TALLY: empty]"),
            ],
        )
        h1 = _hash_user_utterances(conv)
        raw_data = conv.model_dump()
        h2 = _hash_user_utterances_raw(raw_data)
        assert h1 == h2


class TestGeminiModelEnv:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
        import importlib
        import data.generate
        importlib.reload(data.generate)
        assert data.generate.GEMINI_MODEL == "gemini-2.5-pro"

    def test_default(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        import importlib
        import data.generate
        importlib.reload(data.generate)
        assert data.generate.GEMINI_MODEL == "gemini-2.5-flash"


class TestGetClientNoApiKey:
    def test_raises_helpful_error(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        import data.generate
        data.generate._client = None

        with pytest.raises(RuntimeError) as exc_info:
            data.generate._get_client()
        assert "aistudio" in str(exc_info.value).lower()
        assert "GOOGLE_API_KEY" in str(exc_info.value)
