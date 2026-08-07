import pytest

from data.food_db_loader import FoodItem
from data.mock_harness import MockHarness, DEFAULT_SETTINGS, _compute_carbs, _serialize_food


class TestComputeCarbs:
    def test_gram_mode(self, sample_gram_food: FoodItem):
        result = _compute_carbs(sample_gram_food, 100.0, "g")
        assert round(result, 2) == 20.0

    def test_gram_mode_exact(self, sample_liquid_food: FoodItem):
        result = _compute_carbs(sample_liquid_food, 400.0, "ml")
        assert round(result, 2) == 20.0

    def test_piece_mode(self, sample_dual_food: FoodItem):
        result = _compute_carbs(sample_dual_food, 2.0, "pcs")
        assert round(result, 2) == 2.0

    def test_slice_mode(self, sample_piece_food: FoodItem):
        result = _compute_carbs(sample_piece_food, 3.0, "slice")
        assert round(result, 2) == 45.0

    def test_tbsp_mode(self, sample_tbsp_food: FoodItem):
        result = _compute_carbs(sample_tbsp_food, 2.0, "tbsp")
        assert round(result, 2) == 0.0

    def test_wrong_unit_raises(self, sample_gram_food: FoodItem):
        with pytest.raises(ValueError):
            _compute_carbs(sample_gram_food, 2.0, "slice")

    def test_negative_quantity(self, sample_gram_food: FoodItem):
        result = _compute_carbs(sample_gram_food, -100.0, "g")
        assert result < 0


class TestSerializeFood:
    def test_gram_only(self, sample_gram_food: FoodItem):
        result = _serialize_food(sample_gram_food)
        assert result["id"] == 1
        assert result["gram_unit"] == "g"
        assert "piece_unit" not in result

    def test_both_modes(self, sample_dual_food: FoodItem):
        result = _serialize_food(sample_dual_food)
        assert result["gram_unit"] == "g"
        assert result["piece_unit"] == "pcs"


class TestTallyTotalRounding:
    def _make_db(self) -> list[FoodItem]:
        return [
            FoodItem(
                id=1, name="Potatoes", standard_quantity_g=85.0, standard_quantity_pcs=None,
                carbs=15.0, carbs_per_100g=17.65, carbs_per_piece=None,
                has_grams_mode=True, has_pieces_mode=False, is_liquid=False,
                category="starchy_vegetables", gram_unit="g", piece_unit=None,
            ),
            FoodItem(
                id=2, name="Grapes", standard_quantity_g=85.0, standard_quantity_pcs=None,
                carbs=15.0, carbs_per_100g=17.65, carbs_per_piece=None,
                has_grams_mode=True, has_pieces_mode=False, is_liquid=False,
                category="fruits", gram_unit="g", piece_unit=None,
            ),
        ]

    def test_tally_total_matches_sum_of_displayed_carbs(self) -> None:
        h = MockHarness(self._make_db())
        h._exec_search_foods({"queries": ["potato", "grape"]})
        result = h._exec_add_foods_to_tally({
            "items": [
                {"food_id": 1, "quantity": 80, "unit": "g"},
                {"food_id": 2, "quantity": 100, "unit": "g"},
            ]
        })
        entry_sum = round(sum(e["carbs"] for e in result["entries"]), 2)
        assert result["tally_total"] == entry_sum
        assert result["tally_total"] == 31.77

    def test_calculate_final_uses_same_total(self) -> None:
        h = MockHarness(self._make_db())
        h._exec_search_foods({"queries": ["potato", "grape"]})
        h._exec_add_foods_to_tally({
            "items": [
                {"food_id": 1, "quantity": 80, "unit": "g"},
                {"food_id": 2, "quantity": 100, "unit": "g"},
            ]
        })
        result = h._exec_calculate_final({})
        assert result["tally_total"] == 31.77


class TestSearchFoods:
    def test_returns_matches(self, fresh_harness: MockHarness):
        result = fresh_harness._exec_search_foods({"queries": ["potato"]})
        assert "results" in result
        assert len(result["results"]) == 1
        assert len(result["results"][0]) == 1
        assert result["results"][0][0]["name"] == "Potatoes"

    def test_populates_known_ids(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        assert 1 in fresh_harness.known_food_ids

    def test_empty_result(self, fresh_harness: MockHarness):
        result = fresh_harness._exec_search_foods({"queries": ["zzz"]})
        assert result["results"] == [[]]

    def test_accumulates_known_ids(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_search_foods({"queries": ["bread"]})
        assert 1 in fresh_harness.known_food_ids
        assert 2 in fresh_harness.known_food_ids

    def test_case_insensitive(self, fresh_harness: MockHarness):
        result = fresh_harness._exec_search_foods({"queries": ["BREAD"]})
        assert len(result["results"][0]) == 1
        assert result["results"][0][0]["name"] == "Bread"


class TestAddFoodsToTally:
    def test_single_item(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        result = fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        assert result["tally_total"] == 20.0
        assert result["entries"][0]["entry_id"] == 1
        assert result["entries"][0]["food_name"] == "Potatoes"
        assert result["entries"][0]["carbs"] == 20.0
        assert result["entries"][0]["unit"] == "g"

    def test_sequential_ids(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 50, "unit": "g"}]
        })
        entry_ids = [e["entry_id"] for e in fresh_harness.tally_entries]
        assert entry_ids == [1, 2]

    def test_multi_item_batch(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato", "bread"]})
        result = fresh_harness._exec_add_foods_to_tally({
            "items": [
                {"food_id": 1, "quantity": 100, "unit": "g"},
                {"food_id": 2, "quantity": 2, "unit": "slice"},
            ]
        })
        assert result["tally_total"] == 50.0
        assert len(result["entries"]) == 2

    def test_unknown_food_id(self, fresh_harness: MockHarness):
        result = fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 99, "quantity": 100, "unit": "g"}]
        })
        assert "error" in result
        assert "99" in result["error"]

    def test_slice_unit(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["bread"]})
        result = fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 2, "quantity": 3, "unit": "slice"}]
        })
        assert result["tally_total"] == 45.0

    def test_tally_accumulates(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato", "bread", "cheese"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 2, "quantity": 2, "unit": "slice"}]
        })
        assert len(fresh_harness.tally_entries) == 2
        assert round(sum(e["carbs"] for e in fresh_harness.tally_entries), 1) == 50.0

    def test_cup_unit(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["cereal"]})
        result = fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 5, "quantity": 2, "unit": "cup"}]
        })
        assert result["tally_total"] == 50.0


class TestRemoveFoodsFromTally:
    def test_remove_existing(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 50, "unit": "g"}]
        })
        result = fresh_harness._exec_remove_foods_from_tally({"entry_ids": [1]})
        assert result["removed"] == 1
        assert result["tally_total"] == 10.0
        assert len(fresh_harness.tally_entries) == 1

    def test_remove_nonexistent(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        result = fresh_harness._exec_remove_foods_from_tally({"entry_ids": [99]})
        assert result["removed"] == 0
        assert len(fresh_harness.tally_entries) == 1

    def test_remove_all(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 50, "unit": "g"}]
        })
        result = fresh_harness._exec_remove_foods_from_tally({"entry_ids": [1, 2]})
        assert result["removed"] == 2
        assert result["tally_total"] == 0.0
        assert len(fresh_harness.tally_entries) == 0


class TestCalculateFinal:
    def _prepare_tally(self, harness: MockHarness):
        harness._exec_search_foods({"queries": ["potato"]})
        harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 150, "unit": "g"}]
        })

    def test_with_bg_above_threshold(self, fresh_harness: MockHarness):
        self._prepare_tally(fresh_harness)
        result = fresh_harness._exec_calculate_final({"blood_glucose": 180.0})
        assert result["glucose_correction"] > 0
        assert result["glucose_skipped"] is False
        assert result["final_result"] > result["food_insulin"]

    def test_with_bg_below_threshold(self, fresh_harness: MockHarness):
        self._prepare_tally(fresh_harness)
        result = fresh_harness._exec_calculate_final({"blood_glucose": 120.0})
        assert result["glucose_correction"] == 0.0
        assert result["glucose_skipped"] is True
        assert result["final_result"] == result["food_insulin"]

    def test_with_bg_at_threshold(self, fresh_harness: MockHarness):
        self._prepare_tally(fresh_harness)
        result = fresh_harness._exec_calculate_final({"blood_glucose": 130.0})
        assert result["glucose_correction"] > 0
        assert result["glucose_skipped"] is False

    def test_no_bg(self, fresh_harness: MockHarness):
        self._prepare_tally(fresh_harness)
        result = fresh_harness._exec_calculate_final({})
        assert result["blood_glucose"] is None
        assert result["glucose_correction"] == 0.0

    def test_empty_tally(self, fresh_harness: MockHarness):
        result = fresh_harness._exec_calculate_final({})
        assert "error" in result

    def test_meal_time_morning(self, fresh_harness: MockHarness):
        self._prepare_tally(fresh_harness)
        result = fresh_harness._exec_calculate_final({"meal_time": "morning"})
        assert result["meal_divider"] == 14

    def test_meal_time_evening(self, fresh_harness: MockHarness):
        self._prepare_tally(fresh_harness)
        result = fresh_harness._exec_calculate_final({"meal_time": "evening"})
        assert result["meal_divider"] == 12

    def test_meal_hour_inference(self, fresh_harness: MockHarness):
        self._prepare_tally(fresh_harness)
        result = fresh_harness._exec_calculate_final({"meal_hour": 20})
        assert result["meal_time"] == "evening"
        assert result["meal_divider"] == 12

    def test_custom_threshold(self, harness_custom: MockHarness):
        harness_custom._exec_search_foods({"queries": ["potato"]})
        harness_custom._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 150, "unit": "g"}]
        })
        result = harness_custom._exec_calculate_final({"blood_glucose": 110.0})
        assert result["glucose_correction"] > 0
        assert result["threshold"] == 100.0


class TestGetTallySummary:
    def test_returns_all_entries(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        result = fresh_harness._exec_get_tally_summary({})
        assert len(result["entries"]) == 1
        assert result["total_carbs"] == 20.0

    def test_reflects_state(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        fresh_harness._exec_calculate_final({"meal_time": "evening", "blood_glucose": 180.0})
        result = fresh_harness._exec_get_tally_summary({})
        assert result["meal_time"] == "evening"
        assert result["blood_glucose"] == 180.0
        assert result["glucose_enabled"] is True


class TestClearAll:
    def test_resets_everything(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        fresh_harness._exec_calculate_final({"meal_time": "morning", "blood_glucose": 150.0})

        fresh_harness._exec_clear_all({})

        assert len(fresh_harness.known_food_ids) == 0
        assert len(fresh_harness.tally_entries) == 0
        assert fresh_harness.meal_time is None
        assert fresh_harness.meal_hour is None
        assert fresh_harness.blood_glucose is None
        assert fresh_harness._next_entry_id == 1


class TestInferMealTime:
    def _make_harness(self):
        from data.food_db_loader import FoodItem
        db = [FoodItem(
            id=1, name="dummy", standard_quantity_g=100.0, standard_quantity_pcs=None,
            carbs=10.0, carbs_per_100g=10.0, carbs_per_piece=None,
            has_grams_mode=True, has_pieces_mode=False, is_liquid=False, category="other",
            gram_unit="g", piece_unit=None,
        )]
        return MockHarness(db)

    def test_morning(self):
        h = self._make_harness()
        assert h._infer_meal_time(8) == "morning"
        assert h._infer_meal_time(11) == "morning"
        assert h._infer_meal_time(4) == "morning"

    def test_midday(self):
        h = self._make_harness()
        assert h._infer_meal_time(12) == "midday"
        assert h._infer_meal_time(15) == "midday"

    def test_evening(self):
        h = self._make_harness()
        assert h._infer_meal_time(17) == "evening"
        assert h._infer_meal_time(23) == "evening"

    def test_wraparound(self):
        h = self._make_harness()
        assert h._infer_meal_time(0) == "evening"
        assert h._infer_meal_time(3) == "evening"

    def test_none_defaults_midday(self):
        h = self._make_harness()
        assert h._infer_meal_time(None) == "midday"


class TestGetContextBlock:
    def test_empty(self, fresh_harness: MockHarness):
        block = fresh_harness.get_context_block()
        assert "CURRENT TALLY: empty" in block
        assert "KNOWN FOOD IDS: none" in block

    def test_populated(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato", "bread"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 1, "quantity": 100, "unit": "g"}]
        })
        block = fresh_harness.get_context_block()
        assert "Potatoes" in block
        assert "100g" in block
        assert "entry_id: 1" in block
        assert "KNOWN FOOD IDS" in block
        assert "Potatoes(1)" in block

    def test_units_display_correct(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["bread"]})
        fresh_harness._exec_add_foods_to_tally({
            "items": [{"food_id": 2, "quantity": 2, "unit": "slice"}]
        })
        block = fresh_harness.get_context_block()
        assert "2slice" in block


class TestExecute:
    def test_execute_with_dict(self, fresh_harness: MockHarness):
        result = fresh_harness.execute({"name": "clear_all", "arguments": {}})
        assert result == {"success": True}

    def test_reset(self, fresh_harness: MockHarness):
        fresh_harness._exec_search_foods({"queries": ["potato"]})
        fresh_harness.reset()
        assert len(fresh_harness.known_food_ids) == 0
        assert len(fresh_harness.tally_entries) == 0
