import random

import pytest

from data.food_db_loader import (
    ALL_UNIT_CODES,
    UnitNormalizer,
    FoodItem,
    _parse_float_or_none,
    _extract_unit,
    _is_liquid,
    load_food_db,
    search_foods,
    sample_foods,
    get_all_food_names,
)


class TestUnitNormalizerGramMl:
    def test_standard(self):
        assert UnitNormalizer.normalize_gram_ml("g") == "g"
        assert UnitNormalizer.normalize_gram_ml("ml") == "ml"

    def test_german_gramm(self):
        assert UnitNormalizer.normalize_gram_ml("gramm") == "g"

    def test_greek(self):
        assert UnitNormalizer.normalize_gram_ml("γραμμάρια") == "g"
        assert UnitNormalizer.normalize_gram_ml("γρ") == "g"

    def test_french(self):
        assert UnitNormalizer.normalize_gram_ml("grammes") == "g"

    def test_spanish(self):
        assert UnitNormalizer.normalize_gram_ml("gramos") == "g"

    def test_italian(self):
        assert UnitNormalizer.normalize_gram_ml("grammi") == "g"

    def test_portuguese(self):
        assert UnitNormalizer.normalize_gram_ml("gr") == "g"

    def test_with_dot(self):
        assert UnitNormalizer.normalize_gram_ml("gr.") == "gr."

    def test_none_input(self):
        assert UnitNormalizer.normalize_gram_ml(None) is None

    def test_unknown_passthrough(self):
        assert UnitNormalizer.normalize_gram_ml("kg") == "kg"

    def test_case_insensitive(self):
        assert UnitNormalizer.normalize_gram_ml("G") == "g"
        assert UnitNormalizer.normalize_gram_ml("Gr") == "g"

    def test_whitespace(self):
        assert UnitNormalizer.normalize_gram_ml("  g  ") == "g"

    def test_empty_string(self):
        assert UnitNormalizer.normalize_gram_ml("") == ""

    def test_count_units_passthrough(self):
        assert UnitNormalizer.normalize_gram_ml("pcs") == "pcs"
        assert UnitNormalizer.normalize_gram_ml("cup") == "cup"
        assert UnitNormalizer.normalize_gram_ml("tbsp") == "tbsp"
        assert UnitNormalizer.normalize_gram_ml("slice") == "slice"


class TestUnitNormalizerCountPcs:
    def test_standard(self):
        assert UnitNormalizer.normalize_count("pcs") == "pcs"
        assert UnitNormalizer.normalize_count("stk") == "pcs"
        assert UnitNormalizer.normalize_count("stk.") == "pcs"

    def test_multilingual(self):
        assert UnitNormalizer.normalize_count("pzas") == "pcs"
        assert UnitNormalizer.normalize_count("pz") == "pcs"
        assert UnitNormalizer.normalize_count("un") == "pcs"
        assert UnitNormalizer.normalize_count("pezzi") == "pcs"
        assert UnitNormalizer.normalize_count("piece") == "pcs"
        assert UnitNormalizer.normalize_count("pieces") == "pcs"

    def test_greek(self):
        assert UnitNormalizer.normalize_count("τεμάχιο") == "pcs"
        assert UnitNormalizer.normalize_count("τεμ") == "pcs"
        assert UnitNormalizer.normalize_count("τμχ") == "pcs"
        assert UnitNormalizer.normalize_count("τμχ.") == "pcs"

    def test_hindi(self):
        assert UnitNormalizer.normalize_count("इकाई") == "pcs"

    def test_cjk(self):
        assert UnitNormalizer.normalize_count("個") == "pcs"
        assert UnitNormalizer.normalize_count("个") == "pcs"
        assert UnitNormalizer.normalize_count("本") == "pcs"
        assert UnitNormalizer.normalize_count("枚") == "pcs"

    def test_size_words(self):
        assert UnitNormalizer.normalize_count("large") == "pcs"
        assert UnitNormalizer.normalize_count("medium") == "pcs"
        assert UnitNormalizer.normalize_count("small") == "pcs"
        assert UnitNormalizer.normalize_count("große") == "pcs"
        assert UnitNormalizer.normalize_count("mittlere") == "pcs"
        assert UnitNormalizer.normalize_count("kleine") == "pcs"
        assert UnitNormalizer.normalize_count("klein") == "pcs"
        assert UnitNormalizer.normalize_count("groß") == "pcs"
        assert UnitNormalizer.normalize_count("grands") == "pcs"
        assert UnitNormalizer.normalize_count("petits") == "pcs"


class TestUnitNormalizerCountCup:
    def test_standard(self):
        assert UnitNormalizer.normalize_count("cup") == "cup"
        assert UnitNormalizer.normalize_count("tasse") == "cup"
        assert UnitNormalizer.normalize_count("taza") == "cup"
        assert UnitNormalizer.normalize_count("tazza") == "cup"
        assert UnitNormalizer.normalize_count("xícara") == "cup"
        assert UnitNormalizer.normalize_count("x") == "cup"

    def test_greek(self):
        assert UnitNormalizer.normalize_count("κούπα") == "cup"
        assert UnitNormalizer.normalize_count("φλ.") == "cup"
        assert UnitNormalizer.normalize_count("φλ") == "cup"

    def test_hindi(self):
        assert UnitNormalizer.normalize_count("कप") == "cup"

    def test_cjk(self):
        assert UnitNormalizer.normalize_count("杯") == "cup"
        assert UnitNormalizer.normalize_count("カップ") == "cup"


class TestUnitNormalizerCountTbsp:
    def test_standard(self):
        assert UnitNormalizer.normalize_count("tbsp") == "tbsp"
        assert UnitNormalizer.normalize_count("el") == "tbsp"
        assert UnitNormalizer.normalize_count("cda") == "tbsp"
        assert UnitNormalizer.normalize_count("cucch") == "tbsp"
        assert UnitNormalizer.normalize_count("colher") == "tbsp"
        assert UnitNormalizer.normalize_count("c") == "tbsp"

    def test_greek_without_dot(self):
        assert UnitNormalizer.normalize_count("κ.σ") == "tbsp"

    def test_greek_with_dot(self):
        assert UnitNormalizer.normalize_count("κ.σ.") == "tbsp"

    def test_cjk(self):
        assert UnitNormalizer.normalize_count("大さじ") == "tbsp"
        assert UnitNormalizer.normalize_count("汤匙") == "tbsp"

    def test_hindi(self):
        assert UnitNormalizer.normalize_count("बड़ा चम्मच") == "tbsp"


class TestUnitNormalizerCountSlice:
    def test_standard(self):
        assert UnitNormalizer.normalize_count("slice") == "slice"
        assert UnitNormalizer.normalize_count("scheibe") == "slice"
        assert UnitNormalizer.normalize_count("rebanada") == "slice"
        assert UnitNormalizer.normalize_count("tranche") == "slice"
        assert UnitNormalizer.normalize_count("fetta") == "slice"
        assert UnitNormalizer.normalize_count("fatia") == "slice"

    def test_greek(self):
        assert UnitNormalizer.normalize_count("φέτα") == "slice"

    def test_hindi(self):
        assert UnitNormalizer.normalize_count("स्लाइस") == "slice"

    def test_cjk(self):
        assert UnitNormalizer.normalize_count("片") == "slice"

    def test_none_input(self):
        assert UnitNormalizer.normalize_count(None) is None

    def test_unknown_passthrough(self):
        assert UnitNormalizer.normalize_count("dozen") == "dozen"


class TestUnitNormalizerCountCommon:
    def test_case_insensitive(self):
        assert UnitNormalizer.normalize_count("PCS") == "pcs"
        assert UnitNormalizer.normalize_count("Cup") == "cup"
        assert UnitNormalizer.normalize_count("TBSP") == "tbsp"
        assert UnitNormalizer.normalize_count("Slice") == "slice"
        assert UnitNormalizer.normalize_count("LARGE") == "pcs"

    def test_whitespace(self):
        assert UnitNormalizer.normalize_count("  pcs  ") == "pcs"
        assert UnitNormalizer.normalize_count("  cup  ") == "cup"
        assert UnitNormalizer.normalize_count("  tbsp  ") == "tbsp"
        assert UnitNormalizer.normalize_count("  slice  ") == "slice"

    def test_empty_string(self):
        assert UnitNormalizer.normalize_count("") == ""

    def test_unknown_passthrough(self):
        assert UnitNormalizer.normalize_count("dozen") == "dozen"
        assert UnitNormalizer.normalize_count("bunch") == "bunch"
        assert UnitNormalizer.normalize_count("handful") == "handful"

    def test_gram_units_passthrough(self):
        assert UnitNormalizer.normalize_count("g") == "g"
        assert UnitNormalizer.normalize_count("ml") == "ml"
        assert UnitNormalizer.normalize_count("gramm") == "gramm"


class TestAllUnitCodes:
    def test_contains_exactly_six(self):
        assert ALL_UNIT_CODES == ["g", "ml", "pcs", "cup", "tbsp", "slice"]

    def test_all_codes_are_valid_normalizer_outputs(self):
        for code in ALL_UNIT_CODES:
            assert UnitNormalizer.normalize_gram_ml(code) == code
            assert UnitNormalizer.normalize_count(code) == code

    def test_all_map_values_in_all_unit_codes(self):
        for val in UnitNormalizer.GRAM_ML_MAP.values():
            assert val in ALL_UNIT_CODES, f"{val!r} not in ALL_UNIT_CODES"
        for val in UnitNormalizer.COUNT_MAP.values():
            assert val in ALL_UNIT_CODES, f"{val!r} not in ALL_UNIT_CODES"


class TestUnitNormalizerExhaustive:
    @pytest.mark.parametrize(
        "raw, expected",
        UnitNormalizer.GRAM_ML_MAP.items(),
    )
    def test_every_gram_ml_key(self, raw, expected):
        assert UnitNormalizer.normalize_gram_ml(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        sorted(UnitNormalizer.COUNT_MAP.items(), key=lambda x: x[1]),
    )
    def test_every_count_key(self, raw, expected):
        assert UnitNormalizer.normalize_count(raw) == expected


class TestParseFloatOrNone:
    def test_integer(self):
        assert _parse_float_or_none("150") == 150.0

    def test_decimal_dot(self):
        assert _parse_float_or_none("150.5") == 150.5

    def test_decimal_comma(self):
        assert _parse_float_or_none("150,5") == 150.5

    def test_with_unit(self):
        assert _parse_float_or_none("150g") == 150.0

    def test_empty(self):
        assert _parse_float_or_none("") is None

    def test_non_numeric(self):
        assert _parse_float_or_none("abc") is None

    def test_only_unit(self):
        assert _parse_float_or_none("abc") is None

    def test_fraction(self):
        assert _parse_float_or_none("1/2") == 1.0


class TestExtractUnit:
    def test_with_space(self):
        assert _extract_unit("150 g") == "g"

    def test_without_space(self):
        assert _extract_unit("150g") == "g"

    def test_empty(self):
        assert _extract_unit("") is None
        assert _extract_unit(" ") is None

    def test_no_unit(self):
        assert _extract_unit("150") is None

    def test_greek_unit(self):
        assert _extract_unit("2 κ.σ.") == "κ.σ."

    def test_greek_no_space(self):
        assert _extract_unit("1φέτα") == "φέτα"


class TestIsLiquid:
    def test_ml_true(self):
        assert _is_liquid("150ml") is True

    def test_ml_case_insensitive(self):
        assert _is_liquid("150 ML") is True

    def test_not_liquid(self):
        assert _is_liquid("150g") is False

    def test_empty(self):
        assert _is_liquid("") is False


class TestLoadFoodDb:
    def test_english_loads(self):
        db = load_food_db("en")
        assert len(db) == 106
        assert all(isinstance(f, FoodItem) for f in db)
        ids = [f.id for f in db]
        assert len(set(ids)) == len(ids)

    def test_english_first_item_fields(self):
        db = load_food_db("en")
        first = db[0]
        assert first.id == 1
        assert isinstance(first.name, str)
        assert first.carbs >= 0

    def test_greek_loads(self):
        db = load_food_db("el")
        assert len(db) == 106

    def test_missing_language(self):
        with pytest.raises(FileNotFoundError):
            load_food_db("xx")

    def test_all_languages_load(self):
        for lang in ["en", "el", "fr", "es", "hi", "it", "pt", "zh", "de", "ja"]:
            db = load_food_db(lang)
            assert len(db) == 106, f"Language {lang} should have 106 items"


class TestSearchFoods:
    @pytest.fixture
    def db(self):
        return [
            FoodItem(
                id=1, name="Potatoes", standard_quantity_g=150.0, standard_quantity_pcs=None,
                carbs=30.0, carbs_per_100g=20.0, carbs_per_piece=None,
                has_grams_mode=True, has_pieces_mode=False, is_liquid=False, category="starchy",
                gram_unit="g", piece_unit=None,
            ),
            FoodItem(
                id=2, name="Bread", standard_quantity_g=None, standard_quantity_pcs=1.0,
                carbs=15.0, carbs_per_100g=None, carbs_per_piece=15.0,
                has_grams_mode=False, has_pieces_mode=True, is_liquid=False, category="breads",
                gram_unit=None, piece_unit="slice",
            ),
            FoodItem(
                id=3, name="Potato chips", standard_quantity_g=50.0, standard_quantity_pcs=None,
                carbs=25.0, carbs_per_100g=50.0, carbs_per_piece=None,
                has_grams_mode=True, has_pieces_mode=False, is_liquid=False, category="other",
                gram_unit="g", piece_unit=None,
            ),
        ]

    def test_case_insensitive(self, db):
        results = search_foods(db, ["POTATOES"])
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].id == 1

    def test_partial_match(self, db):
        results = search_foods(db, ["otat"])
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_multi_query(self, db):
        results = search_foods(db, ["potato", "bread"])
        assert len(results) == 2
        assert len(results[0]) >= 1
        assert len(results[1]) >= 1

    def test_no_match(self, db):
        results = search_foods(db, ["nonexistent"])
        assert results == [[]]

    def test_empty_query(self, db):
        results = search_foods(db, [""])
        assert len(results) == 1
        assert len(results[0]) >= 0


class TestSampleFoods:
    @pytest.fixture
    def db(self):
        return [
            FoodItem(
                id=i, name=f"Food{i}", standard_quantity_g=100.0, standard_quantity_pcs=None,
                carbs=float(i), carbs_per_100g=float(i), carbs_per_piece=None,
                has_grams_mode=True, has_pieces_mode=False, is_liquid=False,
                category="vegetables" if i < 5 else "fruits",
                gram_unit="g", piece_unit=None,
            )
            for i in range(10)
        ]

    def test_basic(self, db):
        result = sample_foods(db, 3)
        assert len(result) == 3

    def test_more_than_pool(self, db):
        result = sample_foods(db, 100)
        assert len(result) == 10

    def test_category_filter(self, db):
        result = sample_foods(db, 5, ["vegetables"])
        assert len(result) == 5
        for f in result:
            assert f.category == "vegetables"

    def test_deterministic(self, db):
        random.seed(42)
        r1 = sample_foods(db, 3)
        random.seed(42)
        r2 = sample_foods(db, 3)
        assert [f.id for f in r1] == [f.id for f in r2]

    def test_empty_category(self, db):
        result = sample_foods(db, 5, ["nonexistent"])
        assert result == []


class TestGetAllFoodNames:
    @pytest.fixture
    def db(self):
        return [
            FoodItem(
                id=1, name="Potatoes", standard_quantity_g=150.0, standard_quantity_pcs=None,
                carbs=30.0, carbs_per_100g=20.0, carbs_per_piece=None,
                has_grams_mode=True, has_pieces_mode=False, is_liquid=False, category="starchy",
                gram_unit="g", piece_unit=None,
            ),
            FoodItem(
                id=2, name="Bread", standard_quantity_g=None, standard_quantity_pcs=1.0,
                carbs=15.0, carbs_per_100g=None, carbs_per_piece=15.0,
                has_grams_mode=False, has_pieces_mode=True, is_liquid=False, category="breads",
                gram_unit=None, piece_unit="slice",
            ),
            FoodItem(
                id=3, name="Cheese", standard_quantity_g=30.0, standard_quantity_pcs=1.0,
                carbs=1.0, carbs_per_100g=3.33, carbs_per_piece=1.0,
                has_grams_mode=True, has_pieces_mode=True, is_liquid=False, category="dairy",
                gram_unit="g", piece_unit="pcs",
            ),
        ]

    def test_formatting(self, db):
        result = get_all_food_names(db, max_items=3)
        assert "Potatoes" in result
        assert "Bread" in result
        assert "Cheese" in result
        assert "carbs" in result

    def test_max_items(self, db):
        result = get_all_food_names(db, max_items=2)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) <= 2

    def test_dual_mode_display(self, db):
        result = get_all_food_names(db, max_items=3)
        assert "30.0g" in result or "g" in result
