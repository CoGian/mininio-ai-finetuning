from copy import deepcopy

import pytest

from data.food_db_loader import FoodItem
from data.mock_harness import MockHarness, DEFAULT_SETTINGS, _make_search_result
from data.scenarios import Conversation, Turn, ToolCall, ScenarioType


@pytest.fixture
def sample_gram_food() -> FoodItem:
    return FoodItem(
        id=1,
        name="Potatoes",
        standard_quantity_g=150.0,
        standard_quantity_pcs=None,
        carbs=30.0,
        carbs_per_100g=20.0,
        carbs_per_piece=None,
        has_grams_mode=True,
        has_pieces_mode=False,
        is_liquid=False,
        category="starchy_vegetables",
        gram_unit="g",
        piece_unit=None,
    )


@pytest.fixture
def sample_piece_food() -> FoodItem:
    return FoodItem(
        id=2,
        name="Bread",
        standard_quantity_g=None,
        standard_quantity_pcs=1.0,
        carbs=15.0,
        carbs_per_100g=None,
        carbs_per_piece=15.0,
        has_grams_mode=False,
        has_pieces_mode=True,
        is_liquid=False,
        category="breads",
        gram_unit=None,
        piece_unit="slice",
    )


@pytest.fixture
def sample_dual_food() -> FoodItem:
    return FoodItem(
        id=3,
        name="Cheese",
        standard_quantity_g=30.0,
        standard_quantity_pcs=1.0,
        carbs=1.0,
        carbs_per_100g=(1.0 / 30.0) * 100,
        carbs_per_piece=1.0,
        has_grams_mode=True,
        has_pieces_mode=True,
        is_liquid=False,
        category="dairy",
        gram_unit="g",
        piece_unit="pcs",
    )


@pytest.fixture
def sample_liquid_food() -> FoodItem:
    return FoodItem(
        id=4,
        name="Milk",
        standard_quantity_g=200.0,
        standard_quantity_pcs=None,
        carbs=10.0,
        carbs_per_100g=5.0,
        carbs_per_piece=None,
        has_grams_mode=True,
        has_pieces_mode=False,
        is_liquid=True,
        category="dairy",
        gram_unit="ml",
        piece_unit=None,
    )


@pytest.fixture
def sample_cup_food() -> FoodItem:
    return FoodItem(
        id=5,
        name="Cereal",
        standard_quantity_g=None,
        standard_quantity_pcs=1.0,
        carbs=25.0,
        carbs_per_100g=None,
        carbs_per_piece=25.0,
        has_grams_mode=False,
        has_pieces_mode=True,
        is_liquid=False,
        category="other",
        gram_unit=None,
        piece_unit="cup",
    )


@pytest.fixture
def sample_tbsp_food() -> FoodItem:
    return FoodItem(
        id=6,
        name="Oil",
        standard_quantity_g=None,
        standard_quantity_pcs=1.0,
        carbs=0.0,
        carbs_per_100g=None,
        carbs_per_piece=0.0,
        has_grams_mode=False,
        has_pieces_mode=True,
        is_liquid=False,
        category="other",
        gram_unit=None,
        piece_unit="tbsp",
    )


@pytest.fixture
def sample_food_db(
    sample_gram_food,
    sample_piece_food,
    sample_dual_food,
    sample_liquid_food,
    sample_cup_food,
    sample_tbsp_food,
) -> list[FoodItem]:
    return [
        sample_gram_food,
        sample_piece_food,
        sample_dual_food,
        sample_liquid_food,
        sample_cup_food,
        sample_tbsp_food,
    ]


@pytest.fixture
def fresh_harness(sample_food_db: list[FoodItem]) -> MockHarness:
    return MockHarness(sample_food_db)


@pytest.fixture
def custom_settings() -> dict:
    s = deepcopy(DEFAULT_SETTINGS)
    s["glucose_threshold"] = 100.0
    return s


@pytest.fixture
def harness_custom(sample_food_db: list[FoodItem], custom_settings: dict) -> MockHarness:
    return MockHarness(sample_food_db, custom_settings)



_PAD = (
    " This is additional text to ensure the conversation is long enough "
    "to pass the minimum token length validation in the pipeline. "
    "The user is asking about their meal and the assistant provides "
    "a helpful and detailed response with all the nutritional information. "
)


@pytest.fixture
def valid_simple_conversation(sample_gram_food: FoodItem) -> Conversation:
    search_result = _make_search_result(sample_gram_food)
    add_result = {
        "entries": [
            {
                "entry_id": 1,
                "food_name": "Potatoes",
                "quantity": 100,
                "unit": "g",
                "carbs": 20.0,
            }
        ],
        "tally_total": 20.0,
    }

    return Conversation(
        scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
        language="en",
        turns=[
            Turn(
                role="user",
                content=f"I ate 100 grams of potatoes for lunch today.{_PAD}",
                context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(name="search_foods", arguments={"queries": ["potatoes"]})
                ],
                tool_results={"search_foods": {"results": [[search_result]]}},
            ),
            Turn(
                role="tool",
                tool_results={"search_foods": {"results": [[search_result]]}},
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(
                        name="add_foods_to_tally",
                        arguments={"items": [{"food_id": 1, "quantity": 100, "unit": "g"}]},
                    )
                ],
                tool_results={"add_foods_to_tally": add_result},
            ),
            Turn(
                role="tool",
                tool_results={"add_foods_to_tally": add_result},
            ),
            Turn(
                role="assistant",
                action="text",
                content=f"I have added 100g of potatoes to your tally, that is 20 grams of carbohydrates.{_PAD}",
            ),
        ],
    )


@pytest.fixture
def valid_multi_food_with_bg(
    sample_gram_food: FoodItem,
    sample_piece_food: FoodItem,
) -> Conversation:
    potato_result = _make_search_result(sample_gram_food)
    bread_result = _make_search_result(sample_piece_food)

    add_result = {
        "entries": [
            {
                "entry_id": 1,
                "food_name": "Potatoes",
                "quantity": 100,
                "unit": "g",
                "carbs": 20.0,
            },
            {
                "entry_id": 2,
                "food_name": "Bread",
                "quantity": 2,
                "unit": "slice",
                "carbs": 30.0,
            },
        ],
        "tally_total": 50.0,
    }

    calc_result = {
        "final_result": 5.33,
        "food_insulin": 3.33,
        "glucose_correction": 2.00,
        "glucose_skipped": False,
        "tally_total": 50.0,
        "meal_divider": 15,
        "meal_time": "midday",
        "meal_hour": None,
        "blood_glucose": 180.0,
        "threshold": 130.0,
        "baseline": 100.0,
        "divisor": 40.0,
        "breakdown_food": "50.0g / 15 = 3.33U",
        "breakdown_glucose": "(180.0 - 100.0) / 40.0 = 2.00U",
    }

    return Conversation(
        scenario_type=ScenarioType.MULTIPLE_FOODS_WITH_GLUCOSE.value,
        language="en",
        turns=[
            Turn(
                role="user",
                content=(
                    f"I had 100g of potatoes and 2 slices of bread, "
                    f"my blood sugar is 180. Can you help me calculate?{_PAD}"
                ),
                context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(
                        name="search_foods",
                        arguments={"queries": ["potatoes", "bread"]},
                    )
                ],
                tool_results={
                    "search_foods": {"results": [[potato_result], [bread_result]]}
                },
            ),
            Turn(
                role="tool",
                tool_results={
                    "search_foods": {"results": [[potato_result], [bread_result]]}
                },
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(
                        name="add_foods_to_tally",
                        arguments={
                            "items": [
                                {"food_id": 1, "quantity": 100, "unit": "g"},
                                {"food_id": 2, "quantity": 2, "unit": "slice"},
                            ]
                        },
                    )
                ],
                tool_results={"add_foods_to_tally": add_result},
            ),
            Turn(
                role="tool",
                tool_results={"add_foods_to_tally": add_result},
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(
                        name="calculate_final",
                        arguments={"blood_glucose": 180.0},
                    )
                ],
                tool_results={"calculate_final": calc_result},
            ),
            Turn(
                role="tool",
                tool_results={"calculate_final": calc_result},
            ),
            Turn(
                role="assistant",
                action="text",
                content=(
                    f"Here is your calculation: total carbs 50g, food insulin 3.33U, "
                    f"glucose correction 2.00U, final dose 5.33U.{_PAD}"
                ),
            ),
        ],
    )


@pytest.fixture
def invalid_no_context(sample_gram_food: FoodItem) -> Conversation:
    return Conversation(
        scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
        language="en",
        turns=[
            Turn(
                role="user",
                content="I ate potatoes",
            ),
        ],
    )


@pytest.fixture
def invalid_bad_unit(sample_gram_food: FoodItem) -> Conversation:
    search_result = _make_search_result(sample_gram_food)
    add_result = {
        "entries": [
            {
                "entry_id": 1,
                "food_name": "Potatoes",
                "quantity": 2,
                "unit": "slice",
                "carbs": 0.0,
            }
        ],
        "tally_total": 0.0,
    }

    return Conversation(
        scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
        language="en",
        turns=[
            Turn(
                role="user",
                content=f"I ate 2 slices of potatoes.{_PAD}",
                context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(name="search_foods", arguments={"queries": ["potatoes"]})
                ],
                tool_results={"search_foods": {"results": [[search_result]]}},
            ),
            Turn(
                role="tool",
                tool_results={"search_foods": {"results": [[search_result]]}},
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(
                        name="add_foods_to_tally",
                        arguments={"items": [{"food_id": 1, "quantity": 2, "unit": "slice"}]},
                    )
                ],
                tool_results={"add_foods_to_tally": add_result},
            ),
            Turn(
                role="tool",
                tool_results={"add_foods_to_tally": add_result},
            ),
            Turn(
                role="assistant",
                action="text",
                content=f"Added.{_PAD}",
            ),
        ],
    )


@pytest.fixture
def invalid_wrong_order() -> Conversation:
    return Conversation(
        scenario_type=ScenarioType.SIMPLE_SINGLE_FOOD.value,
        language="en",
        turns=[
            Turn(
                role="user",
                content=f"I ate potatoes.{_PAD}",
                context_block="[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]",
            ),
            Turn(
                role="assistant",
                action="tool_call",
                tool_calls=[
                    ToolCall(
                        name="add_foods_to_tally",
                        arguments={"items": [{"food_id": 1, "quantity": 100, "unit": "g"}]},
                    )
                ],
                tool_results={
                    "add_foods_to_tally": {
                        "entries": [
                            {
                                "entry_id": 1,
                                "food_name": "Potatoes",
                                "quantity": 100,
                                "unit": "g",
                                "carbs": 20.0,
                            }
                        ],
                        "tally_total": 20.0,
                    }
                },
            ),
            Turn(
                role="tool",
                tool_results={
                    "add_foods_to_tally": {
                        "entries": [
                            {
                                "entry_id": 1,
                                "food_name": "Potatoes",
                                "quantity": 100,
                                "unit": "g",
                                "carbs": 20.0,
                            }
                        ],
                        "tally_total": 20.0,
                    }
                },
            ),
            Turn(
                role="assistant",
                action="text",
                content=f"Added.{_PAD}",
            ),
        ],
    )
