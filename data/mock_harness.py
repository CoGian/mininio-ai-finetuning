import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from copy import deepcopy

from data.food_db_loader import FoodItem
from data.log_config import logger

DEFAULT_SETTINGS = {
    "glucose_threshold": 130.0,
    "glucose_baseline": 100.0,
    "glucose_divisor": 40.0,
    "meal_dividers": {"morning": 14, "midday": 15, "evening": 12},
    "meal_ranges": {
        "morning": (4, 12),
        "midday": (12, 17),
        "evening": (17, 4),
    },
}

USER_SETTINGS_POOL = [
    {
        "glucose_threshold": 130.0,
        "glucose_baseline": 100.0,
        "glucose_divisor": 40.0,
        "meal_dividers": {"morning": 14, "midday": 15, "evening": 12},
        "meal_ranges": {"morning": (4, 12), "midday": (12, 17), "evening": (17, 4)},
    },
    {
        "glucose_threshold": 120.0,
        "glucose_baseline": 90.0,
        "glucose_divisor": 30.0,
        "meal_dividers": {"morning": 8, "midday": 9, "evening": 7},
        "meal_ranges": {"morning": (4, 12), "midday": (12, 17), "evening": (17, 4)},
    },
    {
        "glucose_threshold": 150.0,
        "glucose_baseline": 110.0,
        "glucose_divisor": 55.0,
        "meal_dividers": {"morning": 18, "midday": 20, "evening": 16},
        "meal_ranges": {"morning": (4, 12), "midday": (12, 17), "evening": (17, 4)},
    },
    {
        "glucose_threshold": 110.0,
        "glucose_baseline": 80.0,
        "glucose_divisor": 25.0,
        "meal_dividers": {"morning": 20, "midday": 22, "evening": 18},
        "meal_ranges": {"morning": (4, 12), "midday": (12, 17), "evening": (17, 4)},
    },
    {
        "glucose_threshold": 140.0,
        "glucose_baseline": 105.0,
        "glucose_divisor": 50.0,
        "meal_dividers": {"morning": 10, "midday": 11, "evening": 10},
        "meal_ranges": {"morning": (4, 12), "midday": (12, 17), "evening": (17, 4)},
    },
]


def format_user_settings(settings: dict) -> str:
    td = settings["glucose_threshold"]
    bl = settings["glucose_baseline"]
    dv = settings["glucose_divisor"]
    md = settings["meal_dividers"]
    return (
        f"The current user has these settings:\n"
        f"- Glucose threshold: {td:.0f} mg/dL (only correct BG above this)\n"
        f"- Glucose baseline: {bl:.0f} mg/dL\n"
        f"- Glucose divisor: {dv:.0f} mg/dL per insulin unit\n"
        f"- Meal dividers: Morning={md['morning']:.0f}, Midday={md['midday']:.0f}, "
        f"Evening={md['evening']:.0f}\n"
        f"- Meal time ranges: Morning 4:00-12:00, Midday 12:00-17:00, Evening 17:00-4:00"
    )


def format_user_settings_training(settings: dict) -> str:
    td = settings["glucose_threshold"]
    bl = settings["glucose_baseline"]
    dv = settings["glucose_divisor"]
    md = settings["meal_dividers"]
    ms = settings["meal_ranges"]
    return (
        f"CURRENT USER SETTINGS:\n"
        f"- Glucose: threshold={td:.1f} mg/dL, baseline={bl:.1f} mg/dL, "
        f"divisor={dv:.1f} mg/dL per unit\n"
        f"- Meal dividers: Morning={md['morning']}, Midday={md['midday']}, "
        f"Evening={md['evening']}\n"
        f"- Meal time ranges: Morning ({ms['morning'][0]}:00-{ms['morning'][1]}:00), "
        f"Midday ({ms['midday'][0]}:00-{ms['midday'][1]}:00), "
        f"Evening ({ms['evening'][0]}:00-{ms['evening'][1]}:00)"
    )


def _make_search_result(food: FoodItem) -> dict:
    result = {
        "id": food.id,
        "name": food.name,
        "carbs_per_100g": food.carbs_per_100g,
        "carbs_per_piece": food.carbs_per_piece,
        "has_grams_mode": food.has_grams_mode,
        "has_pieces_mode": food.has_pieces_mode,
    }
    if food.has_grams_mode and food.gram_unit:
        result["gram_unit"] = food.gram_unit
    if food.has_pieces_mode and food.piece_unit:
        result["piece_unit"] = food.piece_unit
    return result


def _serialize_food(f: FoodItem) -> dict:
    result = {
        "id": f.id,
        "name": f.name,
        "carbs_per_100g": f.carbs_per_100g,
        "carbs_per_piece": f.carbs_per_piece,
        "has_grams_mode": f.has_grams_mode,
        "has_pieces_mode": f.has_pieces_mode,
    }
    if f.has_grams_mode and f.gram_unit:
        result["gram_unit"] = f.gram_unit
    if f.has_pieces_mode and f.piece_unit:
        result["piece_unit"] = f.piece_unit
    return result


ALL_GRAM_UNITS = {"g", "ml"}
ALL_PIECE_UNITS = {"pcs", "cup", "tbsp", "slice"}


def _compute_carbs(food: FoodItem, quantity: float, unit: str) -> float:
    if unit in ALL_GRAM_UNITS and food.has_grams_mode:
        return (quantity * food.carbs) / food.standard_quantity_g
    elif unit in ALL_PIECE_UNITS and food.has_pieces_mode:
        return (quantity * food.carbs) / food.standard_quantity_pcs
    else:
        raise ValueError(f"Cannot compute carbs for {unit} on {food.name}")


class MockHarness:
    def __init__(self, food_db: List[FoodItem], settings: dict = None):
        self.food_db = {f.id: f for f in food_db}
        self.settings = settings or deepcopy(DEFAULT_SETTINGS)
        self.known_food_ids: set[int] = set()
        self.tally_entries: list[dict] = []
        self.meal_time: Optional[str] = None
        self.meal_hour: Optional[int] = None
        self.blood_glucose: Optional[float] = None
        self._next_entry_id: int = 1

    def execute(self, tool_call) -> dict:
        name = tool_call.name if hasattr(tool_call, 'name') else tool_call["name"]
        args = tool_call.arguments if hasattr(tool_call, 'arguments') else tool_call.get("arguments", {})
        if args is None:
            args = {}

        logger.trace(f"Harness -> {name}({json.dumps(args, ensure_ascii=False)})")
        method = getattr(self, f"_exec_{name}")
        result = method(args)
        result_str = json.dumps(result, ensure_ascii=False)
        logger.trace(f"Harness <- {name}: {result_str[:200]}")
        return result

    def _exec_search_foods(self, args: dict) -> dict:
        queries = args["queries"]
        results = []
        for q in queries:
            q_lower = q.lower()
            matches = [f for f in self.food_db.values()
                       if q_lower in f.name.lower()]
            for m in matches:
                self.known_food_ids.add(m.id)
            results.append([_serialize_food(m) for m in matches])
        return {"results": results}

    def _exec_add_foods_to_tally(self, args: dict) -> dict:
        items = args["items"]
        entries = []
        total = 0.0
        for item in items:
            fid = item["food_id"]
            if fid not in self.known_food_ids:
                return {"error": f"Unknown food_id: {fid}. Search first."}
            food = self.food_db[fid]
            carbs = _compute_carbs(food, item["quantity"], item["unit"])
            if carbs < 0:
                carbs = 0.0
            entry = {
                "entry_id": self._next_entry_id,
                "food_name": food.name,
                "quantity": item["quantity"],
                "unit": item["unit"],
                "carbs": round(carbs, 2),
            }
            self._next_entry_id += 1
            entries.append(entry)
            total += carbs
            self.tally_entries.append(entry)
        return {"entries": entries, "tally_total": round(total, 2)}

    def _exec_remove_foods_from_tally(self, args: dict) -> dict:
        eids = set(args["entry_ids"])
        removed = 0
        new_tally = []
        for entry in self.tally_entries:
            if entry["entry_id"] in eids:
                removed += 1
            else:
                new_tally.append(entry)
        self.tally_entries = new_tally
        total = sum(e["carbs"] for e in self.tally_entries)
        return {"removed": removed, "tally_total": round(total, 2)}

    def _exec_calculate_final(self, args: dict) -> dict:
        if not self.tally_entries:
            return {"error": "Add at least one food first."}

        meal_time = args.get("meal_time")
        meal_hour = args.get("meal_hour")
        blood_glucose = args.get("blood_glucose")

        if meal_time:
            self.meal_time = meal_time
        elif not meal_time:
            meal_time = self._infer_meal_time(meal_hour)
            self.meal_time = meal_time
        else:
            meal_time = self.meal_time or self._infer_meal_time(meal_hour)

        if meal_hour is not None:
            self.meal_hour = meal_hour
        if blood_glucose is not None:
            self.blood_glucose = blood_glucose

        divider = self.settings["meal_dividers"].get(meal_time, 15)
        tally_total = sum(e["carbs"] for e in self.tally_entries)

        food_insulin = tally_total / divider

        threshold = self.settings["glucose_threshold"]
        baseline = self.settings["glucose_baseline"]
        divisor = self.settings["glucose_divisor"]

        glucose_correction = 0.0
        glucose_skipped = True
        if blood_glucose is not None and blood_glucose >= threshold:
            glucose_correction = max(0, (blood_glucose - baseline) / divisor)
            glucose_skipped = False

        final = food_insulin + glucose_correction

        breakdown_food = f"{tally_total:.2f}g / {divider} = {food_insulin:.2f}U"
        breakdown_glucose = ""
        if blood_glucose is not None and not glucose_skipped:
            breakdown_glucose = f"({blood_glucose} - {baseline}) / {divisor} = {glucose_correction:.2f}U"
        elif blood_glucose is not None:
            breakdown_glucose = f"BG {blood_glucose} < {threshold} threshold, correction skipped"

        return {
            "final_result": round(final, 2),
            "food_insulin": round(food_insulin, 2),
            "glucose_correction": round(glucose_correction, 2),
            "glucose_skipped": glucose_skipped,
            "tally_total": round(tally_total, 2),
            "meal_divider": divider,
            "meal_time": meal_time,
            "meal_hour": meal_hour or self.meal_hour,
            "blood_glucose": blood_glucose,
            "threshold": threshold,
            "baseline": baseline,
            "divisor": divisor,
            "breakdown_food": breakdown_food,
            "breakdown_glucose": breakdown_glucose,
        }

    def _exec_get_tally_summary(self, args: dict) -> dict:
        entries = [dict(e) for e in self.tally_entries]
        return {
            "entries": entries,
            "total_carbs": sum(e["carbs"] for e in self.tally_entries),
            "food_insulin": 0.0,
            "meal_time": self.meal_time,
            "meal_hour": self.meal_hour,
            "blood_glucose": self.blood_glucose,
            "glucose_enabled": self.blood_glucose is not None,
        }

    def _exec_clear_all(self, args: dict) -> dict:
        self.known_food_ids.clear()
        self.tally_entries.clear()
        self.meal_time = None
        self.meal_hour = None
        self.blood_glucose = None
        self._next_entry_id = 1
        return {"success": True}

    def _infer_meal_time(self, hour: Optional[int]) -> str:
        if hour is None:
            return "midday"
        ranges = self.settings["meal_ranges"]
        for period, (start, end) in ranges.items():
            if end > start:
                if start <= hour < end:
                    return period
            else:
                if hour >= start or hour < end:
                    return period
        return "midday"

    def get_context_block(self) -> str:
        if not self.tally_entries:
            return "[CURRENT TALLY: empty]\n[KNOWN FOOD IDS: none]"

        tally_lines = []
        for e in self.tally_entries:
            tally_lines.append(
                f"  {e['food_name']} {e['quantity']}{e['unit']} "
                f"= {e['carbs']}g (entry_id: {e['entry_id']})"
            )

        known_lines = []
        for fid in sorted(self.known_food_ids):
            known_lines.append(f"{self.food_db[fid].name}({fid})")

        total_carbs = sum(e["carbs"] for e in self.tally_entries)
        tally_str = f"[CURRENT TALLY: {len(self.tally_entries)} items, {total_carbs:.1f}g total]\n"
        tally_str += "\n".join(tally_lines)
        known_str = f"\n\n[KNOWN FOOD IDS: {', '.join(known_lines)}]"
        return tally_str + known_str

    def reset(self):
        self._exec_clear_all({})
