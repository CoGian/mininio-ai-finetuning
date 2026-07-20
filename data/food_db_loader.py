from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
import random
import re

FOOD_DB_DIR = Path(__file__).parent / "food_db"

ALL_UNIT_CODES = ["g", "ml", "pcs", "cup", "tbsp", "slice"]


class UnitNormalizer:
    GRAM_ML_MAP = {
        "g": "g", "gr": "g", "gramm": "g", "grammi": "g", "gramos": "g",
        "grammes": "g", "γραμμάρια": "g", "γρ": "g",
        "ml": "ml",
    }

    COUNT_MAP = {
        "pcs": "pcs", "stk": "pcs", "stk.": "pcs", "pzas": "pcs",
        "pz": "pcs", "un": "pcs", "pezzi": "pcs", "piece": "pcs",
        "pieces": "pcs", "τεμάχιο": "pcs", "τεμ": "pcs", "τμχ": "pcs",
        "τμχ.": "pcs", "इकाई": "pcs",
        "個": "pcs", "个": "pcs", "本": "pcs", "枚": "pcs",
        "large": "pcs", "medium": "pcs", "small": "pcs",
        "große": "pcs", "mittlere": "pcs", "kleine": "pcs",
        "klein": "pcs", "groß": "pcs", "grands": "pcs", "petits": "pcs",

        "cup": "cup", "tasse": "cup", "taza": "cup", "tazza": "cup",
        "κούπα": "cup", "xícara": "cup", "x": "cup",
        "φλ.": "cup", "φλ": "cup", "कप": "cup",
        "杯": "cup", "カップ": "cup",

        "tbsp": "tbsp", "el": "tbsp", "cda": "tbsp", "cucch": "tbsp",
        "κ.σ": "tbsp", "κ.σ.": "tbsp", "c": "tbsp", "colher": "tbsp",
        "汤匙": "tbsp",
        "大さじ": "tbsp", "बड़ा चम्मच": "tbsp",

        "slice": "slice", "scheibe": "slice", "rebanada": "slice",
        "tranche": "slice", "fetta": "slice", "fatia": "slice",
        "φέτα": "slice", "स्लाइस": "slice",
        "片": "slice",
    }

    @staticmethod
    def normalize_gram_ml(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        lower = raw.strip().lower()
        return UnitNormalizer.GRAM_ML_MAP.get(lower, raw.strip())

    @staticmethod
    def normalize_count(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        lower = raw.strip().lower()
        return UnitNormalizer.COUNT_MAP.get(lower, raw.strip())


@dataclass
class FoodItem:
    id: int
    name: str
    standard_quantity_g: Optional[float]
    standard_quantity_pcs: Optional[float]
    carbs: float
    carbs_per_100g: Optional[float]
    carbs_per_piece: Optional[float]
    has_grams_mode: bool
    has_pieces_mode: bool
    is_liquid: bool
    category: str
    gram_unit: Optional[str] = None
    piece_unit: Optional[str] = None


def _parse_float_or_none(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    match = re.match(r'[\d]+(?:[.,]\d+)?', value)
    if match:
        num_str = match.group().replace(",", ".")
        return float(num_str)
    return None


def _extract_unit(raw: str) -> Optional[str]:
    raw = raw.strip()
    if not raw:
        return None
    match = re.match(r'[\d.,/ ]+', raw)
    if match:
        unit_text = raw[match.end():].strip()
        return unit_text if unit_text else None
    return None


def _is_liquid(quantity_str: str) -> bool:
    return "ml" in quantity_str.lower()


def _classify_category(name: str, standard_quantity_g: Optional[float], carbs: float) -> str:
    lower = name.lower()
    dairy_keywords = ["milk", "yogurt", "kefir", "yoghurt", "soy milk", "soymilk",
                      "la'it", "milch", "jogurt", "yaourt", "yogur"]
    if any(kw in lower for kw in dairy_keywords):
        return "dairy"
    legume_keywords = ["pea", "chickpea", "bean", "lentil", "lentils",
                       "ervilhas", "grao", "feijao", "lentilhas",
                       "chiche", "lentejas", "garbanzo",
                       "chana", "dal",
                       "revithia", "fakes", "fasolia",
                       "ceci", "fagioli", "lenticchie", "piselli"]
    if any(kw in lower for kw in legume_keywords):
        return "legumes"
    starchy_keywords = ["sweet potato", "potato", "corn", "chestnut",
                        "batata", "patata", "kartoffel", "pomme de terre",
                        "mais", "maiz", "milho", "castanha"]
    if any(kw in lower for kw in starchy_keywords):
        if standard_quantity_g is None:
            return "starchy_vegetables"
    if carbs <= 5.5:
        return "vegetables"
    if carbs >= 12 and carbs <= 16:
        return "fruits"
    if carbs >= 15 and carbs <= 16:
        return "breads"
    return "other"


def load_food_db(lang: str) -> List[FoodItem]:
    filepath = FOOD_DB_DIR / f"{lang}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Food DB not found for language '{lang}': {filepath}")

    items = []
    with open(filepath, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            continue
        parts = line.split(";")
        if len(parts) < 4:
            continue
        name = parts[0].strip()
        qty_g_str = parts[1].strip() if len(parts) > 1 else ""
        qty_pcs_str = parts[2].strip() if len(parts) > 2 else ""
        carbs_str = parts[3].strip() if len(parts) > 3 else ""

        standard_quantity_g = _parse_float_or_none(qty_g_str)
        standard_quantity_pcs = _parse_float_or_none(qty_pcs_str)
        carbs = _parse_float_or_none(carbs_str) or 0.0

        raw_gram_unit = _extract_unit(qty_g_str)
        raw_piece_unit = _extract_unit(qty_pcs_str)

        gram_unit = UnitNormalizer.normalize_gram_ml(raw_gram_unit)
        piece_unit = UnitNormalizer.normalize_count(raw_piece_unit)

        has_grams_mode = standard_quantity_g is not None and gram_unit is not None
        has_pieces_mode = standard_quantity_pcs is not None and piece_unit is not None
        is_liquid = _is_liquid(qty_g_str)

        carbs_per_100g = None
        if has_grams_mode and standard_quantity_g and standard_quantity_g > 0:
            carbs_per_100g = round((carbs / standard_quantity_g) * 100, 2)

        carbs_per_piece = None
        if has_pieces_mode and standard_quantity_pcs and standard_quantity_pcs > 0:
            carbs_per_piece = round(carbs / standard_quantity_pcs, 2)

        category = _classify_category(name, standard_quantity_g, carbs)

        items.append(FoodItem(
            id=i,
            name=name,
            standard_quantity_g=standard_quantity_g,
            standard_quantity_pcs=standard_quantity_pcs,
            carbs=carbs,
            carbs_per_100g=carbs_per_100g,
            carbs_per_piece=carbs_per_piece,
            has_grams_mode=has_grams_mode,
            has_pieces_mode=has_pieces_mode,
            is_liquid=is_liquid,
            category=category,
            gram_unit=gram_unit,
            piece_unit=piece_unit,
        ))

    return items


@lru_cache(maxsize=32)
def load_food_db_cached(lang: str) -> List[FoodItem]:
    return load_food_db(lang)


def search_foods(db: List[FoodItem], queries: List[str]) -> List[List[FoodItem]]:
    results = []
    for q in queries:
        q_lower = q.lower()
        matches = [f for f in db if q_lower in f.name.lower()]
        results.append(matches)
    return results


def sample_foods(db: List[FoodItem], n: int, categories: Optional[List[str]] = None) -> List[FoodItem]:
    pool = db
    if categories:
        cat_set = set(categories)
        pool = [f for f in db if f.category in cat_set]
    if len(pool) <= n:
        return list(pool)
    return random.sample(pool, n)


def get_all_food_names_and_ids(db: List[FoodItem], max_items: int = 20) -> tuple[str, list[str]]:
    foods = sample_foods(db, min(max_items, len(db)))
    lines = []
    ids = []
    for f in foods:
        modes = []
        if f.has_grams_mode and f.gram_unit:
            modes.append(f"{f.standard_quantity_g}{f.gram_unit}")
        if f.has_pieces_mode and f.piece_unit:
            modes.append(f"{f.standard_quantity_pcs}{f.piece_unit}")
        modes_str = " / ".join(modes) if modes else "?"
        unit_tag = _build_unit_tag(f)
        lines.append(f"  {f.id}: {f.name} ({modes_str} = {f.carbs}g carbs) {unit_tag}")
        ids.append(f.id)
    return "\n".join(lines), ids


def _build_unit_tag(f: FoodItem) -> str:
    if f.has_grams_mode and f.has_pieces_mode:
        g = f.gram_unit or "g"
        p = f.piece_unit or "pcs"
        return f'[USE: unit="{g}" or "{p}"]'
    elif f.has_grams_mode:
        g = f.gram_unit or "g"
        return f'[USE: unit="{g}" only]'
    elif f.has_pieces_mode:
        p = f.piece_unit or "pcs"
        return f'[USE: unit="{p}" only]'
    return ""


def get_all_food_names(db: List[FoodItem], max_items: int = 20) -> str:
    s, _ = get_all_food_names_and_ids(db, max_items)
    return s
