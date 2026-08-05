import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from data.log_config import logger, setup_logging
from data.mock_harness import USER_SETTINGS_POOL

SRC_DIR = Path("data/output/raw_migrated")


def _infer_settings_idx(results: list[dict]) -> Optional[int]:
    candidates = set(range(len(USER_SETTINGS_POOL)))

    for result in results:
        result_candidates = set()

        for i, pool_entry in enumerate(USER_SETTINGS_POOL):
            if result.get("threshold") != pool_entry["glucose_threshold"]:
                continue
            if result.get("baseline") != pool_entry["glucose_baseline"]:
                continue
            if result.get("divisor") != pool_entry["glucose_divisor"]:
                continue
            meal_time = result.get("meal_time")
            if meal_time:
                expected_divider = pool_entry["meal_dividers"].get(meal_time)
                if expected_divider is None or result.get("meal_divider") != expected_divider:
                    continue
            result_candidates.add(i)

        if not result_candidates:
            return None
        candidates &= result_candidates
        if not candidates:
            return None

    if len(candidates) == 1:
        return candidates.pop()
    return None


def _collect_calc_results(turns: list) -> list[dict]:
    results = []
    for turn in turns:
        if turn.get("role") != "tool" or not turn.get("tool_results"):
            continue
        calc = turn["tool_results"].get("calculate_final")
        if calc and isinstance(calc, dict) and "error" not in calc:
            results.append(calc)
    return results


def validate_settings() -> None:
    logger.info("=== SETTINGS VALIDATION ===")

    total = {"verified": 0, "mismatched": 0, "unverifiable": 0, "inconsistent": 0}
    detail_lines: list[str] = []

    for src_path in sorted(SRC_DIR.glob("*.jsonl")):
        file_totals = defaultdict(int)
        with open(src_path, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f):
                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    conv = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                assigned = conv.get("user_settings_idx")
                calc_results = _collect_calc_results(conv.get("turns", []))

                if not calc_results:
                    file_totals["unverifiable"] += 1
                    continue

                inferred = _infer_settings_idx(calc_results)

                if inferred is None:
                    file_totals["inconsistent"] += 1
                    detail_lines.append(
                        f"  {src_path.name}:{line_no + 1} INCONSISTENT — "
                        f"no single config matches all calc results; "
                        f"assigned_idx={assigned}"
                    )
                elif assigned is not None and inferred != assigned:
                    file_totals["mismatched"] += 1
                    assigned_cfg = USER_SETTINGS_POOL[assigned]
                    inferred_cfg = USER_SETTINGS_POOL[inferred]
                    detail_lines.append(
                        f"  {src_path.name}:{line_no + 1} MISMATCH — "
                        f"assigned_idx={assigned} ({assigned_cfg['glucose_threshold']}/"
                        f"{assigned_cfg['glucose_baseline']}/{assigned_cfg['glucose_divisor']}), "
                        f"inferred_idx={inferred} ({inferred_cfg['glucose_threshold']}/"
                        f"{inferred_cfg['glucose_baseline']}/{inferred_cfg['glucose_divisor']})"
                    )
                else:
                    file_totals["verified"] += 1

        for k in total:
            total[k] += file_totals[k]

        logger.info(
            f"  {src_path.name}: {sum(file_totals.values())} conversations"
        )
        logger.info(
            f"    Verified:       {file_totals['verified']}"
        )
        logger.info(
            f"    Mismatched:     {file_totals['mismatched']}"
        )
        logger.info(
            f"    Unverifiable:   {file_totals['unverifiable']}"
        )
        logger.info(
            f"    Inconsistent:   {file_totals['inconsistent']}"
        )

    grand = sum(total.values())
    logger.info("")
    logger.info(f"  TOTAL: {grand} conversations")
    logger.info(f"    Verified:       {total['verified']} "
                f"({total['verified'] / max(grand, 1) * 100:.1f}%)")
    logger.info(f"    Mismatched:     {total['mismatched']} "
                f"({total['mismatched'] / max(grand, 1) * 100:.1f}%)")
    logger.info(f"    Unverifiable:   {total['unverifiable']} "
                f"({total['unverifiable'] / max(grand, 1) * 100:.1f}%)")
    logger.info(f"    Inconsistent:   {total['inconsistent']} "
                f"({total['inconsistent'] / max(grand, 1) * 100:.1f}%)")

    if detail_lines:
        label = "Mismatches + Inconsistencies" if (
            total["mismatched"] + total["inconsistent"] > 0
        ) else "Details"
        logger.info(f"\n--- {label} ({len(detail_lines)}) ---")
        for line in detail_lines:
            logger.info(line)

    if total["mismatched"] == 0 and total["inconsistent"] == 0:
        logger.success("\n=== VALIDATION PASSED ===")
    else:
        logger.error("\n=== VALIDATION FAILED ===")


if __name__ == "__main__":
    setup_logging()
    validate_settings()
