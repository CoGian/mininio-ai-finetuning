from pathlib import Path
from collections import defaultdict, Counter

from data.log_config import logger, StepTimer, setup_logging
from data.scenarios import Conversation, ScenarioType, SCENARIO_WEIGHTS

OUTPUT_DIR = Path("data/output")
REPORT_PATH = OUTPUT_DIR / "stats_report.md"


def generate_stats(raw_dir: str = "data/output/raw"):
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"No raw data found at {raw_path}")
        return

    logger.info("=== GENERATING STATS REPORT ===")

    with StepTimer("Stats report"):
        all_convs = []
        broken = 0
        for lang_file in sorted(raw_path.glob("*.jsonl")):
            with open(lang_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        conv = Conversation.model_validate_json(line)
                        all_convs.append(conv)
                    except Exception:
                        broken += 1

        if broken:
            logger.warning(f"Skipped {broken} invalid conversations")
        logger.info(f"Loaded {len(all_convs)} conversations")

        report_lines = []
        report_lines.append("# Dataset Generation Statistics Report\n")

        report_lines.append(f"## Summary\n")
        report_lines.append(f"- **Total conversations**: {len(all_convs)}")
        report_lines.append(f"- **Languages**: {len(set(c.language for c in all_convs))}")
        report_lines.append(f"- **Scenario types**: {len(set(c.scenario_type for c in all_convs))}")
        report_lines.append("")

        report_lines.append("## Per-Language Counts by Scenario Type\n")
        lang_scenario = defaultdict(lambda: defaultdict(int))
        lang_totals = defaultdict(int)
        for conv in all_convs:
            lang_scenario[conv.language][conv.scenario_type] += 1
            lang_totals[conv.language] += 1

        report_lines.append("| Language | " + " | ".join(s.value for s in ScenarioType) + " | Total |")
        report_lines.append("|" + "---|" * (len(ScenarioType) + 2))
        for lang in sorted(lang_scenario.keys()):
            counts = [str(lang_scenario[lang].get(s.value, 0)) for s in ScenarioType]
            total = str(lang_totals[lang])
            report_lines.append(f"| {lang} | " + " | ".join(counts) + f" | {total} |")
        report_lines.append("")

        report_lines.append("## Scenario Type Distribution vs Targets\n")
        scenario_counts = Counter(c.scenario_type for c in all_convs)
        total = len(all_convs)
        report_lines.append("| Scenario | Target % | Actual % | Count |")
        report_lines.append("|----------|----------|----------|-------|")
        for scenario in ScenarioType:
            target_pct = SCENARIO_WEIGHTS[scenario] * 100
            actual_pct = (scenario_counts.get(scenario.value, 0) / total * 100) if total > 0 else 0
            report_lines.append(f"| {scenario.value} | {target_pct:.0f}% | {actual_pct:.1f}% | {scenario_counts.get(scenario.value, 0)} |")
        report_lines.append("")

        report_lines.append("## Average Turns Per Conversation\n")
        avg_turns = sum(len(c.turns) for c in all_convs) / max(1, len(all_convs))
        report_lines.append(f"- **Average turns**: {avg_turns:.1f}\n")

        report_lines.append("## Token Count Distribution\n")
        token_counts = [
            len(" ".join(t.content or "" for t in c.turns)) // 4
            for c in all_convs
        ]
        if token_counts:
            report_lines.append(f"- **Min tokens**: {min(token_counts)}")
            report_lines.append(f"- **Max tokens**: {max(token_counts)}")
            report_lines.append(f"- **Median tokens**: {sorted(token_counts)[len(token_counts)//2]}")
            report_lines.append(f"- **Average tokens**: {sum(token_counts) / len(token_counts):.0f}")

        report_lines.append("")

        report_lines.append("## User Input Token Distribution\n")
        user_token_counts = [
            len(t.content or "") // 4
            for c in all_convs
            for t in c.turns
            if t.role == "user" and t.content
        ]
        if user_token_counts:
            sorted_utc = sorted(user_token_counts)
            report_lines.append(f"- **Min user tokens**: {min(user_token_counts)}")
            report_lines.append(f"- **Max user tokens**: {max(user_token_counts)}")
            report_lines.append(f"- **Median user tokens**: {sorted_utc[len(sorted_utc)//2]}")
            report_lines.append(f"- **Average user tokens**: {sum(user_token_counts) / len(user_token_counts):.0f}")
            report_lines.append(f"- **Total user turns**: {len(user_token_counts)}")
        report_lines.append("")

        report_lines.append("## Language Distribution\n")
        for lang in sorted(lang_totals.keys()):
            pct = lang_totals[lang] / max(1, total) * 100
            report_lines.append(f"- **{lang}**: {lang_totals[lang]} ({pct:.1f}%)")
        report_lines.append("")

        report_text = "\n".join(report_lines)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report_text, encoding="utf-8")
        logger.success(f"Stats report written to {REPORT_PATH}")


if __name__ == "__main__":
    setup_logging()
    generate_stats()
