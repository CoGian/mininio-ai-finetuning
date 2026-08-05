# Evaluation criteria and scoring weights per AI_INTEGRATION_PLAN.md Section 10.5
# Phase 5: Model Selection

from dataclasses import dataclass


@dataclass
class EvalMetrics:
    tool_call_accuracy: float = 0.0
    sequence_correctness: float = 0.0
    clarification_quality: float = 0.0
    natural_language_quality: float = 0.0
    latency_score: float = 0.0
    memory_score: float = 0.0

    def weighted_score(self) -> float:
        return (
            self.tool_call_accuracy * 0.40
            + self.sequence_correctness * 0.25
            + self.clarification_quality * 0.15
            + (self.natural_language_quality / 5.0) * 0.10
            + self.latency_score * 0.05
            + self.memory_score * 0.05
        )

    def report(self) -> str:
        return (
            f"Tool call accuracy:       {self.tool_call_accuracy:.1%} (weight 40%)\n"
            f"Sequence correctness:     {self.sequence_correctness:.1%} (weight 25%)\n"
            f"Clarification quality:    {self.clarification_quality:.1%} (weight 15%)\n"
            f"Natural language quality: {self.natural_language_quality:.1f}/5 (weight 10%)\n"
            f"Latency score:            {self.latency_score:.3f} (weight 5%)\n"
            f"Memory score:             {self.memory_score:.3f} (weight 5%)\n"
            f"=== Weighted score: {self.weighted_score():.4f} ==="
        )


TOOL_CALL_WEIGHT = 0.40
SEQUENCE_WEIGHT = 0.25
CLARIFICATION_WEIGHT = 0.15
NL_QUALITY_WEIGHT = 0.10
LATENCY_WEIGHT = 0.05
MEMORY_WEIGHT = 0.05

TOOL_CALL_TARGET = 0.85
SEQUENCE_TARGET = 0.90
CLARIFICATION_TARGET = 0.80
NL_QUALITY_TARGET = 3.5
TTFT_TARGET_MS = 1000
MEMORY_TARGET_MB = 1500

FINAL_RESULT_TOLERANCE = 0.01
TIEBREAKER_MARGIN = 0.05
