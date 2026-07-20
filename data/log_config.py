import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger


_console_level = "INFO"


def setup_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    debug_harness: bool = False,
) -> None:
    global _console_level, logger
    _console_level = "DEBUG" if verbose else "INFO"

    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
        level=_console_level,
        colorize=True,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_level = "TRACE" if debug_harness else "DEBUG"
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=file_level,
            rotation="10 MB",
            retention="7 days",
        )
        logger.info(f"File log: {log_file}")

    if debug_harness:
        logger.info("Harness debug tracing: ENABLED")


class StepTimer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        logger.info(f"Starting: {self.name}")
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        logger.info(f"Finished: {self.name} ({_format_duration(elapsed)})")


class ProgressTracker:
    def __init__(self, total: int, label: str = "", log_interval: int = 5):
        self.total = total
        self.label = label
        self.log_interval = log_interval
        self.done = 0
        self.start_time = time.perf_counter()
        self._last_log_count = 0

    def update(self, n: int = 1) -> None:
        self.done += n
        if self.done - self._last_log_count >= self.log_interval:
            self._log_progress()
            self._last_log_count = self.done

    def bulk_resume(self, count: int) -> None:
        self.done = count
        self._last_log_count = count

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        pct = (self.done / self.total * 100) if self.total > 0 else 0
        rate = self.done / elapsed * 3600 if elapsed > 0 else 0
        logger.info(
            f"{self.label} Finished: {self.done}/{self.total} ({pct:.0f}%) "
            f"-- {_format_duration(elapsed)} (avg {rate:.1f}/h)"
        )

    def _log_progress(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        eta_str = ""
        if self.done > 0 and elapsed > 1:
            avg = elapsed / self.done
            remaining = self.total - self.done
            eta = max(0, avg * remaining)
            eta_str = f"  (ETA: {_format_duration(eta)})"
        logger.info(
            f"{self.label} {self.done}/{self.total}{eta_str}"
        )

    @property
    def rate_per_hour(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return self.done / elapsed * 3600 if elapsed > 0 else 0

    @property
    def elapsed_str(self) -> str:
        return _format_duration(time.perf_counter() - self.start_time)


class GlobalProgressTracker:
    def __init__(self, total_items: int, total_langs: int):
        self.total = total_items
        self.total_langs = total_langs
        self.done = 0
        self.start_time = time.perf_counter()
        self._last_log_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def update(self, n: int = 1) -> None:
        self.done += n
        if self.done - self._last_log_count >= 50:
            self._log_progress()
            self._last_log_count = self.done

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

    def _log_progress(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        pct = (self.done / self.total * 100) if self.total > 0 else 0
        rate = self.done / elapsed * 60 if elapsed > 0 else 0
        avg_item = elapsed / max(self.done, 1)
        remaining = self.total - self.done
        eta = max(0, avg_item * remaining)
        logger.info(
            f"[global] {self.done}/{self.total} ({pct:.0f}%) "
            f"| {rate:.1f}/min | global ETA: {_format_duration(eta)}"
        )

    @property
    def rate_per_hour(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return self.done / elapsed * 3600 if elapsed > 0 else 0

    @property
    def elapsed_str(self) -> str:
        return _format_duration(time.perf_counter() - self.start_time)

    @property
    def total_cost(self) -> float:
        # Gemini 2.5 Flash: $0.15/1M input, $0.60/1M output
        input_cost = (self._total_input_tokens / 1_000_000) * 0.15
        output_cost = (self._total_output_tokens / 1_000_000) * 0.60
        return input_cost + output_cost


def log_conversation_result(
    lang: str,
    scenario: str,
    success: bool,
    elapsed: float,
    attempts: int,
    error: Optional[str] = None,
) -> None:
    tag = f"[{lang}] {scenario}"
    dur = f"{elapsed:.1f}s ({attempts} attempt{'s' if attempts > 1 else ''})"
    if success:
        logger.debug(f"{tag} -- {dur}")
    elif attempts < 3:
        logger.warning(f"{tag} -- retry {attempts}/3: {dur} -- {error}")
    else:
        err_msg = error or "unknown"
        logger.error(f"{tag} -- FAIL: {dur} -- {err_msg}")


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"
