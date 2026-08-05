from collections.abc import Callable
from typing import Literal

from datasets import Dataset, DatasetDict
from loguru import logger


def load_dataset_for_model(model_type: Literal["lfm", "gemma"], data_dir: str = "data/output") -> DatasetDict:
    train = Dataset.from_json(f"{data_dir}/{model_type}/train.jsonl")
    eval_ds = Dataset.from_json(f"{data_dir}/{model_type}/eval.jsonl")
    return DatasetDict({"train": train, "eval": eval_ds})


def report_token_lengths(
    dataset: DatasetDict,
    tokenizer,
    model_type: str,
    max_length: int = 4096,
    n_samples: int = 500,
) -> int:
    sample = dataset["train"].shuffle(seed=42).select(range(min(n_samples, len(dataset["train"]))))
    lengths: list[int] = []

    for row in sample:
        tokens = tokenizer(row["text"], add_special_tokens=False, truncation=False)
        lengths.append(len(tokens["input_ids"]))

    lengths.sort()
    n = len(lengths)
    p50 = lengths[n // 2]
    p90 = lengths[int(n * 0.90)]
    p95 = lengths[int(n * 0.95)]
    p99 = lengths[int(n * 0.99)]
    p999 = lengths[int(n * 0.999)] if n > 1000 else lengths[-1]
    max_val = lengths[-1]
    mean_val = sum(lengths) / n

    pct_exceed = sum(1 for l in lengths if l > max_length) / n * 100

    logger.info(
        f"[{model_type}] Token length stats (n={n} sampled): "
        f"mean={mean_val:.0f} p50={p50} p90={p90} p95={p95} p99={p99} "
        f"p99.9={p999} max={max_val} "
        f"exceed_{max_length}={pct_exceed:.1f}%",
    )

    if pct_exceed > 1.0:
        logger.warning(
            f"{pct_exceed:.1f}% of examples exceed max_seq_length={max_length}. "
            f"Consider increasing to {int(max_val * 1.1)} to cover all examples.",
        )

    return max_val


def check_bos_prefix(
    dataset: DatasetDict,
    tokenizer,
    model_type: str,
) -> None:
    bos_text = tokenizer.bos_token or ""
    if not bos_text:
        logger.info(f"[{model_type}] No BOS token defined — nothing to check.")
        return

    for split_name in ["train", "eval"]:
        row = dataset[split_name][0]
        text: str = row["text"]

        if text.startswith(bos_text):
            bos_id = tokenizer.bos_token_id
            tokenized_with = tokenizer(text, add_special_tokens=True, truncation=True, max_length=8)
            tokenized_without = tokenizer(text, add_special_tokens=False, truncation=True, max_length=8)
            with_ids = tokenized_with["input_ids"]
            double_bos = (
                bos_id is not None
                and len(with_ids) >= 2
                and with_ids[0] == bos_id
                and with_ids[1] == bos_id
            )
            if double_bos:
                logger.info(
                    f"[{model_type}] {split_name}: BOS prefix OK — text includes BOS, "
                    f"tokenizer adds another (expected). Training uses add_special_tokens=False.",
                )
            else:
                logger.info(
                    f"[{model_type}] {split_name}: BOS prefix OK — no duplication.",
                )
        else:
            logger.warning(
                f"[{model_type}] {split_name}: text does NOT start with expected BOS '{bos_text}'.",
            )


def prepare_dataset(
    dataset: DatasetDict,
    masking_fn: Callable[[dict], dict],
) -> DatasetDict:
    mapped = {}
    for split_name in ["train", "eval"]:
        mapped[split_name] = dataset[split_name].map(
            masking_fn,
            batched=True,
            remove_columns=dataset[split_name].column_names,
        )
    return DatasetDict(mapped)
