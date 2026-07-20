from typing import Literal
from datasets import Dataset, DatasetDict


def load_dataset_for_model(
    model_type: Literal["lfm", "gemma"],
) -> DatasetDict:
    train = Dataset.from_json(f"data/output/{model_type}/train.jsonl")
    eval_ds = Dataset.from_json(f"data/output/{model_type}/eval.jsonl")
    return DatasetDict({"train": train, "eval": eval_ds})
