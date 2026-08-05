import os
from dataclasses import dataclass, field
from typing import Any, Optional


def detect_env() -> str:
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    return "local"


@dataclass
class TrainingConfig:
    data_dir: str
    output_dir: str
    max_seq_length: int = 4096
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    seed: int = 3407
    save_steps: int = 500
    logging_steps: int = 25

    @classmethod
    def for_env(
        cls,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        **overrides,
    ) -> "TrainingConfig":
        env = detect_env()

        if data_dir is None:
            if env == "colab":
                data_dir = "/content/data/output"
            elif env == "kaggle":
                data_dir = "/kaggle/input/mininio-data/data/output"
            else:
                data_dir = "data/output"

        if output_dir is None:
            if env == "colab":
                output_dir = "/content/drive/MyDrive/mininio-checkpoints"
            elif env == "kaggle":
                output_dir = "/kaggle/working/checkpoints"
            else:
                output_dir = "finetuning/output"

        kwargs: dict[str, Any] = {
            "data_dir": data_dir,
            "output_dir": output_dir,
        }
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class LFMConfig(TrainingConfig):
    _model_name: str = field(default="unsloth/LFM2.5-1.2B-Instruct", init=False)

    @property
    def model_name(self) -> str:
        return self._model_name


@dataclass
class GemmaConfig(TrainingConfig):
    _model_name: str = field(default="unsloth/gemma-4-E2B-it", init=False)

    @property
    def model_name(self) -> str:
        return self._model_name
