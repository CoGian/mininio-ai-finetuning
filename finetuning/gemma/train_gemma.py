import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch
from loguru import logger
from trl import SFTConfig, SFTTrainer

from finetuning.common.config import TrainingConfig, detect_env
from finetuning.common.data_loader import (
    check_bos_prefix,
    load_dataset_for_model,
    prepare_dataset,
    report_token_lengths,
)
from finetuning.common.masking import make_masking_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 E2B")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (Gemma default)")
    parser.add_argument("--report-to", default=None)
    args = parser.parse_args()

    env = args.env or detect_env()

    report_to = args.report_to
    if report_to is None:
        report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    logger.info(f"Report to: {report_to}")

    config = TrainingConfig.for_env(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
    )

    gemma_data_dir = os.path.join(config.data_dir, "gemma")
    output_dir = os.path.join(config.output_dir, "gemma")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Environment: {env}")
    logger.info(f"Data dir: {gemma_data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(
        f"Training config: bs={config.per_device_batch_size} "
        f"ga={config.gradient_accumulation_steps} "
        f"epochs={config.num_train_epochs} lr={config.learning_rate} "
        f"max_seq={config.max_seq_length} seed={config.seed}",
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-4-E2B-it",
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
    )

    logger.info("Loading dataset...")
    dataset = load_dataset_for_model("gemma", config.data_dir)
    train_size = len(dataset["train"])
    logger.info(
        f"Train: {train_size} examples, "
        f"Eval: {len(dataset['eval'])} examples",
    )

    effective_bs = config.per_device_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = (train_size + effective_bs - 1) // effective_bs
    total_steps = steps_per_epoch * config.num_train_epochs
    dynamic_save = max(50, total_steps // 3)
    config.save_steps = dynamic_save
    logger.info(
        f"Steps per epoch: {steps_per_epoch}, total: {total_steps}, "
        f"save/checkpoint every {dynamic_save} steps",
    )

    logger.info("Checking BOS prefix...")
    check_bos_prefix(dataset, tokenizer, "gemma")

    logger.info("Reporting token lengths...")
    report_token_lengths(dataset, tokenizer, "gemma", config.max_seq_length)

    logger.info("Applying custom masking (model turns only, no tool results)...")
    masking_fn = make_masking_fn(tokenizer, "gemma", config.max_seq_length)
    dataset = prepare_dataset(dataset, masking_fn)

    logger.info("Setting up SFTTrainer...")
    lr_str = f"{config.learning_rate:.0e}".replace(".0e-0", "e").replace(".0e-", "e-")
    run_name = f"gemma-lora-bs{effective_bs}-lr{lr_str}-ep{config.num_train_epochs}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        weight_decay=0.001,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        max_length=None,
        seed=config.seed,
        report_to=report_to,
        fp16=False,
        bf16=args.bf16,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="steps",
        eval_steps=config.save_steps,
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=sft_config,
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    total_memory = round(gpu_stats.total_memory / 1024**3, 3)
    logger.info(f"GPU: {gpu_stats.name} ({total_memory} GB)")
    logger.info(f"Reserved before training: {start_memory} GB")

    logger.info("Training...")
    trainer_stats = trainer.train()

    end_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    training_memory = round(end_memory - start_memory, 3)
    peak_pct = round(end_memory / total_memory * 100, 1)

    logger.info(
        f"Training complete in "
        f"{round(trainer_stats.metrics['train_runtime'], 1)}s "
        f"({round(trainer_stats.metrics['train_runtime'] / 60, 1)} min)",
    )
    logger.info(
        f"Peak memory: {end_memory} GB ({peak_pct}%), "
        f"training overhead: {training_memory} GB",
    )

    lora_dir = os.path.join(output_dir, "lora_adapter")
    logger.info(f"Saving LoRA adapter to {lora_dir}...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    merged_dir = os.path.join(output_dir, "merged_16bit")
    logger.info(f"Saving merged 16-bit model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    logger.info("Done!")


if __name__ == "__main__":
    main()
