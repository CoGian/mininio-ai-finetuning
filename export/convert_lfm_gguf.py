import argparse
import os

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LFM LoRA to GGUF")
    parser.add_argument("--lora-dir", required=True, help="Path to saved LoRA adapter dir")
    parser.add_argument("--output-dir", required=True, help="Directory for GGUF output")
    parser.add_argument(
        "--quant",
        default="q8_0",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Quantization method (default: q8_0)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.lora_dir,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    logger.info(f"Saving GGUF ({args.quant}) to {args.output_dir}...")
    model.save_pretrained_gguf(
        args.output_dir,
        tokenizer,
        quantization_method=args.quant,
    )

    logger.info(f"GGUF saved. Files in {args.output_dir}:")
    for f in sorted(os.listdir(args.output_dir)):
        logger.info(f"  {f}")


if __name__ == "__main__":
    main()
