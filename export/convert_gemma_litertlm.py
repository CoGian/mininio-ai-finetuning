import argparse
import os
import subprocess
import sys

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Gemma merged model to LiteRT-LM format")
    parser.add_argument("--merged-dir", required=True, help="Path to merged_16bit model dir")
    parser.add_argument("--output-dir", required=True, help="Directory for LiteRT-LM output")
    parser.add_argument(
        "--quant",
        default="int4",
        choices=["int4", "int8"],
        help="Quantization (default: int4)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Converting Gemma model from {args.merged_dir} to LiteRT-LM ({args.quant})...")

    cmd = [
        sys.executable,
        "-m",
        "litert_lm",
        "convert",
        "--model-path",
        args.merged_dir,
        "--output-dir",
        args.output_dir,
        "--quantize",
        args.quant,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error(f"Conversion failed:\n{result.stderr}")
        sys.exit(1)

    logger.info(f"Conversion complete. Output: {args.output_dir}")
    if result.stdout:
        logger.info(result.stdout)


if __name__ == "__main__":
    main()
