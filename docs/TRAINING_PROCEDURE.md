# Training Procedure — Mininio AI Fine-Tuning

## Overview

Dual-candidate fine-tuning pipeline: both candidate models are trained on identical 8,000 synthetic tool-calling conversations generated across 10 languages, then evaluated head-to-head to select the winner for Android deployment.

| | Candidate A (Primary) | Candidate B (Fallback) |
|---|---|---|
| Model | LFM2.5-1.2B-Instruct | Gemma 4 E2B |
| Framework | Unsloth `FastLanguageModel` | Unsloth `FastModel` |
| Trainer | TRL SFTTrainer | TRL SFTTrainer |
| Data | 5925 train / 701 eval (90/10 stratified) | Same |
| Max seq length | 4096 | 4096 |
| Total steps | 2,223 (3 epochs) | 4,446 (3 epochs) |
| Trainable params | 9.1M of 1.18B (0.78%) | LoRA (Q/K/V/MLP) |
| 4-bit | Yes | Yes |

---

## 1. Data Format

The same 8,000 conversations are rendered into two model-specific templates at generation time and stored as flat `text` fields in `data/output/{lfm,gemma}/train.jsonl` and `eval.jsonl`.

### Why pre-rendered text, not messages arrays

Both models share identical semantic content (same tool calls, same responses, same tool results) but differ only in template tokens. Rendering at data-generation time avoids template re-parsing overhead during training. The training loop passes the raw string directly to the tokenizer via `dataset_text_field=""` and `skip_prepare_dataset=True`.

### LFM format (ChatML)

```
<|startoftext|>
<|im_start|>system
[system prompt with tools + user settings]
<|im_end|>
<|im_start|>user
[user message + context block]
<|im_end|>
<|im_start|>assistant
<|tool_call_start|>search_foods(queries=["potatoes","bread"])<|tool_call_end|>
<|im_end|>
<|im_start|>tool
[{"name": "potatoes", "carbs_per_100g": 17.5}, ...]
<|im_end|>
<|im_start|>assistant
Here's what I found: potatoes at 17.5g carbs per 100g, bread at 49g per 100g...
<|im_end|>
```

Key elements:
- `<|startoftext|>` — LFM2.5 BOS/reset token; always included
- `<|im_start|>role\n...<|im_end|>` — ChatML turn delimiters
- `<|tool_call_start|>...<|tool_call_end|>` — native tool-call tokens (LFM2.5 has BFCLv3 score 49.12 pre-fine-tuning)
- Tool arguments use `key="value"` or `key=[...]` format (not JSON)

### Gemma format (Unsloth chat template)

```
<bos>
<|turn>system
[system prompt with tools + user settings]
<turn|>
<|turn>user
[user message + context block]
<turn|>
<|turn>model
search_foods({"queries": ["potatoes", "bread"]})
<turn|>
<|turn>model
Tool results:
search_foods -> [{"name": "potatoes", "carbs_per_100g": 17.5}, ...]
<turn|>
<|turn>model
Here's what I found: potatoes at 17.5g carbs per 100g, bread at 49g...
<turn|>
```

Key elements:
- `<bos>` — Gemma BOS token
- `<|turn>role\n...<turn|>` — Unsloth turn delimiters (not the Android LiteRT-LM format; that conversion happens at export time)
- Tool calls use `name({"key": "val"})` JSON-argument format
- **Tool results are rendered as model turns** starting with `Tool results:` — this matters for masking (see Section 2)

### BOS prefix check

Both training scripts run `check_bos_prefix()` to verify that pre-rendered text does not cause double-BOS when `add_special_tokens=True` is used. The check confirms the tokenizer appends no extra BOS token on top of the one already in the text. Current data: both models pass, no duplication.

---

## 2. Loss Function & Custom Masking

### Why not `train_on_responses_only`

TRL's `train_on_responses_only` identifies `response_part="<|im_start|>assistant\n"` and masks everything from assistant output through the next instruction start. In tool-calling conversations this includes tool results (which come after the assistant's tool-call turn):

```
assistant: <tool call>  ← trained (correct)
tool: {results}          ← ALSO trained (WRONG — leaks carb values into loss)
user: next message       ← masked
```

Tool results contain exact carb counts, tally intermediates, and final insulin doses. If trained on, the model learns to memorize numeric outputs rather than learn _which tool to call_. Our custom masking excludes all tool results.

### How custom masking works

`finetuning/common/masking.py` — three-step pipeline:

1. **Locate assistant spans** in raw text via regex:
   - LFM: `<|im_start|>assistant\n(...)<|im_end|>`
   - Gemma: `<|turn>model\n(...)<turn|>` — but skip spans where content starts with `Tool results:`

2. **Map character spans to token indices** using the tokenizer's `offset_mapping`

3. **Mask labels**: `labels[i] = input_ids[i]` for assistant tokens, `labels[i] = -100` everywhere else

**Result**: loss is computed **only** on the model's own decisions — tool call selections, parameter values, and natural language responses. System prompt, user messages, tool results, and template tokens all get `-100` labels (ignored by `CrossEntropyLoss(ignore_index=-100)`).

### Loss function

Standard cross-entropy (TRL SFTTrainer default). No custom loss — the masking handles exclusion. The effective training signal per conversation is typically 10-40% of total tokens (the rest are masked context).

---

## 3. Optimizer

### 8-bit AdamW (`optim="adamw_8bit"`)

| Property | Value |
|---|---|
| Implementation | `bitsandbytes` 8-bit AdamW |
| Memory vs 32-bit | ~25% (2 states × 1 byte vs 2 states × 4 bytes per param) |
| Convergence | Empirically equivalent to 32-bit AdamW for QLoRA fine-tuning |
| Weight decay | 0.01 (LFM) / 0.001 (Gemma) |

**Why 8-bit**: QLoRA already quantizes the base model to 4-bit NF4. Using 8-bit for the optimizer states keeps the entire pipeline (model weights + optimizer + gradients) under 8GB VRAM on an RTX 4060. The lower weight decay on Gemma reflects that it learns tool calling from scratch — less regularization needed to allow the model to adapt.

### Learning rate schedule

```
lr = 2e-4 × min(step / 100, 1.0) × (1 - step / 2223)  [linear warmup + linear decay]
```

- **`lr=2e-4`**: Standard for QLoRA — matches published Unsloth recipes for both LFM and Gemma fine-tuning. Lower than full fine-tuning (typically 5e-5) because LoRA adapters start at zero and need a stronger initial signal.
- **`warmup_steps=100`**: ~4.5% of total steps (2,223). Prevents gradient explosion from AdamW's initial momentum estimates being biased toward zero.
- **Linear decay to 0**: Simpler than cosine; sufficient for 3-epoch runs where the model converges well before the final step.

---

## 4. QLoRA Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `load_in_4bit` | `True` | NF4 quantization of frozen base model — 4× memory reduction |
| `r` (rank) | 16 | Standard LoRA rank; enough expressivity for 6k examples without overfitting |
| `alpha` | 32 | 2× rank — standard heuristic; controls LoRA contribution scale |
| `lora_dropout` | 0.0 | No dropout needed — low-rank constraint is its own regularizer |
| `bias` | `"none"` | No bias training — QLoRA adapters handle the shift |
| `use_gradient_checkpointing` | `"unsloth"` | Unsloth-optimized gradient checkpointing; trades ~20% compute for ~50% VRAM |

### Target modules

**LFM** (`FastLanguageModel.get_peft_model`):
```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "out_proj",   # attention
    "in_proj", "w1", "w2", "w3",                  # MLP
]
```
All attention projection + MLP linear layers. 9,142,272 trainable params of 1,179,482,880 total (0.78%).

**Gemma** (`FastModel.get_peft_model`):
```python
finetune_vision_layers=False,
finetune_language_layers=True,
finetune_attention_modules=True,
finetune_mlp_modules=True,
```
Unsloth's structured selector — equivalent to all Q/K/V/MLP linear layers in the language model. Vision layers frozen (not needed for text-only tool calling).

---

## 5. Training Hyperparameters

| Setting | LFM | Gemma | Rationale |
|---|---|---|---|
| `per_device_batch_size` | 2 | 1 | Gemma 2B vs LFM 1.2B — half micro-batch fits 8GB VRAM |
| `gradient_accumulation` | 4 | 8 | Same effective batch size 8 for both |
| `epochs` | 3 | 3 | ~18k effective examples — enough for domain adaptation, prevents overfitting |
| `max_seq_length` | 4096 | 4096 | Upgraded from plan's 2048; data p50=1276 p99=2543 p99.9=3086. 4096 covers all with margin. Padding-free via Unsloth means no memory penalty for excess capacity. |
| `seed` | 3407 | 3407 | Unsloth convention; reproducible shuffles |
| `save_steps` | 500 | 500 | Checkpoint every 500 steps (~22% of an epoch); 4-5 checkpoints per run |
| `eval_steps` | 500 | 500 | Eval loss on held-out 701 examples at same cadence |
| `save_total_limit` | 2 | 2 | Keep last 2 checkpoints to manage disk |
| `logging_steps` | 25 | 25 | Per-step logging to wandb/console every 25 steps |
| `report_to` | wandb/none | wandb/none | Auto-detected: wandb if `WANDB_API_KEY` env var is set, otherwise none |

### Sequence length rationale

The AI Integration Plan originally specified `max_seq_length=2048`. After sampling 500 training examples with actual tokenizers, we measured:

| Statistic | LFM |
|---|---|
| Mean | 1,301 |
| Median | 1,276 |
| P90 | 1,720 |
| P95 | 2,002 |
| P99 | 2,543 |
| P99.9 | 3,086 |
| Max | 3,086 |
| Exceed 2048 | ~0.6% |

The plan's 2048 would truncate ~0.6% of conversations. Moving to 4096 covers 100% with zero memory penalty — Unsloth's padding-free mode only allocates memory for actual token positions, not the full 4096.

### Unsloth optimizations (automatic)

| Optimization | Effect |
|---|---|
| Padding-free | No padding tokens computed; VRAM ≈ actual text length |
| Double buffering | Parallel H2D transfer + backward compute |
| Gradient offloading | Smart CPU offload for gradient checkpoints |
| Fused LoRA kernels | 2× faster forward/backward vs standard PEFT |

---

## 6. Post-Training Pipeline

### 6.1 Training outputs

Both scripts produce per `output_dir/{model}/`:

```
checkpoint-500/      ← intermediate (keeps last 2)
checkpoint-1000/
...
lora_adapter/        ← final LoRA weights + tokenizer
merged_16bit/        ← LoRA merged into base, saved as 16-bit safetensors
```

### 6.2 Evaluation

`evaluation/evaluate.py` replays the 701 eval conversations through the fine-tuned model using the mock harness for tool results:

```bash
python evaluation/evaluate.py \
  --checkpoint-dir finetuning/output/lfm/lora_adapter \
  --model-type lfm \
  --eval-path data/output/lfm/eval.jsonl
```

**Six metrics** (from AI Integration Plan Section 10.5):

| Metric | Weight | How measured |
|---|---|---|
| Tool call accuracy | 40% | % of individual tool calls with correct name + correct parameters |
| Sequence correctness | 25% | % of conversations ending with correct `final_result` (within 1%) |
| Clarification quality | 15% | % of ambiguous-food scenarios where model asks the right clarifying question |
| Natural language quality | 10% | Human rating 1-5 across all 10 languages |
| Latency | 5% | Time-to-first-token + tokens/sec on reference device |
| Memory | 5% | Peak RAM during 10-conversation stress test |

**Selection:** weighted score ≥ 0.70 passes. If both pass and scores are within 5pp, tiebreaker favors Gemma (no license revenue cap).

### 6.3 Export — LFM → GGUF

`export/convert_lfm_gguf.py` loads the LoRA adapter, merges into the base model, and quantizes:

```bash
python export/convert_lfm_gguf.py \
  --lora-dir finetuning/output/lfm/lora_adapter \
  --output-dir export/lfm \
  --quant q4_k_m
```

| Quant | File size (est.) | Use case |
|---|---|---|
| `q4_k_m` | ~719 MB | Production (Android llama.cpp) |
| `q5_k_m` | ~850 MB | Quality-sensitive (medium accuracy gain) |
| `q8_0` | ~1.1 GB | Max quality with 2× size |
| `f16` | ~2.4 GB | Reference/benchmark only |

Target runtime: llama.cpp via JNI on Android.

### 6.4 Export — Gemma → LiteRT-LM

`export/convert_gemma_litertlm.py` converts the merged_16bit model via Google's `litert_lm` CLI:

```bash
python export/convert_gemma_litertlm.py \
  --merged-dir finetuning/output/gemma/merged_16bit \
  --output-dir export/gemma \
  --quant int4
```

| Quant | File size (est.) | Use case |
|---|---|---|
| `int4` | <1 GB | Production (Android LiteRT-LM) |
| `int8` | ~2 GB | Higher quality; may not fit PAD size limits |

Target runtime: LiteRT-LM Android SDK (Google first-party). The export step also handles the Unsloth template → LiteRT-LM runtime format conversion (function call markers, escape tokens).

### 6.5 Device testing

Final verification on physical devices:
- **LFM + GGUF**: Samsung S25 Ultra (Snapdragon 8 Elite), Google Pixel 9 (Tensor G4)
- **Gemma + LiteRT-LM**: Pixel 9 (Tensor G4, NPU-accelerated), Samsung S25

Measure: TTFT <1s, throughput >10 tok/s, peak RAM <1.5 GB.

---

## 7. Reproducibility

All randomness is seeded at 3407 (model init, data shuffle, dropout). For exact reproduction:

```bash
# Install dependencies (CUDA 13.0, Python 3.12)
uv sync

# Train
python -m finetuning.lfm.train_lfm --seed 3407
python -m finetuning.gemma.train_gemma --seed 3407

# Evaluate
python evaluation/evaluate.py \
  --checkpoint-dir finetuning/output/lfm/lora_adapter \
  --model-type lfm \
  --eval-path data/output/lfm/eval.jsonl

# Export
python export/convert_lfm_gguf.py \
  --lora-dir finetuning/output/lfm/lora_adapter \
  --output-dir export/lfm --quant q4_k_m
```

Hardware: NVIDIA RTX 4060 8GB (local), A100 40GB (cloud, ~4-6h LFM, ~6-8h Gemma), or Colab T4 ~2-4h.
