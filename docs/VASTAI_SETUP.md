# Vast.ai VM Setup & Training Guide

How to provision a Vast.ai GPU instance, set up the Mininio fine-tuning environment,
and run LFM 2.5 and Gemma 4 training.

---

## 1. Prerequisites

- [Vast.ai account](https://cloud.vast.ai/) with an **SSH public key** added
  (`Account` → `SSH Keys` → paste your `~/.ssh/id_ed25519.pub`)
- HuggingFace token (`HF_TOKEN`) for model downloads
- Weights & Biases API key (`WANDB_API_KEY`) for training metrics (optional)

If you haven't generated an SSH key yet:
```bash
ssh-keygen -t ed25519 -C "vastai" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub   # paste into Vast.ai account
```

---

## 2. GPU Selection

| Minimum | Recommended |
|---------|-------------|
| V100 16 GB | A100 40 GB |
| 24 GB RAM | 48 GB RAM |
| 32 GB disk | 64 GB disk |

> [!WARNING]
> **Do NOT rent a machine labelled "CUDA 13.0".** CUDA 13 dropped support for
> compute capability 7.0 (V100) and will cause `torch` to fail with a
> capability mismatch warning. Stick to instances with CUDA 12.x.

V100 is the sweet spot: cheap (~$0.12/hr), 16 GB VRAM fits both models with
QLoRA 4-bit, and fp16 tensor cores give ~2× training speed vs T4.

Cost estimate for a full run (both models, 3 epochs each):
- V100: **~$0.40** (~3.5 hrs total)
- A100: **~$1.50** (~2 hrs total, faster)

---

## 3. Rent & Connect

1. On [cloud.vast.ai](https://cloud.vast.ai/), filter by `GPU: Tesla V100`, sort by price.
2. Pick an instance with CUDA 12.x, 16+ GB VRAM, 24+ GB RAM, 32+ GB disk.
3. Click **Rent** → copy the SSH command from the Connect modal.

```bash
ssh -p <port> root@<ip>
```

### Optional: forward W&B dashboard
```bash
ssh -p <port> root@<ip> -L 8080:localhost:8080
```

---

## 4. One-Shot Setup

Paste the entire block after SSH-ing in. This installs everything, clones the
repo, sets up the virtualenv, copies your `.env`, and syncs training data.

```bash
# --- system deps ---
apt update && apt install -y -qq git tmux python3-venv

# --- clone & venv ---
git clone https://github.com/CoGian/mininio-ai-finetuning.git
cd mininio-ai-finetuning
python3 -m venv .venv && source .venv/bin/activate
pip install uv && uv sync

# --- .env (edit now) ---
cp .env.example .env
nano .env
# Fill in:
#   HF_TOKEN=hf_...
#   WANDB_API_KEY=...
#   WANDB_PROJECT=mininio-ai-finetuning
#   GOOGLE_API_KEY=...  (only if regenerating data)

# --- data ---
# Option A: generate fresh (requires GOOGLE_API_KEY, costs ~$12)
# python data/generate.py --count-per-lang 800 --languages all --log-file
# python data/assemble.py

# Option B: upload from your local machine
# On your machine:
#   scp -P <port> -r data/output root@<ip>:/root/mininio-ai-finetuning/data/
python data/assemble.py
```

---

## 5. Train

Use `nohup` so training survives SSH disconnect. Logs go to `lfm-train.log`
and `gemma-train.log`.

### LFM 2.5-1.2B (~50–70 min on V100)

```bash
cd /root/mininio-ai-finetuning && source .venv/bin/activate

nohup python -m finetuning.lfm.train_lfm \
  --data-dir data/output \
  --output-dir finetuning/output/lfm \
  --batch-size 16 \
  --grad-accum 4 \
  --epochs 3 \
  --max-seq-length 4096 \
  --fp16 \
  > lfm-train.log 2>&1 &

tail -f lfm-train.log
```

### Gemma 4 E2B (~2–2.5 hrs on V100)

```bash
cd /root/mininio-ai-finetuning && source .venv/bin/activate

nohup python -m finetuning.gemma.train_gemma \
  --data-dir data/output \
  --output-dir finetuning/output/gemma \
  --batch-size 12 \
  --grad-accum 4 \
  --epochs 3 \
  --max-seq-length 4096 \
  --fp16 \
  > gemma-train.log 2>&1 &

tail -f gemma-train.log
```

### If VRAM runs out (OOM)

Reduce `--batch-size` and increase `--grad-accum` to keep the same effective
batch size:

| LFM effective batch | Command tweak |
|---------------------|---------------|
| 64 (default) | `--batch-size 16 --grad-accum 4` |
| 64 (fallback) | `--batch-size 8 --grad-accum 8` |
| 64 (minimal) | `--batch-size 4 --grad-accum 16` |

| Gemma effective batch | Command tweak |
|-----------------------|---------------|
| 48 (default) | `--batch-size 12 --grad-accum 4` |
| 48 (fallback) | `--batch-size 8 --grad-accum 6` |
| 48 (minimal) | `--batch-size 4 --grad-accum 12` |

All configurations produce identical weight updates thanks to Unsloth's
gradient accumulation fix.

### Check progress

```bash
# Is it still running?
ps aux | grep train_

# View last 20 lines
tail -20 lfm-train.log

# Watch GPU usage
watch -n 2 nvidia-smi

# Disk space
df -h /
```

---

## 6. Retrieve Results

Once training completes, the outputs are at:

```
finetuning/output/lfm/
  lora_adapter/        ← PEFT adapter weights
  merged_16bit/        ← merged full model (if save succeeded)
  checkpoints/         ← intermediate checkpoints

finetuning/output/gemma/
  lora_adapter/
  merged_16bit/
  checkpoints/
```

Copy them back to your local machine:

```bash
# From your local machine — replace <port> and <ip>
scp -P <port> -r root@<ip>:/root/mininio-ai-finetuning/finetuning/output/lfm ./finetuning/output/
scp -P <port> -r root@<ip>:/root/mininio-ai-finetuning/finetuning/output/gemma ./finetuning/output/
```

---

## 7. Troubleshooting

### `RuntimeError: operator torchvision::nms does not exist`

Your VM has CUDA 13.0. The repo targets CUDA 12.8 (`cu128`). Either pick a
different instance (CUDA 12.x) or install PyTorch for your CUDA version
manually — but CUDA 13 dropped V100 support entirely.

### `UserWarning: Found GPU0 Tesla V100 which is of cuda capability 7.0`

Same root cause — CUDA 13 dropped SM 7.0. Switch to a CUDA 12.x instance.

### `CUDA out of memory`

Reduce `--batch-size` (see fallback table above). The `--fp16` flag is
already enabled — it halves activation VRAM on V100.

### Training stalled / no log output

The first model download from HuggingFace can take 5–10 minutes with no
output. Be patient. If it exceeds 15 minutes, kill and restart.

### SSH disconnects mid-training

That's what `nohup` is for — training keeps running. Just SSH back in and
run `tail -f lfm-train.log` to reattach to the output.
