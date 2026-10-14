"""Student code for Lab Day 10 (Qwen3-VL-4B LoRA grounding). Tools in scripts/ are left untouched."""
import os

# Kaggle gives two T4s. Every run uses exactly one GPU: otherwise transformers.Trainer (E5)
# wraps the model in DataParallel, and peak-VRAM numbers stop being comparable between runs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Fewer "out of memory" errors caused by fragmentation when sequence lengths vary between batches.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
