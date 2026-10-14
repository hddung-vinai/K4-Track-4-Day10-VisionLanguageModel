"""Supervised finetuning (E1-E4). Example:
    python -m d10.train --config configs/e1.yaml --out work/runs/e1
"""
import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch

from d10.data import GroundingDataset, collate
from d10.modeling import apply_setup, autocast, load_base, load_config, load_processor, save_trained
from scripts.labcommon import read_jsonl

WARMUP_FRAC = 0.05
SAVE_EVERY = 100
LOG_EVERY = 10


def lr_factor(step, total):
    """Linear warmup over the first 5% of steps, then cosine decay to 0."""
    warmup = max(1, int(WARMUP_FRAC * total))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def endless(loader):
    while True:
        yield from loader


def train(cfg, out_dir, data_dir="data", num_workers=2, model=None, processor=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    rows = read_jsonl(Path(data_dir) / "splits" / "train.jsonl")
    n_all = len(rows)
    fraction = cfg.get("train_fraction", 1.0)
    if fraction < 1.0:                               # E4 data_size: fixed random subset of images
        rows = random.Random(seed).sample(rows, max(1, round(fraction * n_all)))
    print(f"train images: {len(rows)}/{n_all}")

    processor = processor or load_processor(cfg["model_id"])
    model = model or load_base(cfg["model_id"])
    model, info = apply_setup(model, cfg)
    print(f"LoRA modules: {info['lora_modules']} | trainable params: {info['trainable_params']:,} "
          f"(LoRA {info['lora_params']:,} + projector {info['projector_params']:,})")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    loader = torch.utils.data.DataLoader(
        GroundingDataset(rows, data_dir, processor), batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0, generator=torch.Generator().manual_seed(seed),
        collate_fn=partial(collate, pad_token_id=processor.tokenizer.pad_token_id))
    batches = endless(loader)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    max_steps, accum = cfg["max_steps"], cfg["grad_accum"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_factor, total=max_steps))
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    device = next(model.parameters()).device

    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    t0, running = time.time(), 0.0
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(accum):                       # gradient accumulation: accum x batch_size images
            batch = {k: v.to(device) for k, v in next(batches).items()}
            with autocast():
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}: possible fp16 overflow, lower lr")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        running += step_loss
        if step % LOG_EVERY == 0 or step == max_steps:
            n = LOG_EVERY if step % LOG_EVERY == 0 else step % LOG_EVERY
            rec = {"step": step, "loss": round(running / n, 4), "lr": sched.get_last_lr()[0],
                   "minutes": round((time.time() - t0) / 60, 2),
                   "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if use_cuda else 0.0}
            running = 0.0
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if step % SAVE_EVERY == 0 and step < max_steps:
            save_trained(model, out_dir)             # survive a killed session

    minutes = (time.time() - t0) / 60
    save_trained(model, out_dir)
    run = {**info, "name": cfg["name"], "model_id": cfg["model_id"], "config": cfg,
           "train_images": len(rows), "final_loss": rec["loss"],
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0,
           "train_minutes": minutes,
           "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu"}
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {minutes:.1f} min, peak VRAM {run['peak_vram_gb']:.2f} GB -> {out_dir}")
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-fraction", type=float, help="E4 data_size: fraction of train images")
    ap.add_argument("--lora-r", type=int, help="E4 rank: sets r and alpha = 2*r")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch-size", type=int, help="with --grad-accum, use 1 and 8 if out of memory")
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--name")
    ap.add_argument("--force", action="store_true", help="retrain even if run.json exists")
    a = ap.parse_args(argv)

    if (Path(a.out) / "run.json").exists() and not a.force:
        print(f"skip training: {a.out}/run.json exists (use --force to retrain)")
        return
    cfg = load_config(a.config)
    for key, val in (("train_fraction", a.train_fraction), ("max_steps", a.max_steps),
                     ("batch_size", a.batch_size), ("grad_accum", a.grad_accum), ("name", a.name)):
        if val is not None:
            cfg[key] = val
    if a.lora_r is not None:
        cfg["lora"] = {**cfg["lora"], "r": a.lora_r, "alpha": 2 * a.lora_r}
    train(cfg, a.out, a.data_dir)


if __name__ == "__main__":
    main()
