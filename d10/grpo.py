"""E5 (optional): GRPO with the IoU reward, starting from a finetuned adapter. Example:
    python -m d10.grpo --init work/runs/e1 --out work/runs/e5
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from d10.labels import PROMPT, objects_to_answer
from d10.modeling import PROJECTOR_FILE, load_config, load_finetuned, load_processor
from d10.reward import gt_from_objects, reward
from scripts.labcommon import read_jsonl


def completion_text(completion):
    """trl passes a chat completion as [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, str):
        return completion
    content = completion[0]["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content


def iou_reward(completions, gt, **kwargs):
    return [reward(completion_text(c), json.loads(g)) for c, g in zip(completions, gt)]


def build_dataset(rows, data_dir, tokenizer, max_answer_tokens):
    import datasets
    # A target longer than max_completion_length can never be generated in full: the cut-off
    # JSON would get reward 0 and push the model towards predicting fewer boxes.
    keep = [r for r in rows
            if len(tokenizer(objects_to_answer(r["objects"])).input_ids) <= max_answer_tokens]
    print(f"GRPO train images: {len(keep)}/{len(rows)} (answer fits in {max_answer_tokens} tokens)")
    ds = datasets.Dataset.from_dict({
        "prompt": [[{"role": "user", "content": PROMPT}] for _ in keep],     # trainer adds the image
        "image": [str(Path(data_dir) / r["image_path"]) for r in keep],
        "gt": [json.dumps(gt_from_objects(r["objects"])) for r in keep],
    })
    return ds.cast_column("image", datasets.Image())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True, help="SFT run dir to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/e5.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "run.json").exists() and not a.force:
        print(f"skip GRPO: {out}/run.json exists (use --force)")
        return

    from trl import GRPOConfig, GRPOTrainer
    g = load_config(a.config)["grpo"]
    init = json.loads((Path(a.init) / "run.json").read_text())
    processor = load_processor(init["model_id"])
    model = load_finetuned(init["model_id"], a.init, trainable=True)
    # Policy = the LLM LoRA only. A ViT LoRA (if the init run is E3) stays frozen, so memory
    # use with 4 completions per prompt does not depend on which SFT run was the best one.
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name and "language_model" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rows = read_jsonl(Path(a.data_dir) / "splits" / "train.jsonl")
    dataset = build_dataset(rows, a.data_dir, processor.tokenizer, g["max_completion_length"] - 8)
    use_cuda = torch.cuda.is_available()
    args = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=a.max_steps or g["max_steps"],
        learning_rate=g["learning_rate"], beta=g["beta"],
        num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        per_device_train_batch_size=g["num_generations"],     # one image, 4 completions per step
        gradient_accumulation_steps=1, use_vllm=False,
        fp16=use_cuda, bf16=False, gradient_checkpointing=True,
        logging_steps=1, save_strategy="no", report_to="none", seed=42,
        use_cpu=not use_cuda)
    trainer = GRPOTrainer(model=model, reward_funcs=iou_reward, args=args, train_dataset=dataset,
                          processing_class=processor)
    # For a quantised model trl casts every trainable adapter weight to bf16 inside __init__.
    # fp16 GradScaler cannot unscale bf16 gradients (and a T4 has no native bf16), so the adapter
    # goes back to float32, exactly as in SFT. The optimizer is only created inside train().
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    if (Path(a.init) / PROJECTOR_FILE).exists():              # projector is unchanged by GRPO
        shutil.copy2(Path(a.init) / PROJECTOR_FILE, out / PROJECTOR_FILE)
    logs = [h for h in trainer.state.log_history if "reward" in h]
    with open(out / "train_log.jsonl", "w") as f:
        for h in logs:
            f.write(json.dumps(h) + "\n")
    run = {"name": "E5", "model_id": init["model_id"], "init": init["name"], "config": g,
           "trainable_params": trainable, "train_images": len(dataset),
           "reward_first10": sum(h["reward"] for h in logs[:10]) / max(1, len(logs[:10])),
           "reward_last10": sum(h["reward"] for h in logs[-10:]) / max(1, len(logs[-10:])),
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0,
           "train_minutes": minutes}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GRPO done: {minutes:.1f} min, mean reward {run['reward_first10']:.3f} -> "
          f"{run['reward_last10']:.3f} (first/last 10 steps)")


if __name__ == "__main__":
    main()
