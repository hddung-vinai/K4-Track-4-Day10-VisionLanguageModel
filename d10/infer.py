"""Batched greedy inference -> predictions.jsonl ({"image_id", "raw"} per line). Examples:
    python -m d10.infer --out work/e0.jsonl                          # zero-shot
    python -m d10.infer --run work/runs/e1 --out work/e1.jsonl       # finetuned
An interrupted run resumes: image_ids already in --out are skipped.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from d10.data import CHAT_KW, user_message
from d10.modeling import autocast, load_base, load_finetuned, load_processor
from scripts.labcommon import read_jsonl

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


@torch.no_grad()
def generate_batch(model, processor, images, max_new_tokens):
    conversations = [[user_message(img)] for img in images]
    inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, processor_kwargs={"padding": True},
        **CHAT_KW).to(model.device)
    with autocast():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]          # drop the (left-padded) prompt
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def predict(model, processor, rows, out_path, data_dir="data", batch_size=8, max_new_tokens=512):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["image_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go -> {out_path}")
    model.eval()
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            images = [Image.open(Path(data_dir) / r["image_path"]).convert("RGB") for r in chunk]
            for r, raw in zip(chunk, generate_batch(model, processor, images, max_new_tokens)):
                f.write(json.dumps({"image_id": r["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // batch_size) % 5 == 0:
                print(f"  {i + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s  last: {raw[:90]!r}",
                      flush=True)
    if todo:
        print(f"inference: {(time.time() - t0) / len(todo):.2f} s/image")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run dir with adapter (+ projector.pt); omit for zero-shot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, help="only the first N images (debugging)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args(argv)

    rows = read_jsonl(Path(a.data_dir) / "splits" / f"{a.split}.jsonl")[:a.limit]
    if Path(a.out).exists() and {r["image_id"] for r in rows} <= {r["image_id"] for r in read_jsonl(a.out)}:
        print(f"skip inference: {a.out} already has all {len(rows)} images")
        return
    model_id = a.model_id
    if a.run:
        model_id = json.loads((Path(a.run) / "run.json").read_text())["model_id"]
    processor = load_processor(model_id)
    model = load_finetuned(model_id, a.run) if a.run else load_base(model_id)
    predict(model, processor, rows, a.out, a.data_dir, a.batch_size, a.max_new_tokens)


if __name__ == "__main__":
    main()
