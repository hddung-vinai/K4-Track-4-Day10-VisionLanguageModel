"""Draw ground-truth and predicted boxes, and plot training loss. Example:
    python -m d10.viz --pred work/e1.jsonl --name E1
writes report/figures/E1_{good,bad}_*.jpg and prints one caption line per figure.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from d10.labels import box_to_pixels, yolo_to_box
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, parse_output

GT_COLOR, PRED_COLOR = (0, 220, 0), (255, 40, 40)


def draw(image, gt=(), pred=(), caption=None):
    """gt / pred: [{"label", "bbox": (x1, y1, x2, y2)}] on the 0-1000 scale. GT green, prediction red."""
    image = image.convert("RGB").copy()
    d = ImageDraw.Draw(image)
    w, h = image.size
    for boxes, color, below in ((gt, GT_COLOR, False), (pred, PRED_COLOR, True)):
        for b in boxes:
            x1, y1, x2, y2 = box_to_pixels(b["bbox"], w, h)
            d.rectangle((x1, y1, x2, y2), outline=color, width=2)
            d.text((x1 + 2, y2 + 1 if below else max(0, y1 - 11)), b["label"], fill=color)
    if caption:
        d.rectangle((0, 0, w, 12), fill=(0, 0, 0))
        d.text((3, 0), caption, fill=(255, 255, 255))
    return image


def gt_as_seen_by_model(row):
    """GT boxes after the label conversion of step 2 (what the model is trained on)."""
    return [{"label": o["label"], "bbox": tuple(yolo_to_box(o["label"], o["yolo"])["bbox_2d"])}
            for o in row["objects"]]


def per_image(rows, preds):
    out = []
    for r in rows:
        parsed = parse_output(preds.get(r["image_id"]))
        per, _ = _match(_gt_boxes(r), parsed or [])
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        out.append({"row": r, "pred": parsed, "tp": tp, "fp": fp, "fn": fn})
    return out


def pick(stats, n_good, n_bad):
    """Good: every box right (most boxes first). Bad: most errors first; broken JSON counts too."""
    good = sorted((s for s in stats if s["tp"] and not s["fp"] and not s["fn"]),
                  key=lambda s: (-s["tp"], s["row"]["image_id"]))[:n_good]
    bad = sorted((s for s in stats if s["fp"] + s["fn"]),
                 key=lambda s: (-(s["fp"] + s["fn"]), s["row"]["image_id"]))[:n_bad]
    return good, bad


def save_cases(pred_path, name, out_dir="report/figures", data_dir="data", split="test",
               n_good=4, n_bad=4):
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    stats = [s for s in per_image(rows, preds) if s["row"]["image_id"] in preds]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for kind, group in zip(("good", "bad"), pick(stats, n_good, n_bad)):
        for i, s in enumerate(group, 1):
            r = s["row"]
            note = "" if s["pred"] is not None else " INVALID-JSON"
            caption = f"{name} TP={s['tp']} FP={s['fp']} FN={s['fn']}{note} | green=GT red=pred"
            img = draw(Image.open(Path(data_dir) / r["image_path"]), _gt_boxes(r), s["pred"] or [], caption)
            path = out_dir / f"{name}_{kind}_{i}.jpg"
            img.save(path, quality=90)
            lines.append(f"{path} | {r['image_id']} | TP={s['tp']} FP={s['fp']} FN={s['fn']}{note}")
    print("\n".join(lines))
    return lines


def plot_loss(logs_dir="report/logs", out="report/figures/loss.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for log in sorted(Path(logs_dir).glob("*/train_log.jsonl")):
        recs = [r for r in read_jsonl(log) if "loss" in r and "step" in r]
        if recs:
            ax.plot([r["step"] for r in recs], [r["loss"] for r in recs], label=log.parent.name)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss (mean of 10 steps)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred")
    ap.add_argument("--name")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--loss", action="store_true", help="plot training loss of all runs")
    a = ap.parse_args(argv)
    if a.pred:
        save_cases(a.pred, a.name or Path(a.pred).stem.upper(), a.out, a.data_dir)
    if a.loss:
        plot_loss(out=str(Path(a.out) / "loss.png"))


if __name__ == "__main__":
    main()
