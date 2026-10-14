"""Summary table from report/evidence/*.json plus simple statistics of the raw outputs.
    python -m d10.summary            # prints markdown, writes report/summary.md
"""
import argparse
import json
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import parse_output

HUGE_BOX = 0.25      # a box covering more than 25% of the frame (median GT box is about 0.7%)


def evidence_table(ev_dir):
    head = ["Mã", "Tham số huấn luyện", "VRAM đỉnh (GB)", "Phút", "valid", "P", "R", "F1", "CI95",
            "FP/ảnh", "mean IoU", *[f"F1 {c}" for c in CLASSES]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p in sorted(Path(ev_dir).glob("e*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m, meta = d["micro"], d.get("meta", {})
        row = [p.stem, f"{meta.get('trainable_params', 0):,}", meta.get("peak_vram_gb", ""),
               meta.get("train_minutes", ""), f"{d['parse_valid_rate']:.3f}",
               f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
               f"{d['f1_ci95'][0]:.3f}–{d['f1_ci95'][1]:.3f}", f"{d['fp_per_image']:.2f}",
               f"{d['mean_iou']:.3f}", *[f"{d['per_class'][c]['f1']:.3f}" for c in CLASSES]]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def output_stats(pred_path):
    """How the model answers, independent of the ground truth: used to spot degenerate
    behaviour (always "[]", one giant box) and to describe E0's failure modes."""
    rows = read_jsonl(pred_path)
    n = len(rows)
    invalid = empty = boxes = huge = unknown = 0
    for r in rows:
        parsed = parse_output(r["raw"])
        if parsed is None:
            invalid += 1
            continue
        empty += not parsed
        boxes += len(parsed)
        for b in parsed:
            x1, y1, x2, y2 = b["bbox"]
            huge += (x2 - x1) * (y2 - y1) / 1e6 > HUGE_BOX
            unknown += b["label"] not in CLASSES
    return {"n": n, "invalid_json": invalid, "empty_list": empty, "boxes_per_image": boxes / max(n, 1),
            "huge_box_frac": huge / max(boxes, 1), "unknown_label_frac": unknown / max(boxes, 1)}


def stats_table(pred_dir):
    head = ["Mã", "ảnh", "JSON hỏng", "trả []", "box/ảnh", "tỉ lệ box >25% khung", "tỉ lệ nhãn lạ"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p in sorted(Path(pred_dir).glob("e*.jsonl")):
        s = output_stats(p)
        lines.append(f"| {p.stem} | {s['n']} | {s['invalid_json']} | {s['empty_list']} | "
                     f"{s['boxes_per_image']:.2f} | {s['huge_box_frac']:.3f} | {s['unknown_label_frac']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="report")
    a = ap.parse_args(argv)
    report = Path(a.report)
    text = ("## Kết quả trên test (từ report/evidence)\n\n" + evidence_table(report / "evidence")
            + "\n\n## Thống kê output thô (từ report/predictions)\n\n"
            + stats_table(report / "predictions") + "\n")
    (report / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
