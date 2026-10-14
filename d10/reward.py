"""IoU reward for GRPO (E5). Pure Python so it can be unit-tested without a GPU."""
from scripts.score import IOU_THR, iou_xyxy, parse_output

VALID_BONUS = 0.2


def gt_from_objects(objects):
    """Split-file objects (YOLO) -> [{"label", "bbox": (x1, y1, x2, y2)}] on the 0-1000 scale."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def reward(completion, gt_boxes):
    """0.0 for broken JSON; otherwise 0.2 + 0.8 * (sum of matched IoU) / max(n_gt, n_pred, 1).

    Dividing by max(n_gt, n_pred) punishes both missed boxes and invented boxes, so neither
    "always answer []" nor "spray many boxes" is a winning strategy.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted(
        ((iou_xyxy(g["bbox"], p["bbox"]), i, j)
         for i, g in enumerate(gt_boxes) for j, p in enumerate(pred) if g["label"] == p["label"]),
        reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:                 # greedy one-to-one matching, best IoU first
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return VALID_BONUS + (1 - VALID_BONUS) * total / max(len(gt_boxes), len(pred), 1)
