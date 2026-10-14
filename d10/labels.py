"""Prompt and label conversion: YOLO (normalised centre/size) -> JSON on the 0-1000 scale. Pure Python."""
import json

SCALE = 1000

PROMPT = (
    'Detect road damage in this image. The classes are "pothole", "crack" and "manhole". '
    'Answer with ONLY a JSON list, one item per object, in the form '
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
    "Coordinates are integers on a 0-1000 scale relative to the image width and height "
    "(not pixels). Answer [] if there is no damage."
)


def yolo_to_box(label, yolo):
    """One YOLO box [xc, yc, w, h] in [0,1] -> {"label", "bbox_2d": [x1, y1, x2, y2]} in 0-1000."""
    xc, yc, w, h = yolo
    raw = ((xc - w / 2), (yc - h / 2), (xc + w / 2), (yc + h / 2))
    x1, y1, x2, y2 = (min(max(int(round(v * SCALE)), 0), SCALE) for v in raw)
    # A very thin box can collapse to zero width/height after rounding. score.py rejects the
    # WHOLE output when any box has x2 <= x1, so never teach the model a degenerate box.
    if x2 <= x1:
        x1, x2 = (x1, x1 + 1) if x1 < SCALE else (SCALE - 1, SCALE)
    if y2 <= y1:
        y1, y2 = (y1, y1 + 1) if y1 < SCALE else (SCALE - 1, SCALE)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects):
    """All objects of one image -> the target string the model must generate ("[]" if none)."""
    boxes = [yolo_to_box(o["label"], o["yolo"]) for o in objects]
    # top-to-bottom, left-to-right: a stable order, so the same image always has the same target
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixels(bbox_2d, width, height):
    """Inverse mapping, used to draw boxes and to check the conversion round-trips."""
    x1, y1, x2, y2 = bbox_2d
    return (x1 / SCALE * width, y1 / SCALE * height, x2 / SCALE * width, y2 / SCALE * height)
