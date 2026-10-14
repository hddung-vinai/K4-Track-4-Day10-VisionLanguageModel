"""Tests for my own pure-Python code (labels and the GRPO reward). Run: python -m pytest tests_d10 -q"""
import json

import pytest

from d10.grpo import completion_text
from d10.labels import box_to_pixels, objects_to_answer, yolo_to_box
from d10.reward import gt_from_objects, reward
from scripts.score import evaluate, parse_output


def _box(label, x1, y1, x2, y2):
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def _gt(label, x1, y1, x2, y2):
    return {"label": label, "bbox": (x1, y1, x2, y2)}


GT = [_gt("pothole", 400, 400, 600, 600)]
PERFECT = json.dumps([_box("pothole", 400, 400, 600, 600)])


# ---- labels -------------------------------------------------------------------------------

def test_yolo_to_box_scale_and_roundtrip_to_pixels():
    b = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.2])
    assert b == _box("pothole", 400, 400, 600, 600)
    assert box_to_pixels(b["bbox_2d"], 640, 360) == pytest.approx((256, 144, 384, 216))


def test_yolo_to_box_clamps_and_never_degenerates():
    assert yolo_to_box("crack", [0.99, 0.01, 0.2, 0.2])["bbox_2d"] == [890, 0, 1000, 110]
    for yolo in ([0.5, 0.5, 0.0001, 0.0001], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]):
        x1, y1, x2, y2 = yolo_to_box("crack", yolo)["bbox_2d"]
        assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def test_objects_to_answer_is_sorted_compact_and_parseable():
    objs = [{"label": "crack", "yolo": [0.5, 0.8, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.7, 0.2, 0.1, 0.1]},
            {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    ans = objects_to_answer(objs)
    assert " " not in ans
    assert [b["label"] for b in json.loads(ans)] == ["pothole", "manhole", "crack"]
    assert len(parse_output(ans)) == 3
    assert objects_to_answer([]) == "[]"


def test_answer_scores_f1_one_with_the_lab_scorer():
    entry = {"image_id": "a", "width": 640, "height": 360,
             "objects": [{"label": "pothole", "yolo": [0.31234, 0.4567, 0.0931, 0.0412]},
                         {"label": "crack", "yolo": [0.7, 0.6, 0.25, 0.05]}]}
    r = evaluate([entry], {"a": objects_to_answer(entry["objects"])}, n_boot=10)
    assert r["micro"]["f1"] == 1.0 and r["mean_iou"] > 0.9


# ---- reward -------------------------------------------------------------------------------

def test_reward_perfect_is_one():
    assert reward(PERFECT, GT) == pytest.approx(1.0)


@pytest.mark.parametrize("raw", ["", "not json", '[{"label":"pothole","bbox_2d":[1,2,3]}]',
                                 '[{"label":"pothole","bbox_2d":[400,400,600'])
def test_reward_broken_json_is_zero(raw):
    assert reward(raw, GT) == 0.0


def test_reward_wrong_class_only_gets_valid_json_bonus():
    assert reward(json.dumps([_box("crack", 400, 400, 600, 600)]), GT) == pytest.approx(0.2)


def test_reward_extra_box_is_penalised_through_the_denominator():
    raw = json.dumps([_box("pothole", 400, 400, 600, 600), _box("pothole", 10, 10, 50, 50)])
    assert reward(raw, GT) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_reward_duplicate_box_matches_only_once():
    raw = json.dumps([_box("pothole", 400, 400, 600, 600)] * 3)
    assert reward(raw, GT) == pytest.approx(0.2 + 0.8 * 1 / 3)


def test_reward_empty_cases():
    assert reward("[]", []) == 1.0                      # nothing there, nothing predicted
    assert reward("[]", GT) == pytest.approx(0.2)       # missed everything: valid JSON only
    assert reward(PERFECT, []) == pytest.approx(0.2)    # invented a box on a clean image


def test_reward_low_iou_does_not_count_and_partial_iou_scales():
    assert reward(json.dumps([_box("pothole", 550, 550, 750, 750)]), GT) == pytest.approx(0.2)
    half = json.dumps([_box("pothole", 400, 400, 600, 550)])      # IoU 0.75
    assert reward(half, GT) == pytest.approx(0.2 + 0.8 * 0.75)


def test_reward_giant_box_is_not_a_shortcut():
    assert reward(json.dumps([_box("pothole", 0, 0, 1000, 1000)]), GT) == pytest.approx(0.2)


def test_reward_missing_one_of_two():
    gt = GT + [_gt("crack", 100, 100, 200, 200)]
    assert reward(PERFECT, gt) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_reward_is_bounded():
    for raw in (PERFECT, "[]", "x", json.dumps([_box("pothole", 0, 0, 1000, 1000)] * 5)):
        for gt in ([], GT):
            assert 0.0 <= reward(raw, gt) <= 1.0


def test_gt_from_objects_matches_scale():
    gt = gt_from_objects([{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}])
    assert gt[0]["label"] == "pothole" and gt[0]["bbox"] == pytest.approx((400, 400, 600, 600))


def test_completion_text_accepts_chat_and_plain_formats():
    assert completion_text("[]") == "[]"
    assert completion_text([{"role": "assistant", "content": "[]"}]) == "[]"
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "[]"}]}]) == "[]"
