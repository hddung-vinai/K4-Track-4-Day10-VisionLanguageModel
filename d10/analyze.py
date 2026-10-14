"""Extra analysis for REPORT.md, computed from report/predictions/*.jsonl (not part of the
evidence written by scripts/score.py). Needs data/splits/ from prepare_data.py. Run:
    python -m d10.analyze            # add --params to count parameters per block (needs network)
Covers: per-class counts, GT box sizes, paired bootstrap of F1 differences, localisation vs
label errors, recall at several IoU thresholds, E0 failure modes, and the coordinate-scale trap.
"""
import sys

import json, random, statistics as st, collections
from scripts.labcommon import read_jsonl, CLASSES
from scripts.score import parse_output, _gt_boxes, _match, _prf, evaluate, iou_xyxy
test = read_jsonl("data/splits/test.jsonl")
P = {k: {r["image_id"]: r["raw"] for r in read_jsonl(f"report/predictions/{k}.jsonl")} for k in ("e0","e1","e2","e3","e4_data_size_0.1")}

print("== per-class P/R/F1 + counts")
for k in ("e0","e1","e2","e3","e4_data_size_0.1"):
    d = json.load(open(f"report/evidence/{k}.json"))
    print(k, {c: (round(v["precision"],3), round(v["recall"],3), round(v["f1"],3), v["tp"], v["fp"], v["fn"]) for c, v in d["per_class"].items()}, "micro", d["micro"]["tp"], d["micro"]["fp"], d["micro"]["fn"])

print("== GT box stats on test / train (area % of frame, median w,h in px)")
for name in ("train", "test"):
    rows = read_jsonl(f"data/splits/{name}.jsonl")
    for c in CLASSES:
        b = [o["yolo"] for r in rows for o in r["objects"] if o["label"] == c]
        print(name, c, len(b), "area%% median %.2f" % (100*st.median(w*h for _,_,w,h in b)), "w px %.0f h px %.0f" % (640*st.median(x[2] for x in b), 360*st.median(x[3] for x in b)),
              "aspect w/h median %.1f" % st.median((640*x[2])/(360*x[3]) for x in b if x[3] > 0), "zero-size:", sum(x[2] == 0 or x[3] == 0 for x in b))

def rows_for(k):
    out = []
    for e in test:
        parsed = parse_output(P[k].get(e["image_id"]))
        per, _ = _match(_gt_boxes(e), parsed or [])
        out.append(tuple(sum(v[i] for v in per.values()) for i in range(3)))
    return out
R = {k: rows_for(k) for k in P}
def f1(rows, idx): 
    tp = sum(rows[i][0] for i in idx); fp = sum(rows[i][1] for i in idx); fn = sum(rows[i][2] for i in idx)
    return _prf(tp, fp, fn)["f1"]
print("== paired bootstrap of F1 difference (2000 resamples of the 311 test images, seed 0)")
n = len(test)
for a, b in (("e1","e0"),("e2","e1"),("e3","e2"),("e3","e1"),("e3","e4_data_size_0.1"),("e4_data_size_0.1","e1")):
    rng = random.Random(0); ds = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(f1(R[a], idx) - f1(R[b], idx))
    ds.sort()
    print(f"{a} - {b}: point {f1(R[a], range(n)) - f1(R[b], range(n)):+.3f}  CI95 [{ds[50]:+.3f}, {ds[1949]:+.3f}]  P(diff<=0)={sum(d<=0 for d in ds)/2000:.3f}")

print("== localisation vs classification: for E1/E3, GT matched at IoU>=0.5 ignoring label")
for k in ("e0","e1","e3"):
    loc = cls_wrong = tot = 0; conf = collections.Counter()
    for e in test:
        gt = _gt_boxes(e); pr = parse_output(P[k].get(e["image_id"])) or []
        for g in gt:
            tot += 1
            best = max(pr, key=lambda p: iou_xyxy(g["bbox"], p["bbox"]), default=None)
            if best and iou_xyxy(g["bbox"], best["bbox"]) >= 0.5:
                loc += 1
                if best["label"] != g["label"]: cls_wrong += 1; conf[(g["label"], best["label"])] += 1
    print(k, "GT", tot, "localised(any label)", loc, "of which wrong label", cls_wrong, dict(conf))

print("== IoU threshold sensitivity (best IoU of same-label pred per GT) for E3: recall at 0.3/0.5/0.7")
for k in ("e1","e3"):
    best = []
    for e in test:
        pr = parse_output(P[k].get(e["image_id"])) or []
        for g in _gt_boxes(e):
            best.append(max((iou_xyxy(g["bbox"], p["bbox"]) for p in pr if p["label"] == g["label"]), default=0))
    print(k, {t: round(sum(b >= t for b in best)/len(best), 3) for t in (0.1, 0.3, 0.5, 0.7)})

print("== E0 raw analysis")
raws = [P["e0"][e["image_id"]] for e in test]
print("fenced:", sum("```" in r for r in raws), "len>=900 chars:", sum(len(r) > 900 for r in raws))
inv = [r for r in raws if parse_output(r) is None]
print("invalid:", len(inv)); 
for r in inv[:3]: print("  INV", repr(r[:160]), "...", repr(r[-60:]), len(r))
allb = [b for r in raws for b in (parse_output(r) or [])]
print("boxes", len(allb), "labels", collections.Counter(b["label"] for b in allb))
print("coords > 1000:", sum(max(b["bbox"]) > 1000 for b in allb), "full width (w>=900):", sum(b["bbox"][2]-b["bbox"][0] >= 900 for b in allb),
      "area>25%:", sum((b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1])/1e6 > .25 for b in allb))
print("median area pct pred e0: %.2f  e3: %.2f" % (st.median((b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1])/1e4 for b in allb),
      st.median((b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1])/1e4 for r in P["e3"].values() for b in (parse_output(r) or []))))
for e in test[:400:60]: print("  E0", e["image_id"], repr(P["e0"][e["image_id"]][:200]))
print("E3 invalid:", [ (k, repr(v[:120]), len(v)) for k, v in P["e3"].items() if parse_output(v) is None])

print("== reverse trap: E3 predictions converted to pixel, scored as norm1000")
conv = {}
for e in test:
    pr = parse_output(P["e3"][e["image_id"]]) or []
    conv[e["image_id"]] = json.dumps([{"label": p["label"], "bbox_2d": [round(p["bbox"][0]*e["width"]/1000), round(p["bbox"][1]*e["height"]/1000), round(p["bbox"][2]*e["width"]/1000), round(p["bbox"][3]*e["height"]/1000)]} for p in pr if round(p["bbox"][2]*e["width"]/1000) > round(p["bbox"][0]*e["width"]/1000) and round(p["bbox"][3]*e["height"]/1000) > round(p["bbox"][1]*e["height"]/1000)])
r = evaluate(test, conv, coord="norm1000"); print("pixel output scored as norm1000: F1 %.3f valid %.3f" % (r["micro"]["f1"], r["parse_valid_rate"]))
r = evaluate(test, conv, coord="pixel"); print("pixel output scored as pixel: F1 %.3f" % r["micro"]["f1"])
print("== loss logs")
for k in ("e1","e2","e3","e4_data_size_0.1"):
    L = read_jsonl(f"report/logs/{k}/train_log.jsonl"); print(k, [(r["step"], r["loss"]) for r in L if r["step"] in (10, 50, 100, 200, 300)], "vram", L[-1]["vram_gb"], "min", L[-1]["minutes"])


if "--params" in sys.argv:
    # exact tensor shapes from the checkpoint's safetensors headers; no weights are downloaded
    import math
    from huggingface_hub import get_safetensors_metadata
    counts = collections.Counter()
    for f in get_safetensors_metadata("Qwen/Qwen3-VL-4B-Instruct").files_metadata.values():
        for name, t in f.tensors.items():
            if "visual.merger" in name: key = "projector: visual.merger"
            elif "visual.deepstack_merger_list" in name: key = "projector: visual.deepstack_merger_list"
            elif "visual." in name: key = "ViT (blocks, patch/pos embed)"
            elif "embed_tokens" in name: key = "LLM token embedding (tied with lm_head)"
            else: key = "LLM layers"
            counts[key] += math.prod(t.shape)
    print("== parameters per block")
    for key, v in counts.items():
        print(f"{key:42s} {v:>14,}")
    print(f"{'total':42s} {sum(counts.values()):>14,}")
