"""Run experiments end to end: train -> infer on test -> score with scripts/score.py. Examples:
    python -m d10.pipeline e0
    python -m d10.pipeline e1 e2 e3         # add --stage train / --stage eval to split the work
    python -m d10.pipeline e4               # ablation data_size (10% vs 100%) on the best of E1-E3
    python -m d10.pipeline e5               # GRPO from the best of E1-E3
    python -m d10.pipeline e0 e1 e3 --smoke # 2 steps, 8 images: only checks that the code runs
Every stage runs in its own process (the GPU is fully released in between) and is skipped when
its output already exists, so the same command resumes after a crash.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from d10.modeling import load_config

E4_FRACTION = 0.1
SMOKE_STEPS, SMOKE_IMAGES = 2, 8


class Pipeline:
    def __init__(self, smoke=False, base=None, stage="all", batch=None):
        self.smoke, self.base, self.stage, self.batch = smoke, base, stage, batch
        self.work = Path("work/smoke") if smoke else Path("work")
        # smoke results must never land in report/: they are not real evidence
        self.report = self.work / "report" if smoke else Path("report")
        self.evidence = self.report / "evidence"

    def sh(self, *cmd):
        cmd = [sys.executable, *map(str, cmd)]
        print("\n$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    def run_dir(self, key):
        return self.work / "runs" / key

    def pred(self, key):
        return self.work / f"{key}.jsonl"

    def infer(self, key, run=None, max_new_tokens=None, model_id=None):
        cmd = ["-m", "d10.infer", "--out", self.pred(key)]
        if run:
            cmd += ["--run", run]
        if model_id:
            cmd += ["--model-id", model_id]
        if max_new_tokens:
            cmd += ["--max-new-tokens", max_new_tokens]
        if self.smoke:
            cmd += ["--limit", SMOKE_IMAGES]
        self.sh(*cmd)

    def score(self, key, name, run=None, pred=None, extra=()):
        """Evidence is always written by the lab's own scripts/score.py."""
        cmd = ["scripts/score.py", "--pred", pred or self.pred(key), "--name", name,
               "--out", self.evidence / f"{key}.json"]
        if run:
            r = json.loads((Path(run) / "run.json").read_text())
            cmd += ["--meta", f"trainable_params={int(r['trainable_params'])}",
                    "--meta", f"peak_vram_gb={r['peak_vram_gb']:.2f}",
                    "--meta", f"train_minutes={r['train_minutes']:.2f}",
                    "--meta", f"train_images={int(r['train_images'])}"]
        for item in extra:
            cmd += ["--meta", item]
        self.sh(*cmd)
        self.archive(key, run, pred or self.pred(key))

    def archive(self, key, run, pred):
        """Keep the small text logs next to the evidence (weights stay in work/)."""
        dst = self.report / "predictions" / f"{key}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred, dst)
        if run:
            logs = self.report / "logs" / key
            logs.mkdir(parents=True, exist_ok=True)
            for f in ("run.json", "train_log.jsonl"):
                if (Path(run) / f).exists():
                    shutil.copy2(Path(run) / f, logs / f)

    def train_args(self):
        args = ["--max-steps", SMOKE_STEPS] if self.smoke else []
        if self.batch:                       # out-of-memory fallback, same effective batch of 8
            args += ["--batch-size", self.batch[0], "--grad-accum", self.batch[1]]
        return args

    def best_base(self):
        """Best of E1-E3 by micro F1 on test (override with --base)."""
        if self.base:
            return self.base
        f1 = {}
        for key in ("e1", "e2", "e3"):
            p = self.evidence / f"{key}.json"
            if p.exists():
                f1[key] = json.loads(p.read_text())["micro"]["f1"]
        if not f1:
            raise SystemExit("run e1-e3 first (or pass --base)")
        best = max(f1, key=f1.get)
        print(f"base setup: {best} (F1 {f1})")
        return best

    def e0(self):
        cfg = load_config("configs/e0.yaml")
        self.infer("e0", max_new_tokens=cfg["max_new_tokens"], model_id=cfg["model_id"])
        self.score("e0", "E0")

    def sft(self, key):
        run = self.run_dir(key)
        if self.stage in ("all", "train"):
            self.sh("-m", "d10.train", "--config", f"configs/{key}.yaml", "--out", run, *self.train_args())
        if self.stage in ("all", "eval"):
            self.infer(key, run)
            self.score(key, key.upper(), run)

    def e4(self):
        base = self.best_base()
        small = f"e4_data_size_{E4_FRACTION}"
        run = self.run_dir(small)
        # the ONLY change against the base setup is --train-fraction
        self.sh("-m", "d10.train", "--config", f"configs/{base}.yaml", "--out", run,
                "--train-fraction", E4_FRACTION, "--name", small, *self.train_args())
        self.infer(small, run)
        common = ["variable=data_size", f"base={base.upper()}"]
        self.score(small, "E4_data_size_0.1", run, extra=[f"train_fraction={E4_FRACTION}", *common])
        # the other value (100%) is the base run itself: same predictions, scored again
        self.score("e4_data_size_1.0", "E4_data_size_1.0", self.run_dir(base), self.pred(base),
                   extra=["train_fraction=1.0", *common])

    def e5(self):
        base = self.best_base()
        run = self.run_dir("e5")
        self.sh("-m", "d10.grpo", "--init", self.run_dir(base), "--out", run,
                *(["--max-steps", SMOKE_STEPS] if self.smoke else []))
        self.infer("e5", run)
        self.score("e5", "E5", run, extra=[f"init={base.upper()}"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("exps", nargs="+", choices=["e0", "e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base", choices=["e1", "e2", "e3"], help="base setup for e4/e5")
    ap.add_argument("--stage", choices=["all", "train", "eval"], default="all",
                    help="e1-e3 only: train, or infer + score, or both")
    ap.add_argument("--batch", nargs=2, type=int, metavar=("BATCH_SIZE", "GRAD_ACCUM"),
                    help="override batch size and accumulation for SFT, e.g. --batch 1 8 if out of memory")
    a = ap.parse_args(argv)
    p = Pipeline(a.smoke, a.base, a.stage, a.batch)
    for exp in a.exps:
        print(f"\n===== {exp.upper()}{' (smoke)' if a.smoke else ''} =====", flush=True)
        getattr(p, exp)() if exp in ("e0", "e4", "e5") else p.sft(exp)


if __name__ == "__main__":
    main()
