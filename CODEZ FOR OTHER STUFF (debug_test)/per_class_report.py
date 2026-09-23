#!/usr/bin/env python3
r"""
per_class_report.py -- the missing number: per-class AP on val AND test,
for BOTH the epoch-100 checkpoint (best.pt/last.pt) and, if you have it,
an earlier checkpoint -- so you can see whether instruments/uterus carried
the aggregate mAP while rare classes sat near zero, and whether the last
~25 epochs of overfitting cost you anything class-specifically.

    cd "D:\Study\CDC Project 1\Project"
    .\.venv\Scripts\Activate.ps1
    python per_class_report.py > per_class_report.txt 2>&1

Paste the whole output back. This does not change your weights or data.
"""
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Study\CDC Project 1\Project")
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_final"
RUN_DIR = PROJECT_ROOT / "runs" / "segment" / "surgical"
YAML_P = YOLO_ROOT / "data.yaml"
IMGSZ = 512


def get_names():
    names = []
    for line in YAML_P.read_text().splitlines():
        m = re.match(r"\s+(\d+):\s*(.+)", line)
        if m:
            names.append(m.group(2).strip())
    return names


def table(metrics, names, title):
    print(f"\n--- {title} ---")
    for attr, kind in (("seg", "MASK"), ("box", "BOX")):
        m = getattr(metrics, attr, None)
        if m is None:
            continue
        try:
            idx = [int(c) for c in m.ap_class_index]
            maps = np.asarray(m.maps)
            ap50 = np.asarray(getattr(m, "ap50", []))
            p = np.asarray(getattr(m, "p", []))
            r = np.asarray(getattr(m, "r", []))
        except Exception as e:
            print(f"  [{kind}] could not extract ({type(e).__name__}: {e})")
            continue
        print(f"  [{kind}] {'class':<24}{'AP50':>8}{'AP50-95':>10}{'P':>8}{'R':>8}")
        print("  " + "-" * 58)

        def g(a, j):
            return f"{float(a[j]):.3f}" if j < len(a) else "   -  "

        for j, ci in enumerate(idx):
            nm = names[ci] if 0 <= ci < len(names) else str(ci)
            mv = f"{float(maps[ci]):.3f}" if ci < len(maps) else "   -  "
            print(f"       {nm:<24}{g(ap50,j):>8}{mv:>10}{g(p,j):>8}{g(r,j):>8}")
        missing = [names[i] for i in range(len(names)) if i not in idx]
        if missing:
            print(f"       (ABSENT -- no instances in this split: {missing})")


def eval_weights(tag, wpath, names):
    if not wpath.exists():
        print(f"\n[{tag}] MISSING {wpath} -- skipping")
        return
    from ultralytics import YOLO

    model = YOLO(str(wpath))
    ckpt_names = [model.names[i] for i in range(len(model.names))]
    if ckpt_names != names:
        print(f"\n[{tag}] WARNING: checkpoint class order != data.yaml order!")
        print(f"  checkpoint: {ckpt_names}")
        print(f"  data.yaml : {names}")

    splits = ["val", "test"] if (YOLO_ROOT / "images" / "test").exists() else ["val"]
    for split in splits:
        try:
            mt = model.val(
                data=str(YAML_P), split=split, imgsz=IMGSZ, batch=1,
                device=0, plots=False, verbose=False,
            )
            table(mt, names, f"{tag} -- {split.upper()}")
        except Exception as e:
            print(f"\n[{tag}] {split} eval failed: {type(e).__name__}: {e}")


def main():
    if not YAML_P.exists():
        sys.exit(f"missing {YAML_P}")
    names = get_names()
    print(f"classes: {names}")
    print(f"val images   : {len(list((YOLO_ROOT/'images'/'val').glob('*.jpg')))}")
    print(f"test images  : {len(list((YOLO_ROOT/'images'/'test').glob('*.jpg')))}")

    eval_weights("BEST (epoch ~selected by Ultralytics)", RUN_DIR / "weights" / "best.pt", names)
    eval_weights("LAST (epoch 100)", RUN_DIR / "weights" / "last.pt", names)

    print("\nDONE -- send the whole output back.")


if __name__ == "__main__":
    main()
