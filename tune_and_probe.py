#!/usr/bin/env python3
r"""
tune_and_probe.py -- get the most out of the weights you ALREADY have.

No retraining. No re-annotation. Two jobs:

  PART 1  Measure what shape your training frames actually were, and sweep
          every sensible way of feeding the live video to the model
          (letterbox vs scope-circle crop, 512/640/768/1024, TTA on/off).
          Ranks them by how much the model actually finds.

  PART 2  Tune a per-class confidence threshold on the val set by maximising
          F1, instead of the hand-guessed numbers currently in live_infer.py.
          These are defensible numbers you can show in a review.

    cd "D:\Study\CDC Project 1\Project"
    .\.venv\Scripts\Activate.ps1
    python tune_and_probe.py --videos test1.avi test2.avi

Writes tuned_config.json, which the new live_infer.py reads directly.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\Study\CDC Project 1\Project")
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_final"
MERGED_IMAGES = PROJECT_ROOT / "Merged_Dataset" / "images"
BEST = PROJECT_ROOT / "runs" / "segment" / "surgical" / "weights" / "best.pt"
OUT_CFG = PROJECT_ROOT / "tuned_config.json"


def log(m=""):
    print(m, flush=True)


# ===================================================================
# PART 0 -- what shape was the training data REALLY?
# ===================================================================
def probe_source_shapes(n=400):
    log("=" * 72)
    log("PART 0 -- original frame shapes in Merged_Dataset (decides everything)")
    log("=" * 72)
    if not MERGED_IMAGES.exists():
        log(f"  MISSING {MERGED_IMAGES}")
        return None
    files = sorted(MERGED_IMAGES.iterdir())
    files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not files:
        log("  no images found")
        return None
    step = max(1, len(files) // n)
    shapes = Counter()
    fills = []
    for f in files[::step][:n]:
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        shapes[(w, h)] += 1
        # how much of the frame is non-black (i.e. actual scope content)?
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fills.append(float((g > 15).mean()))

    log(f"  sampled {sum(shapes.values())} of {len(files)} images")
    log(f"  {'resolution':<16}{'count':>8}{'aspect':>9}")
    for (w, h), c in shapes.most_common(10):
        log(f"  {f'{w}x{h}':<16}{c:>8}{w/h:>9.2f}")
    if fills:
        log(f"  non-black content fraction: mean {np.mean(fills):.2f}, "
            f"min {np.min(fills):.2f}, max {np.max(fills):.2f}")

    top = shapes.most_common(1)[0][0]
    ar = top[0] / top[1]
    log("")
    if 0.9 <= ar <= 1.15:
        log("  >>> Training frames were ~SQUARE. Your 16:9 live video letterboxed")
        log("      to square arrives at roughly HALF the scale the model expects.")
        log("      The scope-circle crop below should help a lot.")
    elif ar > 1.6:
        log("  >>> Training frames were ~16:9, same as your live video.")
        log("      Letterbox is the matched option; crop is a scale change.")
    else:
        log(f"  >>> Training frames aspect {ar:.2f} -- in between; test both.")
    return top


# ===================================================================
# preprocessing variants
# ===================================================================
def pad_square_black(img):
    h, w = img.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    out = cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, (t, l, s)


def scope_bbox(img, thresh=15):
    """Bounding box of the non-black circular scope view, squared off."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (g > thresh).astype(np.uint8)
    if m.sum() < 1000:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    H, W = img.shape[:2]
    s = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, min(cx - s // 2, W - s))
    y1 = max(0, min(cy - s // 2, H - s))
    s = min(s, W - x1, H - y1)
    return x1, y1, s


def prep_letterbox(img, size):
    sq, meta = pad_square_black(img)
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA), ("lb", meta)


def prep_crop(img, size):
    bb = scope_bbox(img)
    if bb is None:
        return prep_letterbox(img, size)
    x, y, s = bb
    sub = img[y:y + s, x:x + s]
    return cv2.resize(sub, (size, size), interpolation=cv2.INTER_AREA), ("crop", (x, y, s))


VARIANTS = [
    ("letterbox 512", prep_letterbox, 512, False),
    ("letterbox 640", prep_letterbox, 640, False),
    ("letterbox 768", prep_letterbox, 768, False),
    ("letterbox 1024", prep_letterbox, 1024, False),
    ("crop      512", prep_crop, 512, False),
    ("crop      640", prep_crop, 640, False),
    ("crop      768", prep_crop, 768, False),
    ("crop     1024", prep_crop, 1024, False),
    ("letterbox 512 +TTA", prep_letterbox, 512, True),
    ("crop      768 +TTA", prep_crop, 768, True),
]


def probe_videos(model, names, videos, n_frames, conf):
    log("")
    log("=" * 72)
    log("PART 1 -- which preprocessing gets the most out of the live video")
    log("=" * 72)
    results = {}
    for v in videos:
        vp = Path(v)
        if not vp.exists():
            vp = PROJECT_ROOT / v
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            log(f"  cannot open {v}")
            continue
        W = int(cap.get(3)); H = int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 500
        log(f"\n  {vp.name}: {W}x{H}, {total} frames, aspect {W/H:.2f}")

        frames = []
        for i in range(n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n_frames))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        cap.release()
        if not frames:
            continue

        bb = scope_bbox(frames[0])
        if bb:
            log(f"  scope circle detected: {bb[2]}x{bb[2]} at ({bb[0]},{bb[1]}) "
                f"-> crop keeps {bb[2]*bb[2]/(W*H)*100:.0f}% of pixels, "
                f"but fills 100% of the model input instead of "
                f"{H/max(W,H)*100:.0f}%")

        log("")
        log(f"  {'variant':<22}{'dets':>6}{'classes':>9}{'meanconf':>10}  breakdown")
        log("  " + "-" * 86)
        for label, fn, size, tta in VARIANTS:
            per_cls = Counter()
            confs = []
            for fr in frames:
                inp, _ = fn(fr, size)
                r = model.predict(inp, imgsz=size, conf=conf, verbose=False,
                                  augment=tta, max_det=50)[0]
                if r.boxes is not None and len(r.boxes):
                    for ci, s in zip(r.boxes.cls.cpu().numpy().astype(int),
                                     r.boxes.conf.cpu().numpy()):
                        per_cls[names[ci]] += 1
                        confs.append(float(s))
            tot = sum(per_cls.values())
            mc = np.mean(confs) if confs else 0.0
            brk = ", ".join(f"{k.split()[-1]}={v}" for k, v in per_cls.most_common(5))
            log(f"  {label:<22}{tot:>6}{len(per_cls):>9}{mc:>10.3f}  {brk}")
            results[(vp.name, label)] = (tot, len(per_cls), mc)
    return results


# ===================================================================
# PART 2 -- per-class thresholds tuned on val
# ===================================================================
def load_gt(lbl_path):
    """class -> list of xyxy boxes (normalised), from YOLO-seg polygons."""
    out = defaultdict(list)
    if not lbl_path.exists():
        return out
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) < 7:
            continue
        c = int(p[0])
        xy = np.array(p[1:], dtype=np.float32).reshape(-1, 2)
        out[c].append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
    return out


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tune_thresholds(model, names, limit, iou_thr=0.5):
    log("")
    log("=" * 72)
    log("PART 2 -- per-class confidence tuned on val (maximise F1)")
    log("=" * 72)
    img_dir = YOLO_ROOT / "images" / "val"
    lbl_dir = YOLO_ROOT / "labels" / "val"
    files = sorted(img_dir.glob("*.jpg"))
    if limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    log(f"  scoring {len(files)} val images at conf=0.01 ...")

    # per class: list of (score, is_tp), and total GT count
    recs = defaultdict(list)
    n_gt = Counter()

    for i, f in enumerate(files):
        gt = load_gt(lbl_dir / f"{f.stem}.txt")
        for c, boxes in gt.items():
            n_gt[c] += len(boxes)
        img = cv2.imread(str(f))
        if img is None:
            continue
        r = model.predict(img, imgsz=512, conf=0.01, verbose=False, max_det=100)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        H, W = img.shape[:2]
        used = defaultdict(set)
        order = np.argsort(-r.boxes.conf.cpu().numpy())
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cnf = r.boxes.conf.cpu().numpy()
        for k in order:
            c = int(cls[k])
            b = xyxy[k] / np.array([W, H, W, H])
            best_j, best_i = -1, 0.0
            for j, g in enumerate(gt.get(c, [])):
                if j in used[c]:
                    continue
                v = iou(b, g)
                if v > best_i:
                    best_i, best_j = v, j
            tp = best_i >= iou_thr
            if tp:
                used[c].add(best_j)
            recs[c].append((float(cnf[k]), tp))
        if (i + 1) % 200 == 0:
            log(f"    {i+1}/{len(files)}")

    grid = np.arange(0.05, 0.91, 0.01)
    best_conf = {}
    log("")
    log(f"  {'class':<24}{'bestConf':>10}{'F1':>8}{'P':>8}{'R':>8}{'GT':>7}")
    log("  " + "-" * 65)
    for c, nm in enumerate(names):
        rs = recs.get(c, [])
        g = n_gt.get(c, 0)
        if not rs or g == 0:
            best_conf[nm] = 0.25
            log(f"  {nm:<24}{'0.25':>10}{'-':>8}{'-':>8}{'-':>8}{g:>7}  (no val data)")
            continue
        sc = np.array([x[0] for x in rs])
        tp = np.array([x[1] for x in rs], dtype=bool)
        bf, bt, bp, br = -1, 0.25, 0, 0
        for t in grid:
            m = sc >= t
            if not m.any():
                continue
            ntp = int(tp[m].sum())
            npred = int(m.sum())
            p = ntp / npred
            r_ = ntp / g
            f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0
            if f1 > bf:
                bf, bt, bp, br = f1, float(t), p, r_
        best_conf[nm] = round(bt, 2)
        log(f"  {nm:<24}{bt:>10.2f}{bf:>8.3f}{bp:>8.3f}{br:>8.3f}{g:>7}")
    return best_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=["test1.avi", "test2.avi"])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--probe-conf", type=float, default=0.15)
    ap.add_argument("--val-limit", type=int, default=400,
                    help="val images to use for threshold tuning (0 = all 867)")
    ap.add_argument("--skip-tune", action="store_true")
    a = ap.parse_args()

    if not BEST.exists():
        sys.exit(f"no weights at {BEST}")
    from ultralytics import YOLO

    model = YOLO(str(BEST))
    names = [model.names[i] for i in range(len(model.names))]
    log(f"classes: {names}")

    src_shape = probe_source_shapes()
    probe_videos(model, names, a.videos, a.frames, a.probe_conf)
    conf = {n: 0.25 for n in names}
    if not a.skip_tune:
        conf = tune_thresholds(model, names, a.val_limit)

    OUT_CFG.write_text(json.dumps({
        "weights": str(BEST),
        "names": names,
        "conf": conf,
        "source_shape": list(src_shape) if src_shape else None,
        "note": "conf tuned on val by F1; preprocessing chosen from PART 1 table",
    }, indent=2))
    log("")
    log(f"wrote {OUT_CFG}")
    log("")
    log("NEXT: read the PART 1 table, pick the top variant, and run")
    log("  python live_infer2.py --source test2.avi --mode <lb|crop> --imgsz <N>")


if __name__ == "__main__":
    main()
