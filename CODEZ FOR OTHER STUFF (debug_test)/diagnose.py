#!/usr/bin/env python3
r"""
diagnose.py -- isolate WHERE the 0.85 mAP / bad-live-video gap comes from.

    cd "D:\Study\CDC Project 1\Project"
    .\.venv\Scripts\Activate.ps1
    python diagnose.py --video test2.avi

Runs seven independent checks. Each prints a verdict line starting with
[CHECK n]. Send me the whole stdout -- the combination of verdicts identifies
the cause; no single one does.

Nothing here modifies your dataset or weights.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\Study\CDC Project 1\Project")
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_final"
CONTROLNET_ROOT = PROJECT_ROOT / "data" / "controlnet"
RUN_DIR = PROJECT_ROOT / "runs" / "segment" / "surgical"
BEST = RUN_DIR / "weights" / "best.pt"
MERGED_ANN = PROJECT_ROOT / "Merged_Dataset" / "annotations"
IMGSZ = 512

_TS_RECORDING_RE = re.compile(r"^(\d{8})_(\d{6})_(\d+)_\d+_")
_SEQFRAME_RE = re.compile(r"^(\d+)frame[_\d]")
_VIDEO_N_RE = re.compile(r"^video(\d+)_")
_FRAMESEC_RE = re.compile(r"^frame_(\d+)_sec")


def hr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def pad_to_square_black(img):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(
        img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def preprocess_like_training(img, size=IMGSZ):
    """EXACTLY what notebook 02 did to build the training images."""
    return cv2.resize(pad_to_square_black(img), (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- CHECK 1
def check_classes(model, names_yaml):
    hr("CHECK 1 -- class order: checkpoint vs data.yaml")
    ckpt = [model.names[i] for i in range(len(model.names))]
    print(f"  checkpoint : {ckpt}")
    print(f"  data.yaml  : {names_yaml}")
    ok = ckpt == names_yaml
    print(f"[CHECK 1] class order match: {ok}")
    if not ok:
        print("  ^ EVERY label and per-class threshold is wrong. Stop here.")
    return ckpt


# ---------------------------------------------------------------- CHECK 2
def check_train_args():
    hr("CHECK 2 -- what the model was ACTUALLY trained with")
    p = RUN_DIR / "args.yaml"
    if not p.exists():
        print(f"[CHECK 2] MISSING {p}")
        return {}
    txt = p.read_text(encoding="utf-8", errors="replace")
    keep = (
        "model", "imgsz", "epochs", "batch", "mosaic", "close_mosaic", "scale",
        "copy_paste", "degrees", "rect", "overlap_mask", "mask_ratio", "single_cls",
        "task", "lr0", "optimizer", "patience",
    )
    got = {}
    for line in txt.splitlines():
        k = line.split(":")[0].strip()
        if k in keep:
            print("  " + line.strip())
            got[k] = line.split(":", 1)[1].strip()
    print(f"[CHECK 2] trained imgsz={got.get('imgsz')} model={got.get('model')} "
          f"mask_ratio={got.get('mask_ratio')}")
    return got


# ---------------------------------------------------------------- CHECK 3
def check_results_csv():
    hr("CHECK 3 -- did training converge, and what is the REAL headline number")
    p = RUN_DIR / "results.csv"
    if not p.exists():
        print(f"[CHECK 3] MISSING {p}")
        return
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    hdr = [h.strip() for h in lines[0].split(",")]
    rows = [r.split(",") for r in lines[1:]]
    print(f"  epochs logged: {len(rows)}")

    def col(name):
        for i, h in enumerate(hdr):
            if name in h:
                return i
        return None

    for want in ("metrics/mAP50(M)", "metrics/mAP50-95(M)",
                 "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        i = col(want)
        if i is None:
            continue
        vals = [float(r[i]) for r in rows if len(r) > i and r[i]]
        if vals:
            print(f"  {want:<24} final={vals[-1]:.4f}  best={max(vals):.4f} "
                  f"(epoch {vals.index(max(vals)) + 1})")
    i = col("val/seg_loss")
    if i is not None:
        vals = [float(r[i]) for r in rows if len(r) > i and r[i]]
        if len(vals) > 10:
            print(f"  val/seg_loss  first={vals[0]:.4f}  min={min(vals):.4f}  "
                  f"last={vals[-1]:.4f}")
            if vals[-1] > min(vals) * 1.15:
                print("  ^ val loss rose after its minimum -> overfitting")
    print("[CHECK 3] read the (M) = MASK numbers above; (B) = box and is not your task")


# ---------------------------------------------------------------- CHECK 4
def check_label_stats(names):
    hr("CHECK 4 -- what the TRAINING LABELS actually contain")
    lbl_dir = YOLO_ROOT / "labels" / "train"
    if not lbl_dir.exists():
        print(f"[CHECK 4] MISSING {lbl_dir}")
        return
    inst = Counter()
    tiny = Counter()
    imgs_with = Counter()
    real_inst, syn_inst = Counter(), Counter()
    n_real = n_syn = 0
    areas = defaultdict(list)

    for f in lbl_dir.glob("*.txt"):
        is_syn = f.stem.startswith("syn_")
        n_syn += is_syn
        n_real += not is_syn
        seen = set()
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 7:
                continue
            c = int(p[0])
            xy = np.array(p[1:], dtype=np.float32).reshape(-1, 2)
            a = cv2.contourArea((xy * IMGSZ).astype(np.float32))
            inst[c] += 1
            areas[c].append(a)
            (syn_inst if is_syn else real_inst)[c] += 1
            if a < 200:
                tiny[c] += 1
            seen.add(c)
        for c in seen:
            imgs_with[c] += 1

    print(f"  train label files: {n_real} real + {n_syn} synthetic "
          f"({n_syn / max(n_real + n_syn, 1):.0%} synthetic)")
    print()
    print(f"  {'class':<24}{'inst':>7}{'real':>8}{'syn':>7}{'imgs':>7}"
          f"{'medArea':>9}{'<200px':>8}")
    print("  " + "-" * 70)
    for c, nm in enumerate(names):
        med = np.median(areas[c]) if areas[c] else 0
        print(f"  {nm:<24}{inst[c]:>7}{real_inst[c]:>8}{syn_inst[c]:>7}"
              f"{imgs_with[c]:>7}{med:>9.0f}{tiny[c]:>8}")
    print("[CHECK 4] compare 'real' vs 'syn'. If a class is mostly synthetic, its")
    print("          val AP is measuring how well ControlNet imitates itself.")


# ---------------------------------------------------------------- CHECK 5
def video_key(name, batch):
    if not name:
        return None
    m = _TS_RECORDING_RE.match(name)
    if m:
        return f"cap_{m.group(1)}_{m.group(2)}_{m.group(3)}"
    m = _SEQFRAME_RE.match(name)
    if m:
        return f"b{batch:02d}_seq{int(m.group(1)):03d}"
    m = _VIDEO_N_RE.match(name)
    if m:
        return f"b{batch:02d}_vid{int(m.group(1)):02d}"
    if _FRAMESEC_RE.match(name):
        return f"b{batch:02d}_fsec"
    return None


def check_annotation_consistency(names):
    """THE instruments question: is a class annotated in some source videos and
    systematically absent in others? That teaches the model to suppress it."""
    hr("CHECK 5 -- per-source-video annotation coverage (the 'instruments' test)")
    cands = sorted(MERGED_ANN.glob("*.json"))
    if not cands:
        print(f"[CHECK 5] no COCO json in {MERGED_ANN}")
        return
    coco = json.loads(cands[0].read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    cat_sorted = sorted(cats)
    cid_to_idx = {cid: i for i, cid in enumerate(cat_sorted)}

    import datetime as dt
    dcs = sorted({im.get("date_captured") for im in coco["images"] if im.get("date_captured")})
    batch_of, idx, prev = {}, -1, None
    for d in dcs:
        t = dt.datetime.fromisoformat(d)
        if prev is None or (t - prev).total_seconds() > 5:
            idx += 1
        batch_of[d] = idx
        prev = t

    vid_of_img = {}
    for im in coco["images"]:
        extra = im.get("extra")
        nm = extra.get("name") if isinstance(extra, dict) else None
        vid_of_img[im["id"]] = video_key(nm, batch_of.get(im.get("date_captured"), -1)) or "UNKNOWN"

    anns_per_img = defaultdict(set)
    for a in coco["annotations"]:
        anns_per_img[a["image_id"]].add(a["category_id"])

    per_vid_imgs = Counter()
    per_vid_cls = defaultdict(Counter)
    for iid, v in vid_of_img.items():
        per_vid_imgs[v] += 1
        for cid in anns_per_img.get(iid, ()):
            per_vid_cls[v][cid] += 1

    print(f"  {len(per_vid_imgs)} source videos/cases\n")
    hdr = f"  {'video':<22}{'imgs':>6}" + "".join(f"{names[cid_to_idx[c]][:9]:>10}" for c in cat_sorted)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    zero_cov = defaultdict(list)
    for v, n in per_vid_imgs.most_common():
        row = f"  {v[:22]:<22}{n:>6}"
        for c in cat_sorted:
            k = per_vid_cls[v][c]
            row += f"{k / n * 100:>9.0f}%"
            if k == 0:
                zero_cov[c].append((v, n))
        print(row)

    print()
    for c in cat_sorted:
        z = zero_cov[c]
        if z:
            nz = sum(n for _, n in z)
            print(f"  {names[cid_to_idx[c]]:<24} ZERO annotations in {len(z)} videos "
                  f"= {nz} frames ({nz / sum(per_vid_imgs.values()) * 100:.0f}% of dataset)")
    print("[CHECK 5] any class with a large ZERO-annotation share is being taught")
    print("          to the model as background in those frames. That is the")
    print("          classic cause of 'trained on 16k instances, detects none'.")


# ---------------------------------------------------------------- CHECK 6
def run_pred(model, img, conf, imgsz, names, augment=False):
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False,
                      augment=augment, max_det=50)[0]
    c = Counter()
    if r.boxes is not None and len(r.boxes):
        for ci, s in zip(r.boxes.cls.cpu().numpy().astype(int),
                         r.boxes.conf.cpu().numpy()):
            c[names[ci]] += 1
    return c, r


def check_train_vs_video(model, names, video, n=12):
    hr("CHECK 6 -- same model on TRAIN frames vs VAL frames vs your live video")
    from ultralytics import YOLO  # noqa

    def sample(split, k):
        d = YOLO_ROOT / "images" / split
        if not d.exists():
            return []
        fs = [p for p in sorted(d.glob("*.jpg")) if not p.stem.startswith("syn_")]
        if not fs:
            return []
        step = max(1, len(fs) // k)
        return fs[::step][:k]

    for split in ("train", "val", "test"):
        fs = sample(split, n)
        if not fs:
            print(f"  {split:<6}: no real images")
            continue
        tot = Counter()
        empty = 0
        for f in fs:
            img = cv2.imread(str(f))
            c, _ = run_pred(model, img, 0.25, IMGSZ, names)
            if not c:
                empty += 1
            tot += c
        print(f"  {split:<6} ({len(fs)} imgs, conf .25): {empty} empty | "
              + ", ".join(f"{k}={v}" for k, v in tot.most_common()))

    if video:
        vp = Path(video)
        if not vp.exists():
            vp = PROJECT_ROOT / video
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"  cannot open video {vp}")
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
            w = int(cap.get(3)); h = int(cap.get(4))
            print(f"\n  video {vp.name}: {w}x{h}, {total} frames")
            for tag, sz, aug in (("as-trained 512", 512, False),
                                 ("upscaled  768", 768, False),
                                 ("upscaled 1024", 1024, False),
                                 ("512 + TTA   ", 512, True)):
                tot, empty = Counter(), 0
                for i in range(n):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
                    ok, fr = cap.read()
                    if not ok:
                        continue
                    sq = preprocess_like_training(fr, sz)
                    c, _ = run_pred(model, sq, 0.25, sz, names, augment=aug)
                    if not c:
                        empty += 1
                    tot += c
                print(f"  {tag} : {empty} empty | "
                      + ", ".join(f"{k}={v}" for k, v in tot.most_common()))
            cap.release()

    print("[CHECK 6] if TRAIN frames are also thin, the weights are the problem,")
    print("          not live_infer.py. If 768/1024 is much better, you are")
    print("          resolution-starved and a fine-tune at higher imgsz will pay.")


# ---------------------------------------------------------------- CHECK 7
def check_val_table(model, names):
    hr("CHECK 7 -- per-class val/test metrics (MASK)")
    yaml_p = YOLO_ROOT / "data.yaml"
    if not yaml_p.exists():
        print(f"[CHECK 7] MISSING {yaml_p}")
        return
    for split in ("val", "test"):
        try:
            m = model.val(data=str(yaml_p), split=split, imgsz=IMGSZ,
                          batch=1, device=0, plots=False, verbose=False)
        except Exception as e:
            print(f"  {split}: failed ({type(e).__name__}: {str(e)[:100]})")
            continue
        seg = getattr(m, "seg", None)
        if seg is None:
            print(f"  {split}: no seg metrics")
            continue
        idx = [int(c) for c in seg.ap_class_index]
        maps = np.asarray(seg.maps)
        ap50 = np.asarray(getattr(seg, "ap50", []))
        p = np.asarray(getattr(seg, "p", []))
        r = np.asarray(getattr(seg, "r", []))
        print(f"\n  --- {split.upper()} (mask) ---")
        print(f"  {'class':<24}{'AP50':>8}{'AP50-95':>10}{'P':>8}{'R':>8}")
        print("  " + "-" * 58)
        for j, ci in enumerate(idx):
            nm = names[ci] if ci < len(names) else str(ci)
            g = lambda a, k: f"{float(a[k]):.3f}" if k < len(a) else "   -  "
            mv = f"{float(maps[ci]):.3f}" if ci < len(maps) else "   -  "
            print(f"  {nm:<24}{g(ap50, j):>8}{mv:>10}{g(p, j):>8}{g(r, j):>8}")
        miss = [names[i] for i in range(len(names)) if i not in idx]
        if miss:
            print(f"  ABSENT from {split} (no AP computable): {miss}")
    print("\n[CHECK 7] a class absent from val contributed NOTHING to the 0.85.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="e.g. test2.avi")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--skip-val", action="store_true")
    a = ap.parse_args()

    if not BEST.exists():
        sys.exit(f"no weights at {BEST}")
    from ultralytics import YOLO

    model = YOLO(str(BEST))

    yaml_p = YOLO_ROOT / "data.yaml"
    names_yaml = []
    if yaml_p.exists():
        for line in yaml_p.read_text().splitlines():
            m = re.match(r"\s+(\d+):\s*(.+)", line)
            if m:
                names_yaml.append(m.group(2).strip())

    names = check_classes(model, names_yaml)
    check_train_args()
    check_results_csv()
    check_label_stats(names)
    try:
        check_annotation_consistency(names)
    except Exception as e:
        print(f"[CHECK 5] failed: {type(e).__name__}: {e}")
    check_train_vs_video(model, names, a.video, a.n)
    if not a.skip_val:
        check_val_table(model, names)

    hr("DONE -- send the whole output")


if __name__ == "__main__":
    main()
