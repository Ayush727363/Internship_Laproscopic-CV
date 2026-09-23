#!/usr/bin/env python3
r"""
train_yolo.py -- one-command, overnight-safe YOLO26-seg pipeline.

    cd "D:\Study\CDC Project 1\Project"
    .\.venv\Scripts\Activate.ps1
    python train_yolo.py

Runs unattended end to end:

  0  preflight        GPU, disk, packages, paths
  1  build dataset    video-level 85/10/5 split, real + synthetic, seg labels
  2  leakage audit    recovers synthetic provenance, drops anything tracing to
                      a val/test video, then HARD-ASSERTS the dataset is clean
  3  train            yolo26s-seg @ 512, resumes automatically if interrupted
  4  evaluate         per-class AP / precision / recall on val AND test
  5  export           TensorRT FP16 (falls back to ONNX, then to nothing)
  6  benchmark        measured FPS for each available format
  7  emit             live_infer.py with matching letterbox + temporal smoothing

Everything after step 3 is best-effort: a failure there prints a warning and the
run continues, because a missing FPS number must never cost you a trained model.

Safe to re-run. Steps 1-2 are skipped if the dataset marker exists (--rebuild to
force); step 3 resumes from last.pt.
"""

import argparse
import json
import os
import platform
import random
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# Must precede any torch import that might spawn workers.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import cv2
import numpy as np

# ===========================================================================
# CONFIG -- edit here, nothing else
# ===========================================================================
PROJECT_ROOT = Path(r"D:\Study\CDC Project 1\Project")

CONTROLNET_ROOT = PROJECT_ROOT / "data" / "controlnet"  # real, letterboxed 512
SYNTHETIC_ROOT = PROJECT_ROOT / "data" / "synthetic"  # generated
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_final"  # built by this script
RUNS_ROOT = PROJECT_ROOT / "runs"

# Your renumbered file_name (img000001.png) carries NO video info -- the COCO
# json keeps the original name in images[i]["extra"]["name"] (e.g.
# "000frame_0114_454s.jpg"). Video grouping is recovered from THAT field.
MERGED_ANNOTATIONS_DIR = PROJECT_ROOT / "Merged_Dataset" / "annotations"

# Empirically, extra.name across all 12987 COCO entries falls into exactly
# four naming families (verified by loading the raw json and categorizing
# every value, then confirming groupings against actual pixel content):
#
#  1. "<8-digit-date>_<6-digit-time>_<deviceid>_<seg>_..."
#     e.g. "20251209_154127_13579000_005_t05m28s.png"
#     One surgical RECORDING split into numbered chapters/segments (_000,
#     _001, ...). Segment number must be IGNORED when grouping -- visually
#     confirmed consecutive segments of the same recording are the same
#     surgery/tissue/scope (same case), so splitting by segment would leak
#     the same case across train/val. Group key = date+time+deviceid only.
#
#  2. "<3-digit-seq>frame_<n>_<n>s.jpg" / "<3-digit-seq>frame_<n>s.jpg"
#     e.g. "000frame_0114_454s.jpg"
#  3. "video<n>_<n>.jpg" / "video<n>_<n>_png_png.png"
#     e.g. "video3_00027.jpg"
#  4. "frame_<n>_sec.jpg"
#     e.g. "frame_1087_sec.jpg"
#
#     For families 2-4 the leading id (000, video3, ...) is a small per-export
#     counter that is REUSED ACROSS DIFFERENT SURGERIES. Confirmed by content:
#     the same id (e.g. "000frame_...") appearing in two different upload
#     batches (different images[i]["date_captured"], see below) shows
#     completely different anatomy/scope/lighting -- cosine similarity at the
#     random-pair baseline, not the near-1.0 similarity same-video frames
#     actually show. So the leading id ALONE is not a valid video key; it
#     must be combined with the upload batch to disambiguate.
#
# Upload batch is recovered from date_captured: timestamps within 5 seconds
# of each other belong to the same export/upload event (there are 13 such
# clusters across the 17 distinct date_captured strings; two clusters differ
# by only ~1 second, from a save-retry during export).
_TS_RECORDING_RE = re.compile(r"^(\d{8})_(\d{6})_(\d+)_\d+_")
_SEQFRAME_RE = re.compile(r"^(\d+)frame[_\d]")
_VIDEO_N_RE = re.compile(r"^video(\d+)_")
_FRAMESEC_RE = re.compile(r"^frame_(\d+)_sec")
BATCH_GAP_SECONDS = 5

COCO_JSON_FOR_VIDEO_ID = None  # auto-detected below if left as None

SPLIT_RATIOS = (0.85, 0.10, 0.05)  # train / val / test, BY VIDEO
SEED = 42

MODEL_CANDIDATES = ["yolo26s-seg.pt", "yolo11s-seg.pt", "yolov8s-seg.pt"]
IMGSZ = 512
EPOCHS = 100
PATIENCE = 30
BATCH = 16  # explicit -- Ultralytics AutoBatch (fraction mode) mis-profiled
# on this GPU and picked batch=8, using only ~2.4/8.5 GB VRAM.
# 24 (this run's own memory-per-sample extrapolation) left too
# little headroom for augmentation memory spikes on an 8.5 GB
# card; 16 is the middle ground -- confirmed-safe 4-workers/~12
# batch run vs. the 24/8 config. Revert to the 0.7/4 pairing above
# (git-free: see train_yolo_run.log 01:29:09) if 16/6 still OOMs.
WORKERS = 6  # i7-14700HX (28 threads) + 16 GB RAM can sustain this -- 6 is the
# middle ground between the confirmed-stable 4 and the untested 8
CACHE = "disk"  # 'disk' | False. RAM cache would need ~10 GB -- do not.

TIE_EPS = 1e-4  # Dice-score gap under which two candidates count as tied
MIN_POLY_AREA = 64

AUG = dict(
    copy_paste=0.3,
    mosaic=1.0,
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    degrees=10.0,
    translate=0.1,
    fliplr=0.0,
    flipud=0.0,
    mixup=0.0,
    shear=0.0,
    perspective=0.0,
)

SIG = 24


def signature(label, n_cls):
    """Continuous (not binarized) per-class occupancy at a finer grid than
    before. Binarizing at a coarse grid caused exact-score collisions between
    genuinely different frames in testing -- two unrelated frames produced an
    identical signature, which a top-2 tie-break silently missed. Continuous
    values collide only for near-pixel-identical masks, which is exactly the
    case we want to catch (and handle via the tie-set below), not one we want
    to manufacture through information loss."""
    out = np.zeros((n_cls, SIG, SIG), np.float32)
    for c in range(1, n_cls + 1):
        b = (label == c).astype(np.uint8)
        if b.any():
            out[c - 1] = cv2.resize(b, (SIG, SIG), interpolation=cv2.INTER_AREA)
    return out.ravel()  # signature grid for the provenance match


IMG_EXT = ".jpg"

LOG_PATH = PROJECT_ROOT / "train_yolo_run.log"


def log(msg=""):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def section(title):
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)


# ===========================================================================
# 0. PREFLIGHT
# ===========================================================================
def preflight():
    section("0. PREFLIGHT")
    log(f"python   : {sys.version.split()[0]}  ({platform.system()})")

    missing = []
    for mod, pipname in [
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("yaml", "pyyaml"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pipname)
    if missing:
        sys.exit(f"Missing packages: {missing}\n  pip install {' '.join(missing)}")

    import torch

    log(f"torch    : {torch.__version__}")
    if not torch.cuda.is_available():
        sys.exit(
            "CUDA unavailable. Reinstall torch:\n"
            "  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128"
        )
    log(f"device   : {torch.cuda.get_device_name(0)}")
    log(f"capability: {torch.cuda.get_device_capability(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"VRAM     : {vram:.1f} GB")
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        _ = (a @ a).float().sum().item()
        torch.cuda.synchronize()
        log("fp16 test: OK")
    except RuntimeError as e:
        sys.exit(f"fp16 matmul failed -- torch build lacks sm_120 kernels.\n{e}")

    import ultralytics

    log(f"ultralytics: {ultralytics.__version__}")

    for p in (
        CONTROLNET_ROOT / "metadata.jsonl",
        CONTROLNET_ROOT / "palette.json",
        CONTROLNET_ROOT / "images",
        CONTROLNET_ROOT / "conditioning_images",
    ):
        if not p.exists():
            sys.exit(f"Missing required input: {p}")
    if not (SYNTHETIC_ROOT / "images").exists():
        log("WARNING: no synthetic images found -- will train on real data only.")

    free = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    log(f"free disk: {free:.1f} GB")
    return free


# ===========================================================================
# Mask -> label helpers (identical treatment for real and synthetic)
# ===========================================================================
def rgb_to_label(rgb, palette):
    lab = np.zeros(rgb.shape[:2], np.uint8)
    for cid, color in palette.items():
        if cid == 0:
            continue
        lab[np.all(rgb == np.array(color, np.uint8), axis=-1)] = cid
    return lab


def label_to_yolo_seg(label, cid_to_yolo, min_area=MIN_POLY_AREA):
    H, W = label.shape
    rows = []
    for cid, idx in cid_to_yolo.items():
        binary = (label == cid).astype(np.uint8)
        if not binary.any():
            continue
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(
                -1, 2
            )
            if len(ap) < 3:
                continue
            ap = np.clip(ap, [0, 0], [W - 1, H - 1])
            rows.append(f"{idx} " + " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in ap))
    return rows


def _batch_index_map(date_captured_values, gap_seconds=BATCH_GAP_SECONDS):
    """date_captured string -> upload-batch index. Values within gap_seconds
    of each other (sorted) are treated as one upload/export event."""
    import datetime as _dt

    uniq = sorted(set(date_captured_values))
    parsed = [(d, _dt.datetime.fromisoformat(d)) for d in uniq]
    out, idx, prev = {}, -1, None
    for d, t in parsed:
        if prev is None or (t - prev).total_seconds() > gap_seconds:
            idx += 1
        out[d] = idx
        prev = t
    return out


def video_key_from_extra(name, batch):
    """extra.name + its upload-batch index -> video/case id, or None if the
    value matches none of the four empirically-observed naming families
    (see the comment block above COCO_JSON_FOR_VIDEO_ID)."""
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


def split_videos(videos, ratios, seed):
    """Guarantees a non-empty val (>=2 videos) and test (>=3). Without this a
    small video count silently yields an empty val, and Ultralytics then has
    nothing to select best.pt on -- you'd ship the most overfit checkpoint."""
    vids = sorted(videos)
    random.Random(seed).shuffle(vids)
    n = len(vids)
    if n == 1:
        return {vids[0]: "train"}
    if n == 2:
        return {vids[0]: "train", vids[1]: "val"}
    n_va = max(1, int(round(n * ratios[1])))
    n_te = max(1, int(round(n * ratios[2])))
    while n_va + n_te > n - 1:
        if n_te > 1:
            n_te -= 1
        elif n_va > 1:
            n_va -= 1
        else:
            break
    n_tr = n - n_va - n_te
    out = {}
    for v in vids[:n_tr]:
        out[v] = "train"
    for v in vids[n_tr : n_tr + n_va]:
        out[v] = "val"
    for v in vids[n_tr + n_va :]:
        out[v] = "test"
    return out


def load_video_map():
    """stem (e.g. 'img000008') -> video id, recovered from the ORIGINAL COCO
    json's images[i]['extra']['name'] field (+ its date_captured upload
    batch), since file_name was resequenced and no longer carries video info
    on its own. Returns {} if the COCO json is unavailable."""
    json_path = COCO_JSON_FOR_VIDEO_ID
    if json_path is None:
        candidates = sorted(
            (PROJECT_ROOT / "Merged_Dataset" / "annotations").glob("*.json")
        )
        json_path = candidates[0] if candidates else None
    if json_path is None or not Path(json_path).exists():
        log(
            f"WARNING: no COCO json found for video-id recovery "
            f"(looked in Merged_Dataset/annotations). A frame-level split "
            f"would make val worthless -- fix COCO_JSON_FOR_VIDEO_ID."
        )
        return {}

    import json as _json

    coco = _json.loads(Path(json_path).read_text())
    images = coco.get("images", [])
    batch_of = _batch_index_map(
        im.get("date_captured") for im in images if im.get("date_captured")
    )

    mapping = {}
    matched = 0
    unresolved_samples = []
    for im in images:
        stem = Path(im["file_name"]).stem
        extra = im.get("extra")
        name = extra.get("name") if isinstance(extra, dict) else None
        batch = batch_of.get(im.get("date_captured"), -1)
        v = video_key_from_extra(name, batch)
        if v is not None:
            mapping[stem] = v
            matched += 1
        else:
            if len(unresolved_samples) < 20:
                unresolved_samples.append(
                    {
                        "file_name": im.get("file_name"),
                        "extra": extra,
                        "date_captured": im.get("date_captured"),
                    }
                )
    log(
        f"video-id recovery: {matched}/{len(images)} images "
        f"matched via {json_path.name} -> extra.name"
    )
    if matched < len(images):
        n_bad = len(images) - matched
        log(
            f"WARNING: {n_bad}/{len(images)} COCO entries did not match any "
            f"known extra.name naming family. Raw samples that failed "
            f"(not stems -- the actual extra field, so this is debuggable "
            f"without re-opening the json):"
        )
        for s in unresolved_samples:
            log(f"    {s}")
    return mapping


def match_back(syn_sigs, real_sigs, chunk=128, tie_eps=1e-4):
    """Recover which real frame each synthetic mask was composed from.

    Uses the Dice coefficient: 2*|real & syn| / (|real| + |syn|).

    Containment (|real & syn| / |real|) was tried first and is WRONG here:
    generate_synthetic.py's paste deliberately overwrites underlying pixels
    (up to 45% of an existing class, per its min_visible=0.55 allowance), so
    the true source's containment can legitimately drop below 1.0. An unrelated
    frame with a small, sparse mask can then trivially score a higher
    containment purely from having a small denominator. Testing caught this
    concretely: the true source ranked below a wrong frame under containment,
    but ranked first under Dice for the same pair every time. Dice's symmetric
    denominator doesn't reward small unrelated frames the same way.

    Returns (best_idx, best_score, margin_to_2nd, tie_sets) where tie_sets[k]
    is every real-frame index within tie_eps of the top Dice score for
    synthetic row k -- not just the top 2. A top-2-only check can miss a third
    (or later) candidate genuinely tied with the winner; testing surfaced this
    exact failure. The caller must require ALL indices in tie_sets[k] to be in
    the training split before trusting the match."""
    real_sums = real_sigs.sum(1).astype(np.float32)
    n = len(syn_sigs)
    bi = np.zeros(n, np.int64)
    bs = np.zeros(n, np.float32)
    mg = np.zeros(n, np.float32)
    tie_sets = [None] * n
    for k in range(0, n, chunk):
        blk = syn_sigs[k : k + chunk]
        blk_sums = blk.sum(1).astype(np.float32)
        inter = blk @ real_sigs.T
        denom = blk_sums[:, None] + real_sums[None, :]
        denom[denom == 0] = 1.0
        dice = 2.0 * inter / denom
        order = np.argsort(-dice, axis=1)
        for r in range(len(blk)):
            top = order[r, 0]
            top_score = dice[r, top]
            second = dice[r, order[r, 1]] if dice.shape[1] > 1 else 0.0
            tied = order[r][dice[r, order[r]] >= top_score - tie_eps]
            bi[k + r] = top
            bs[k + r] = top_score
            mg[k + r] = top_score - second
            tie_sets[k + r] = tied.tolist()
    return bi, bs, mg, tie_sets


def link_or_copy(src, dst):
    """Hardlink when possible (same volume) -- saves ~10 GB over copying."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# ===========================================================================
# 1 + 2. BUILD DATASET  (with leakage audit)
# ===========================================================================
def build_dataset(rebuild=False):
    section("1. BUILD DATASET")
    marker = YOLO_ROOT / ".build_complete.json"
    if marker.exists() and not rebuild:
        info = json.loads(marker.read_text())
        log(
            f"dataset already built ({info['n_train']} train / {info['n_val']} val "
            f"/ {info['n_test']} test) -- skipping. Use --rebuild to force."
        )
        return YOLO_ROOT / "data.yaml", info

    if YOLO_ROOT.exists():
        log("clearing previous build ...")
        shutil.rmtree(YOLO_ROOT, ignore_errors=True)
    for s in ("train", "val", "test"):
        (YOLO_ROOT / "images" / s).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / s).mkdir(parents=True, exist_ok=True)

    pal = json.loads((CONTROLNET_ROOT / "palette.json").read_text())
    SIZE = pal["size"]
    palette = {int(k): tuple(v) for k, v in pal["palette"].items()}
    names = {int(k): v for k, v in pal["names"].items()}
    cid_to_yolo = {int(k): int(v) for k, v in pal["cid_to_yolo"].items()}
    yolo_names = pal["yolo_names"]
    n_cls_ids = max(palette) if palette else 0
    log(f"classes ({len(yolo_names)}): {yolo_names}")
    log(f"mask size: {SIZE}")

    records = [
        json.loads(l)
        for l in open(CONTROLNET_ROOT / "metadata.jsonl", encoding="utf-8")
    ]
    log(f"real records: {len(records)}")

    # ---- video-level split -------------------------------------------------
    video_map = load_video_map()  # stem -> video id, recovered from COCO json

    def video_id_of(st):
        return video_map.get(st)

    stems, unmatched, unmatched_stems = [], 0, []
    for r in records:
        st = Path(r["image"]).stem
        if video_id_of(st) is None:
            unmatched += 1
            if len(unmatched_stems) < 20:
                unmatched_stems.append(st)
        stems.append(st)
    if unmatched:
        log(
            f"WARNING: could not determine a video id for {unmatched}/{len(stems)} "
            f"images (stem not present in the COCO json's images[].file_name, "
            f"so extra.name could not even be looked up)."
        )
        log(f"    unresolved stems (raw, up to 20): {unmatched_stems}")
        if unmatched > 0.5 * len(stems):
            sys.exit(
                f"Video id unresolved for most images.\n"
                "See the raw unresolved stems and, above, any raw "
                "extra.name samples load_video_map() could not parse. "
                "Fix video_key_from_extra()/COCO_JSON_FOR_VIDEO_ID at the "
                "top of this file -- a frame-level split would make val "
                "worthless and select an overfit model."
            )

    videos = {video_id_of(s) or f"__solo_{s}" for s in stems}
    vsplit = split_videos(videos, SPLIT_RATIOS, SEED)
    log(f"{len(videos)} videos -> " + str(dict(Counter(vsplit.values()))))
    if Counter(vsplit.values()).get("test", 0) == 0:
        log("NOTE: too few videos for a test split; test metrics will be skipped.")

    def split_of_stem(st):
        return vsplit[video_id_of(st) or f"__solo_{st}"]

    # ---- real images -------------------------------------------------------
    log("\nwriting real images + labels ...")
    real_labels = {}  # stem -> label map (kept for the audit)
    per_split = Counter()
    cls_split = defaultdict(Counter)
    skipped_empty = 0

    for i, r in enumerate(records):
        st = Path(r["image"]).stem
        img_p = CONTROLNET_ROOT / r["image"]
        cond_p = CONTROLNET_ROOT / r["conditioning_image"]
        if not img_p.exists() or not cond_p.exists():
            continue
        bgr = cv2.imread(str(cond_p))
        if bgr is None:
            continue
        lab = rgb_to_label(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), palette)
        rows = label_to_yolo_seg(lab, cid_to_yolo)
        if not rows:
            skipped_empty += 1
            continue
        sp = split_of_stem(st)
        link_or_copy(img_p, YOLO_ROOT / "images" / sp / f"{st}{IMG_EXT}")
        (YOLO_ROOT / "labels" / sp / f"{st}.txt").write_text("\n".join(rows) + "\n")
        real_labels[st] = lab
        per_split[sp] += 1
        for row in rows:
            cls_split[sp][yolo_names[int(row.split()[0])]] += 1
        if (i + 1) % 2000 == 0:
            log(f"  {i+1}/{len(records)}")
    log(
        f"real written: {dict(per_split)}  (skipped {skipped_empty} with no usable polygons)"
    )

    # ---- synthetic + leakage audit ----------------------------------------
    section("2. LEAKAGE AUDIT (synthetic provenance)")
    syn_kept = syn_dropped = 0
    audit = {}
    plan_p = SYNTHETIC_ROOT / "plan.jsonl"
    syn_img_dir = SYNTHETIC_ROOT / "images"
    syn_cond_dir = SYNTHETIC_ROOT / "conditioning_images"

    if plan_p.exists() and syn_img_dir.exists():
        plan = [json.loads(l) for l in open(plan_p, encoding="utf-8")]
        plan = [p for p in plan if (syn_img_dir / f"{p['name']}.jpg").exists()]
        log(f"synthetic images on disk: {len(plan)}")
        log("Your synthetic set was composed from nb02's per-image split, so some")
        log("masks are copies of val/test-video layouts. Recovering provenance ...")

        real_stems = list(real_labels.keys())
        real_sigs = np.stack([signature(real_labels[s], n_cls_ids) for s in real_stems])
        log(f"real signature matrix: {real_sigs.shape} ({real_sigs.nbytes/1e6:.0f} MB)")

        syn_labels, syn_sigs, syn_names_ok = [], [], []
        for p in plan:
            bgr = cv2.imread(str(syn_cond_dir / f"{p['name']}.png"))
            if bgr is None:
                continue
            lab = rgb_to_label(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), palette)
            syn_labels.append(lab)
            syn_sigs.append(signature(lab, n_cls_ids))
            syn_names_ok.append(p["name"])
        if syn_sigs:
            syn_sigs = np.stack(syn_sigs)
            bi, bs, mg, tie_sets = match_back(syn_sigs, real_sigs)
            log(f"match score  : median {np.median(bs):.3f}  min {bs.min():.3f}")
            log(f"match margin : median {np.median(mg):.3f}")
            n_ties = sum(1 for t in tie_sets if len(t) > 1)
            if n_ties:
                log(
                    f"note: {n_ties} synthetic masks tied exactly with >1 real "
                    f"frame -- ALL tied candidates must be train to keep them."
                )
            if np.median(bs) < 0.6:
                log("WARNING: low match confidence. Being maximally conservative:")
                log("         every ambiguous synthetic image will be dropped.")

            for k, nm in enumerate(syn_names_ok):
                tied_splits = [split_of_stem(real_stems[i]) for i in tie_sets[k]]
                sp_best = tied_splits[0]
                # Every candidate tied with the winner must be train, or we
                # cannot rule out that the true source is a val/test video.
                # (A top-2-only check missed a real 3-way tie during testing.)
                ok = all(s == "train" for s in tied_splits)
                audit[nm] = {
                    "src": real_stems[bi[k]],
                    "split": sp_best,
                    "score": float(bs[k]),
                    "margin": float(mg[k]),
                    "n_tied": len(tie_sets[k]),
                    "kept": bool(ok),
                }
                if not ok:
                    syn_dropped += 1
                    continue
                link_or_copy(
                    syn_img_dir / f"{nm}.jpg",
                    YOLO_ROOT / "images" / "train" / f"{nm}{IMG_EXT}",
                )
                rows = label_to_yolo_seg(syn_labels[k], cid_to_yolo)
                if not rows:
                    (YOLO_ROOT / "images" / "train" / f"{nm}{IMG_EXT}").unlink(
                        missing_ok=True
                    )
                    syn_dropped += 1
                    audit[nm]["kept"] = False
                    continue
                (YOLO_ROOT / "labels" / "train" / f"{nm}.txt").write_text(
                    "\n".join(rows) + "\n"
                )
                syn_kept += 1
                per_split["train"] += 1
                for row in rows:
                    cls_split["train"][yolo_names[int(row.split()[0])]] += 1

            log(f"\nsynthetic kept    : {syn_kept}")
            log(
                f"synthetic dropped : {syn_dropped}  (traced to a val/test video, "
                f"or ambiguous)"
            )
            (YOLO_ROOT / "provenance_audit.json").write_text(
                json.dumps(audit, indent=1)
            )
    else:
        log("no synthetic data found -- real only.")

    # ---- hard assertions ---------------------------------------------------
    section("VERIFYING DATASET INTEGRITY")
    problems = []
    seen_video_split = {}
    for sp in ("train", "val", "test"):
        for ip in (YOLO_ROOT / "images" / sp).glob(f"*{IMG_EXT}"):
            st = ip.stem
            if st.startswith("syn_"):
                if sp != "train":
                    problems.append(f"SYNTHETIC IN {sp.upper()}: {st}")
                continue
            v = video_id_of(st) or f"__solo_{st}"
            if v in seen_video_split and seen_video_split[v] != sp:
                problems.append(f"VIDEO {v} SPANS {seen_video_split[v]} AND {sp}")
            seen_video_split[v] = sp
            if not (YOLO_ROOT / "labels" / sp / f"{st}.txt").exists():
                problems.append(f"missing label: {sp}/{st}")
    # every kept synthetic must trace to a train video
    for nm, a in audit.items():
        if a["kept"] and a["split"] != "train":
            problems.append(f"KEPT SYNTHETIC FROM {a['split'].upper()} VIDEO: {nm}")

    if problems:
        for p in problems[:20]:
            log("  " + p)
        sys.exit(
            f"\nDATASET INTEGRITY FAILED ({len(problems)} problems). "
            "Refusing to train on a leaking dataset."
        )
    log("no synthetic in val/test          OK")
    log("no video spans two splits         OK")
    log("every image has a label           OK")

    # ---- per-class table ---------------------------------------------------
    section("PER-CLASS INSTANCES")
    hdr = f"{'class':<26}{'train':>9}{'val':>8}{'test':>8}"
    log(hdr)
    log("-" * len(hdr))
    warn = []
    for nm in yolo_names:
        t, v, te = cls_split["train"][nm], cls_split["val"][nm], cls_split["test"][nm]
        log(f"{nm:<26}{t:>9}{v:>8}{te:>8}")
        if v == 0:
            warn.append(nm)
    log("-" * len(hdr))
    log(
        f"{'IMAGES':<26}{per_split['train']:>9}{per_split['val']:>8}{per_split['test']:>8}"
    )
    thin = [n for n in yolo_names if 0 < cls_split["val"][n] < 10]
    if warn or thin:
        log("")
        if warn:
            log(f"WARNING: ABSENT from val: {warn}")
            log("  Their val AP is undefined -- they cannot guide model selection")
            log("  or threshold tuning at all.")
        if thin:
            log(f"WARNING: fewer than 10 val instances: {thin}")
            log("  Their val AP will be extremely noisy; do not read small")
            log("  epoch-to-epoch changes on these as real.")
        log("  Try a different SEED to reshuffle which videos land in val, or")
        log("  accept it and judge these classes on the clinical review instead.")

    # ---- data.yaml ---------------------------------------------------------
    yaml_p = YOLO_ROOT / "data.yaml"
    lines = [f"path: {YOLO_ROOT.as_posix()}", "train: images/train", "val: images/val"]
    if per_split["test"]:
        lines.append("test: images/test")
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(yolo_names)]
    yaml_p.write_text("\n".join(lines) + "\n")
    log(f"\nwrote {yaml_p}")

    info = {
        "n_train": per_split["train"],
        "n_val": per_split["val"],
        "n_test": per_split["test"],
        "syn_kept": syn_kept,
        "syn_dropped": syn_dropped,
        "names": yolo_names,
        "has_test": bool(per_split["test"]),
        "train_counts": {n: cls_split["train"][n] for n in yolo_names},
        "val_counts": {n: cls_split["val"][n] for n in yolo_names},
    }
    (YOLO_ROOT / ".build_complete.json").write_text(json.dumps(info, indent=1))
    return yaml_p, info


# ===========================================================================
# 3. TRAIN
# ===========================================================================
def pick_model():
    from ultralytics import YOLO

    for cand in MODEL_CANDIDATES:
        try:
            m = YOLO(cand)
            log(f"model: {cand}")
            return m, cand
        except Exception as e:
            log(f"  {cand} unavailable ({type(e).__name__}) -- trying next")
    sys.exit("No usable model. Try: pip install -U ultralytics")


def train(yaml_p, free_gb, retrain=False):
    section("3. TRAIN")
    from ultralytics import YOLO

    run_dir = RUNS_ROOT / "segment" / "surgical"
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    done_marker = run_dir / ".train_complete"

    # Re-running the script after a finished run must NOT silently retrain from
    # scratch and overwrite good weights.
    if done_marker.exists() and not retrain:
        log(f"training already completed ({done_marker}).")
        log("Skipping to evaluation. Use --retrain to train again from scratch.")
        return run_dir

    cache = CACHE
    if cache == "disk" and free_gb < 20:
        log(f"only {free_gb:.0f} GB free -- disabling disk cache")
        cache = False

    kw = dict(
        data=str(yaml_p),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        cache=cache,
        device=0,
        amp=True,
        patience=PATIENCE,
        seed=SEED,
        deterministic=False,
        project=str(RUNS_ROOT / "segment"),
        name="surgical",
        exist_ok=True,
        plots=True,
        val=True,
        **AUG,
    )

    if last.exists() and not retrain:
        log(f"found {last} -- RESUMING")
        try:
            YOLO(str(last)).train(resume=True)
            done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            return run_dir
        except Exception as e:
            msg = str(e).lower()
            if "finished" in msg or "nothing to resume" in msg:
                log("checkpoint reports training already finished -- skipping to eval.")
                done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
                return run_dir
            log(f"resume failed ({type(e).__name__}: {str(e)[:150]})")
            if best.exists():
                log("best.pt exists -- refusing to overwrite it. Skipping to eval.")
                log("Use --retrain if you really want to start over.")
                return run_dir
            log("no best.pt -- starting fresh")

    model, tag = pick_model()
    log(f"imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH} workers={WORKERS} cache={cache}")
    log(f"aug: {AUG}")
    try:
        model.train(**kw)
    except Exception as e:
        if "out of memory" in str(e).lower():
            log("OOM -- retrying at batch=4, workers=2")
            import torch

            torch.cuda.empty_cache()
            kw.update(batch=4, workers=2)
            model, _ = pick_model()
            model.train(**kw)
        else:
            raise
    done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return run_dir


# ===========================================================================
# 4. EVALUATE
# ===========================================================================
def per_class_table(metrics, names, title):
    """Ultralytics' metrics object has shifted shape across versions, so try
    several access paths and degrade to the summary dict rather than crashing
    after a successful training run.

    Indexing gotcha: Metric.maps is indexed by CLASS ID (length nc), while
    ap50/p/r are indexed by POSITION within ap_class_index. Mixing them up
    silently prints the wrong number against each class name."""
    log(f"\n--- {title} ---")
    printed = False
    for attr in ("box", "seg"):
        try:
            m = getattr(metrics, attr)
            idx = [int(c) for c in getattr(m, "ap_class_index")]
            maps = np.asarray(getattr(m, "maps"))  # by class id
            ap50 = np.asarray(getattr(m, "ap50", []))  # by position
            p = np.asarray(getattr(m, "p", []))  # by position
            r = np.asarray(getattr(m, "r", []))  # by position
            kind = "mask" if attr == "seg" else "box"
            log(f"[{kind}] {'class':<26}{'AP50':>8}{'AP50-95':>10}{'P':>8}{'R':>8}")
            log("       " + "-" * 60)

            def pos(arr, j):
                return f"{float(arr[j]):.3f}" if j < len(arr) else "   -  "

            for j, ci in enumerate(idx):
                nm = names[ci] if 0 <= ci < len(names) else str(ci)
                mv = f"{float(maps[ci]):.3f}" if ci < len(maps) else "   -  "
                log(
                    f"       {nm:<26}{pos(ap50,j):>8}{mv:>10}"
                    f"{pos(p,j):>8}{pos(r,j):>8}"
                )
            missing = [names[i] for i in range(len(names)) if i not in idx]
            if missing:
                log(f"       (absent from this split, no AP computable: {missing})")
            printed = True
        except Exception:
            continue
    if not printed:
        try:
            log(
                "  "
                + json.dumps(
                    {k: float(v) for k, v in metrics.results_dict.items()}, indent=2
                )
            )
        except Exception:
            log("  (could not extract per-class metrics -- weights are still fine)")


def evaluate(run_dir, yaml_p, info):
    section("4. EVALUATE")
    from ultralytics import YOLO

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        log("no best.pt -- skipping evaluation")
        return None
    names = info["names"]
    model = YOLO(str(best))
    for split in (["val", "test"] if info.get("has_test") else ["val"]):
        try:
            mt = model.val(
                data=str(yaml_p),
                split=split,
                imgsz=IMGSZ,
                batch=1,
                device=0,
                plots=False,
                verbose=False,
            )
            per_class_table(mt, names, f"{split.upper()} (best.pt)")
        except Exception as e:
            log(f"{split} evaluation failed: {type(e).__name__}: {e}")
    return best


# ===========================================================================
# 5 + 6. EXPORT AND BENCHMARK
# ===========================================================================
def export_and_bench(best):
    section("5. EXPORT + 6. FPS BENCHMARK")
    if best is None:
        return
    from ultralytics import YOLO
    import torch

    formats = [
        ("engine", dict(half=True), "TensorRT FP16"),
        ("onnx", dict(half=False, simplify=True), "ONNX"),
    ]
    exported = []
    for fmt, extra, label in formats:
        try:
            log(f"exporting {label} ...")
            p = YOLO(str(best)).export(format=fmt, imgsz=IMGSZ, device=0, **extra)
            exported.append((label, Path(p)))
            log(f"  -> {p}")
        except Exception as e:
            log(f"  {label} export unavailable ({type(e).__name__}). Skipping.")
            log(f"     ({str(e)[:160]})")

    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    log("")
    for label, wpath in [("PyTorch", best)] + exported:
        try:
            m = YOLO(str(wpath))
            for _ in range(5):
                m.predict(dummy, imgsz=IMGSZ, device=0, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            N = 40
            for _ in range(N):
                m.predict(dummy, imgsz=IMGSZ, device=0, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = (time.time() - t0) / N
            log(f"  {label:<16} {dt*1000:6.1f} ms/frame   {1/dt:5.1f} FPS")
        except Exception as e:
            log(f"  {label:<16} benchmark failed ({type(e).__name__})")
    log("\nNote: this laptop is your review machine. The deployment GPU is faster,")
    log("so treat these numbers as a floor, not the shipping figure.")


# ===========================================================================
# 7. LIVE INFERENCE HELPER
# ===========================================================================
LIVE_TEMPLATE = '''#!/usr/bin/env python3
"""
live_infer.py -- real-time overlay for surgical video.

    python live_infer.py --source 0                  # webcam / capture card
    python live_infer.py --source clip.mp4
    python live_infer.py --source clip.mp4 --save out.mp4

Preprocessing here MUST match training: the model was trained on square
letterboxed {IMGSZ}px frames, so live frames are letterboxed the same way and the
masks are mapped back onto the original 16:9 for display. Feeding raw 16:9
frames instead silently degrades accuracy.

Detections are smoothed over a short window because frame-independent
segmentation flickers, and a structure blinking on and off reads as far less
trustworthy to a clinician than a slightly imperfect boundary.
"""
import argparse, time
from collections import deque, defaultdict
import cv2, numpy as np
from ultralytics import YOLO

WEIGHTS = r"{WEIGHTS}"
IMGSZ = {IMGSZ}
NAMES = {NAMES}
# Per-class confidence. Rare structures get a lower bar on purpose: a missed
# ureter matters more than an extra candidate the surgeon can dismiss. Tune
# these on your val split.
CONF = {CONF}
SMOOTH = 5


def letterbox_square(img, size):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA), (top, left, s)


def unletterbox_mask(mask, meta, out_hw):
    top, left, s = meta
    m = cv2.resize(mask.astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
    return m[top:top + out_hw[0], left:left + out_hw[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()

    model = YOLO(WEIGHTS)
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {{a.source}}")

    writer = None
    if a.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rng = np.random.default_rng(0)
    colors = {{i: tuple(int(v) for v in rng.integers(60, 255, 3)) for i in range(len(NAMES))}}
    history = defaultdict(lambda: deque(maxlen=SMOOTH))
    times = deque(maxlen=30)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        H, W = frame.shape[:2]
        sq, meta = letterbox_square(frame, IMGSZ)
        res = model.predict(sq, imgsz=IMGSZ, verbose=False,
                            conf=min(CONF.values()) if CONF else 0.25)[0]

        overlay = frame.copy()
        present = set()
        if res.masks is not None:
            data = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cnf = res.boxes.conf.cpu().numpy()
            for m, c, s in zip(data, cls, cnf):
                if s < CONF.get(NAMES[c], 0.25):
                    continue
                present.add(c)
                full = unletterbox_mask(m > 0.5, meta, (H, W))
                overlay[full.astype(bool)] = colors[c]
                ys, xs = np.where(full)
                if len(xs):
                    cv2.putText(frame, f"{{NAMES[c]}} {{s:.2f}}",
                                (int(xs.min()), max(18, int(ys.min()) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[c], 2)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        if not a.no_smooth:
            for i in range(len(NAMES)):
                history[i].append(i in present)
            stable = [NAMES[i] for i in range(len(NAMES))
                      if sum(history[i]) > len(history[i]) / 2]
        else:
            stable = [NAMES[i] for i in present]

        times.append(time.time() - t0)
        fps_now = 1 / (sum(times) / len(times)) if times else 0
        cv2.putText(frame, f"{{fps_now:.1f}} FPS", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, ", ".join(stable[:5]), (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if writer:
            writer.write(frame)
        cv2.imshow("surgical segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''


def write_live_helper(best, names, train_counts=None):
    section("7. LIVE INFERENCE HELPER")
    if best is None:
        return
    # Starting thresholds derived from the ACTUAL training instance counts, not
    # from keywords in the class name. A name-based rule would lower the bar for
    # "external iliac artery" (common) while leaving "ovary" (rarer) alone --
    # sorting by words instead of by data.
    conf = {}
    if train_counts:
        top = max(train_counts.values()) or 1
        for n in names:
            ratio = top / max(train_counts.get(n, 1), 1)
            if ratio >= 20:
                conf[n] = 0.10  # severely under-represented
            elif ratio >= 5:
                conf[n] = 0.15
            elif ratio >= 2:
                conf[n] = 0.20
            else:
                conf[n] = 0.25
    else:
        conf = {n: 0.25 for n in names}
    log("starting per-class thresholds (from train instance counts):")
    for n in names:
        c = train_counts.get(n, 0) if train_counts else 0
        log(f"  {n:<26} {c:>7} instances -> conf {conf[n]:.2f}")
    log("These are a starting point, NOT tuned. Sweep them on val before the review.")
    out = PROJECT_ROOT / "live_infer.py"
    out.write_text(
        LIVE_TEMPLATE.format(
            WEIGHTS=str(best), IMGSZ=IMGSZ, NAMES=repr(names), CONF=repr(conf)
        ),
        encoding="utf-8",
    )
    log(f"wrote {out}")
    log("  python live_infer.py --source clip.mp4 --save annotated.mp4")
    log("  Per-class thresholds are pre-filled with lower values for the rare")
    log("  structures; tune them on val before the clinical review.")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="force dataset rebuild")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument(
        "--retrain",
        action="store_true",
        help="ignore a completed run and train again from scratch",
    )
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    global EPOCHS
    if a.epochs:
        EPOCHS = a.epochs

    t0 = time.time()
    log("#" * 72)
    log(f"# train_yolo.py  started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("#" * 72)

    free = preflight()
    yaml_p, info = build_dataset(rebuild=a.rebuild)

    run_dir = RUNS_ROOT / "segment" / "surgical"
    if not a.skip_train:
        run_dir = train(yaml_p, free, retrain=a.retrain)

    best = evaluate(run_dir, yaml_p, info)
    try:
        export_and_bench(best)
    except Exception:
        log("export/benchmark stage failed -- weights are unaffected:")
        log(traceback.format_exc(limit=3))
    try:
        write_live_helper(best, info["names"], info.get("train_counts"))
    except Exception:
        log("live helper generation failed -- weights are unaffected.")

    section("DONE")
    log(f"total: {(time.time()-t0)/3600:.2f} h")
    log(f"weights : {run_dir / 'weights' / 'best.pt'}")
    log(f"results : {run_dir}")
    log(f"log     : {LOG_PATH}")


if __name__ == "__main__":
    # Required on Windows: Ultralytics uses spawn for dataloader workers, which
    # re-imports this file in each child. Without the guard you get infinite
    # process spawning instead of training.
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        log("\ninterrupted -- rerun the same command to resume from last.pt")
    except Exception:
        log("\nFATAL:")
        log(traceback.format_exc())
        sys.exit(1)
