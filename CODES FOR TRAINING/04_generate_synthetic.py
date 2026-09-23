#!/usr/bin/env python3
"""
Standalone replacement for notebook 04. Built for an unattended overnight run
on an 8 GB card.

Two phases, deliberately separated:

  COMPOSE   builds rebalanced masks and writes them + their YOLO labels to disk.
            Fast, CPU-only, no GPU needed.
  GENERATE  paints an image for every composed mask that does not have one yet.

Because compose finishes before generate starts, and generate skips masks whose
image already exists, a crash at 3 a.m. costs you the current batch and nothing
else. Rerun the same command and it picks up where it stopped.

Usage
-----
    # 1. See how fast your card actually is (60 seconds)
    python generate_synthetic.py --benchmark

    # 2. Compose masks only, inspect them, then generate
    python generate_synthetic.py --compose-only
    python generate_synthetic.py --target 1500 --batch 4

    # 3. Resume after any interruption -- identical command
    python generate_synthetic.py --target 1500 --batch 4
"""

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# expandable_segments is a no-op on Windows (PyTorch logs a warning and ignores
# it). Harmless to set -- it helps if this ever runs on Linux.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Paths -- edit these two if your layout differs
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Study\CDC Project 1\Project")
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

CONTROLNET_ROOT = PROJECT_ROOT / "data" / "controlnet"
CONTROLNET_DIR = PROJECT_ROOT / "controlnet_output"
SYN_ROOT = PROJECT_ROOT / "data" / "synthetic"

SYN_IMG_DIR = SYN_ROOT / "images"
SYN_COND_DIR = SYN_ROOT / "conditioning_images"
SYN_DET_DIR = SYN_ROOT / "labels"
SYN_SEG_DIR = SYN_ROOT / "labels_seg"
PLAN_PATH = SYN_ROOT / "plan.jsonl"

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted anatomy, cartoon, " "unrealistic, text, watermark"
)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(SYN_ROOT / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Mask utilities
# --------------------------------------------------------------------------
def rgb_to_label(rgb, palette):
    label = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for cid, color in palette.items():
        if cid == 0:
            continue
        label[np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)] = cid
    return label


def label_to_rgb(label, palette):
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, color in palette.items():
        rgb[label == cid] = color
    return rgb


def instances_of(label, cid, min_area=64):
    binary = (label == cid).astype(np.uint8)
    if binary.sum() == 0:
        return []
    n, cc, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append(
            {
                "cid": cid,
                "mask": (cc[y : y + h, x : x + w] == i).astype(np.uint8),
                "area": area,
                "cx": float(cents[i][0]),
                "cy": float(cents[i][1]),
            }
        )
    return out


def paste_instance(label, inst, cx, cy, scale, rng, min_visible=0.55):
    H, W = label.shape
    m = inst["mask"]
    h, w = m.shape
    nw, nh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
    if nw >= W or nh >= H:
        return None
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), rng.uniform(-12, 12), 1.0)
    m = cv2.warpAffine(m, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    if m.sum() == 0:
        return None
    x1, y1 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    if x1 < 0 or y1 < 0 or x1 + nw > W or y1 + nh > H:
        return None

    before = {int(c): int((label == c).sum()) for c in np.unique(label) if c != 0}
    out = label.copy()
    region = out[y1 : y1 + nh, x1 : x1 + nw]
    region[m == 1] = inst["cid"]
    out[y1 : y1 + nh, x1 : x1 + nw] = region
    for c, n0 in before.items():
        if n0 and int((out == c).sum()) / n0 < min_visible:
            return None
    return out


def label_to_yolo(label, cid_to_yolo, min_area=64, seg=False):
    H, W = label.shape
    rows = []
    for cid, idx in cid_to_yolo.items():
        binary = (label == cid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            if seg:
                ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(
                    -1, 2
                )
                if len(ap) < 3:
                    continue
                ap = np.clip(ap, [0, 0], [W - 1, H - 1])
                rows.append(
                    f"{idx} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in ap)
                )
            else:
                x, y, w, h = cv2.boundingRect(c)
                rows.append(
                    f"{idx} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                    f"{w / W:.6f} {h / H:.6f}"
                )
    return rows


# --------------------------------------------------------------------------
# Phase A -- compose
# --------------------------------------------------------------------------
def compose(args, meta):
    palette, names = meta["palette"], meta["names"]
    cid_to_yolo, SIZE = meta["cid_to_yolo"], meta["size"]
    train_records = meta["train_records"]
    rng = random.Random(args.seed)

    def load_label(rec):
        bgr = cv2.imread(str(CONTROLNET_ROOT / rec["conditioning_image"]))
        return rgb_to_label(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), palette)

    counts = collections.Counter(c for r in train_records for c in r.get("classes", []))
    name_to_cid = {v: k for k, v in names.items()}

    if args.rare_classes:
        rare = [int(c) for c in args.rare_classes]
    else:
        ranked = [
            (name_to_cid[n], n, c) for n, c in counts.most_common() if n in name_to_cid
        ]
        top = ranked[0][2]
        # ASCENDING: rarest first. Slicing most_common() directly would keep the
        # LARGEST of the rare classes and silently drop the ones that need help.
        rare = [
            cid
            for _c, cid in sorted((c, cid) for cid, _n, c in ranked if c < 0.5 * top)
        ][: args.max_rare]
    log(
        "class counts (train): "
        + ", ".join(f"{n}={c}" for n, c in counts.most_common())
    )
    log("boosting: " + ", ".join(names[c] for c in rare))

    donors = [
        r for r in train_records if any(names[c] in r.get("classes", []) for c in rare)
    ]
    log(f"{len(donors)} donor frames -- banking instances")

    bank = {c: [] for c in rare}
    stats = {c: {"cx": [], "cy": []} for c in rare}
    for i, r in enumerate(donors):
        lab = load_label(r)
        for cid in rare:
            for inst in instances_of(lab, cid):
                bank[cid].append(inst)
                stats[cid]["cx"].append(inst["cx"] / SIZE)
                stats[cid]["cy"].append(inst["cy"] / SIZE)
        if (i + 1) % 500 == 0:
            log(f"  banked {i + 1}/{len(donors)}")

    need = {}
    for c in rare:
        if not bank[c]:
            log(f"  {names[c]}: nothing banked -- cannot boost")
            continue
        cap = args.max_reuse * len(bank[c])
        need[c] = min(args.target, cap)
        flag = "  <- CAPPED by diversity" if need[c] < args.target else ""
        log(f"  {names[c]:<24} bank={len(bank[c]):<5} target={need[c]}{flag}")
    if not need:
        log("nothing to boost -- aborting")
        return []

    recipients = rng.sample(train_records, min(args.bank_sample, len(train_records)))

    def sample_xy(cid, jitter=0.06):
        s = stats[cid]
        if not s["cx"]:
            return None
        i = rng.randrange(len(s["cx"]))
        return (
            float(
                np.clip((s["cx"][i] + rng.uniform(-jitter, jitter)) * SIZE, 0, SIZE - 1)
            ),
            float(
                np.clip((s["cy"][i] + rng.uniform(-jitter, jitter)) * SIZE, 0, SIZE - 1)
            ),
        )

    made = {c: 0 for c in need}
    plan, attempts = [], 0
    max_attempts = 60 * sum(need.values())
    t0 = time.time()

    while any(made[c] < need[c] for c in need) and attempts < max_attempts:
        attempts += 1
        lab = load_label(rng.choice(recipients))
        added = 0
        for _ in range(rng.randint(1, args.max_pastes)):
            hungry = [c for c in need if made[c] < need[c]]
            if not hungry:
                break
            cid = rng.choice(hungry)
            inst = rng.choice(bank[cid])
            for _try in range(12):
                xy = sample_xy(cid)
                if xy is None:
                    break
                out = paste_instance(
                    lab, inst, xy[0], xy[1], rng.uniform(0.85, 1.20), rng
                )
                if out is not None:
                    lab, added = out, added + 1
                    made[cid] += 1
                    break
        if added == 0:
            continue

        nm = f"syn_{len(plan):06d}"
        present = sorted({names[int(c)] for c in np.unique(lab) if int(c) in names})
        caption = "Laparoscopic surgery showing " + ", ".join(present)

        cv2.imwrite(
            str(SYN_COND_DIR / f"{nm}.png"),
            cv2.cvtColor(label_to_rgb(lab, palette), cv2.COLOR_RGB2BGR),
        )
        det = label_to_yolo(lab, cid_to_yolo, seg=False)
        seg = label_to_yolo(lab, cid_to_yolo, seg=True)
        if not det:
            (SYN_COND_DIR / f"{nm}.png").unlink(missing_ok=True)
            continue
        (SYN_DET_DIR / f"{nm}.txt").write_text("\n".join(det) + "\n")
        (SYN_SEG_DIR / f"{nm}.txt").write_text("\n".join(seg) + "\n")
        plan.append({"name": nm, "caption": caption})

        if len(plan) % 250 == 0:
            log(f"  composed {len(plan)} ({time.time() - t0:.0f}s)")

    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        for row in plan:
            f.write(json.dumps(row) + "\n")

    log(
        f"composed {len(plan)} masks from {attempts} attempts in {time.time() - t0:.0f}s"
    )
    log("instances added: " + ", ".join(f"{names[c]}={n}" for c, n in made.items()))
    if attempts >= max_attempts:
        log(
            "hit attempt cap -- placement rejected often; lower --target or relax min_visible"
        )
    return plan


# --------------------------------------------------------------------------
# Phase B -- generate
# --------------------------------------------------------------------------
def _load(cls, path, dtype, **kw):
    """diffusers renamed torch_dtype -> dtype. Support both."""
    try:
        return cls.from_pretrained(path, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=dtype, **kw)


def _try(label, fn):
    """Best-effort optional optimisation -- never kill a 4-hour run over one."""
    try:
        fn()
        log(f"  {label}: on")
        return True
    except Exception as e:
        log(f"  {label}: unavailable ({type(e).__name__}) -- continuing")
        return False


def build_pipe(args):
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    controlnet = _load(ControlNetModel, str(CONTROLNET_DIR), torch.float16)
    pipe = _load(
        StableDiffusionControlNetPipeline,
        BASE_MODEL,
        torch.float16,
        controlnet=controlnet,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # channels_last suits the conv-heavy UNet/ControlNet on modern kernels.
    _try(
        "channels_last",
        lambda: (
            pipe.unet.to(memory_format=torch.channels_last),
            pipe.controlnet.to(memory_format=torch.channels_last),
        ),
    )
    pipe.set_progress_bar_config(disable=True)

    # enable_vae_slicing() was removed from the pipeline in newer diffusers and
    # lives on the VAE now. At 512 with a small batch it barely matters, so try
    # both spellings and shrug if neither exists.
    if not _try("vae slicing", lambda: pipe.vae.enable_slicing()):
        _try("vae slicing (legacy)", lambda: pipe.enable_vae_slicing())

    # from_pretrained leaves the pre-cast fp32 copy in the allocator. Without
    # this you sit at ~5.8 GB instead of ~2.9 GB -- exactly the headroom you
    # want for a bigger batch.
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dtypes = {p.dtype for p in pipe.unet.parameters()}
    log(f"unet dtype={dtypes} | VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    if torch.float32 in dtypes:
        log("WARNING: unet holds fp32 params -- expect half the speed.")
    return pipe


def run_batch(pipe, batch, args, palette, size):
    import torch
    from PIL import Image

    conds, prompts, gens = [], [], []
    for i, row in enumerate(batch):
        conds.append(Image.open(SYN_COND_DIR / f"{row['name']}.png").convert("RGB"))
        prompts.append(row["caption"])
        gens.append(torch.Generator(device="cuda").manual_seed(args.seed + row["idx"]))
    kw = dict(
        prompt=prompts,
        image=conds,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.cond_scale,
        generator=gens,
    )
    if args.guidance > 1.0:
        kw["negative_prompt"] = [NEGATIVE_PROMPT] * len(batch)
    imgs = pipe(**kw).images
    for row, img in zip(batch, imgs):
        img.save(SYN_IMG_DIR / f"{row['name']}.jpg", quality=95)
    return len(batch)


def generate(args, meta, plan, pipe=None, batch=None):
    import torch

    if pipe is None:
        pipe = build_pipe(args)
    bs = batch or args.batch

    todo = [
        dict(r, idx=i)
        for i, r in enumerate(plan)
        if not (SYN_IMG_DIR / f"{r['name']}.jpg").exists()
    ]
    done_already = len(plan) - len(todo)
    log(f"{len(plan)} masks | {done_already} already generated | {len(todo)} to do")
    if not todo:
        return True
    if args.limit:
        todo = todo[: args.limit]

    i, done, failed = 0, 0, []
    t0 = time.time()
    while i < len(todo):
        batch = todo[i : i + bs]
        try:
            done += run_batch(pipe, batch, args, meta["palette"], meta["size"])
            i += len(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs = max(1, bs // 2)
                log(f"OOM -> batch size {bs}")
                continue
            failed.append(batch[0]["name"])
            i += 1
        except Exception as e:
            failed.append((batch[0]["name"], repr(e)[:120]))
            i += len(batch)
        if done and done % 100 < bs:
            el = time.time() - t0
            rate = el / max(done, 1)
            log(
                f"  {done}/{len(todo)}  {rate:.2f}s/img  "
                f"ETA {(len(todo) - done) * rate / 3600:.1f}h  "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
            )

    log(f"generated {done}/{len(todo)} in {(time.time() - t0) / 3600:.2f}h")
    if failed:
        log(f"{len(failed)} failures, first: {failed[:5]}")
    # A run where almost nothing succeeded must NOT report success, or an
    # unattended retry loop will treat the night as finished.
    if done == 0 or len(failed) > 0.5 * len(todo):
        log("FAILED: majority of generations errored -- see messages above")
        return False
    return True


def benchmark(args, meta, plan, pipe=None):
    """Time batch sizes 1,2,4,6 so you pick the fastest that fits.
    Returns the best batch size."""
    import torch

    if pipe is None:
        pipe = build_pipe(args)
    sample = [dict(r, idx=i) for i, r in enumerate(plan[:24])]
    if len(sample) < 12:
        log("fewer than 12 composed masks -- skipping benchmark")
        return args.batch

    log(f"benchmark: {args.steps} steps, guidance {args.guidance}")
    best = (None, 1e9)
    for bs in (1, 2, 4, 6):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            run_batch(pipe, sample[:bs], args, meta["palette"], meta["size"])  # warmup
            torch.cuda.synchronize()
            t0 = time.time()
            n = 0
            for k in range(0, 12, bs):
                n += run_batch(
                    pipe, sample[k : k + bs], args, meta["palette"], meta["size"]
                )
            torch.cuda.synchronize()
            per = (time.time() - t0) / n
            peak = torch.cuda.max_memory_allocated() / 1e9
            log(f"  batch {bs}: {per:.2f} s/img   peak {peak:.2f} GB")
            if per < best[1]:
                best = (bs, per)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  batch {bs}: OOM")
            break
        except Exception as e:
            torch.cuda.empty_cache()
            log(f"  batch {bs}: failed ({type(e).__name__})")
            break

    # The warmup wrote real images for the first few masks; that is fine --
    # generate() skips anything already on disk.
    if best[0]:
        log(f"BEST: batch {best[0]} at {best[1]:.2f} s/img")
        log(f"  {len(plan)} images -> {best[1] * len(plan) / 3600:.1f}h")
        return best[0]
    log("benchmark inconclusive -- falling back to --batch")
    return args.batch


# --------------------------------------------------------------------------
def load_meta():
    pal = json.loads((CONTROLNET_ROOT / "palette.json").read_text())
    records = [
        json.loads(l)
        for l in open(CONTROLNET_ROOT / "metadata.jsonl", encoding="utf-8")
    ]
    return {
        "size": pal["size"],
        "palette": {int(k): tuple(v) for k, v in pal["palette"].items()},
        "names": {int(k): v for k, v in pal["names"].items()},
        "cid_to_yolo": {int(k): int(v) for k, v in pal["cid_to_yolo"].items()},
        "yolo_names": pal["yolo_names"],
        "train_records": [r for r in records if r.get("split", "train") == "train"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target", type=int, default=1500, help="synthetic instances per rare class"
    )
    ap.add_argument(
        "--max-reuse", type=int, default=12, help="max repeats of one banked instance"
    )
    ap.add_argument("--max-rare", type=int, default=6)
    ap.add_argument("--rare-classes", type=int, nargs="+", default=None)
    ap.add_argument("--max-pastes", type=int, default=3)
    ap.add_argument("--bank-sample", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--cond-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--compose-only", action="store_true")
    ap.add_argument(
        "--recompose", action="store_true", help="rebuild the plan from scratch"
    )
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument(
        "--auto",
        action="store_true",
        help="compose if needed, auto-pick the fastest batch size, "
        "then generate everything. This is the overnight mode.",
    )
    args = ap.parse_args()

    for d in (SYN_IMG_DIR, SYN_COND_DIR, SYN_DET_DIR, SYN_SEG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    log(f"size={meta['size']} train_rows={len(meta['train_records'])}")

    if PLAN_PATH.exists() and not args.recompose:
        plan = [json.loads(l) for l in open(PLAN_PATH, encoding="utf-8")]
        log(f"reusing existing plan: {len(plan)} masks (--recompose to rebuild)")
    else:
        plan = compose(args, meta)

    if not plan:
        sys.exit("no plan -- nothing to do")

    (SYN_ROOT / "dataset_names.yaml").write_text(
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(meta["yolo_names"]))
    )

    if args.compose_only:
        log(
            "compose-only: masks and labels written. Inspect them, then rerun to generate."
        )
        return
    if args.auto:
        pipe = build_pipe(args)  # load the model ONCE
        best = benchmark(args, meta, plan, pipe=pipe)
        ok = generate(args, meta, plan, pipe=pipe, batch=best)
        log("DONE" if ok else "INCOMPLETE")
        sys.exit(0 if ok else 1)
    if args.benchmark:
        benchmark(args, meta, plan)
        log("\nNow run without --benchmark, adding --batch <best>.")
        log("Also worth trying --guidance 1.0: it skips classifier-free guidance")
        log("and is ~2x faster. Compare 20 images before committing to it.")
        return
    ok = generate(args, meta, plan)
    log("DONE" if ok else "INCOMPLETE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
