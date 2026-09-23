#!/usr/bin/env python3
r"""
live_infer2.py -- demo-ready overlay. Reads tuned_config.json if present.

    python live_infer2.py --source test2.avi --mode crop --imgsz 768
    python live_infer2.py --source test2.avi --mode lb   --imgsz 512
    python live_infer2.py --source test2.avi --mode crop --imgsz 768 --save demo.mp4

--mode  lb    letterbox the whole 16:9 frame to a square (matches training if
              your Merged_Dataset frames were 16:9)
        crop  crop to the circular scope view first, so the anatomy fills the
              model input instead of sitting in a 512x288 band. Use whichever
              won the PART 1 table in tune_and_probe.py.

Padding is BLACK, matching notebook 02's pad_to_square(src_img, (0,0,0)).
Do not change it to gray.

Masks are drawn from contours transformed into display space, not by upscaling
each mask to max(H,W)^2 -- that was costing you ~15 FPS on 1080p.
Temporal smoothing actually smooths the MASKS now, not just a text label.
"""

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\Study\CDC Project 1\Project")
CFG_PATH = PROJECT_ROOT / "tuned_config.json"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "segment" / "surgical" / "weights" / "best.pt"


def pad_square_black(img):
    h, w = img.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    out = cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, (t, l, s)


def scope_bbox(img, thresh=15):
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


def masks_to_display_polys(mask, mode, meta, disp_scale, min_area=40):
    """Small model-res mask -> polygons in DISPLAY coordinates."""
    mh, mw = mask.shape[:2]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        p = c.reshape(-1, 2).astype(np.float32)
        if mode == "lb":
            top, left, s = meta
            p[:, 0] = (p[:, 0] * (s / mw) - left) * disp_scale
            p[:, 1] = (p[:, 1] * (s / mh) - top) * disp_scale
        else:  # crop
            x0, y0, s = meta
            p[:, 0] = (p[:, 0] * (s / mw) + x0) * disp_scale
            p[:, 1] = (p[:, 1] * (s / mh) + y0) * disp_scale
        polys.append(p.astype(np.int32))
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--mode", choices=["lb", "crop"], default="lb")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--save", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--smooth", type=int, default=5,
                    help="frames of temporal mask smoothing; 0 disables")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--conf-scale", type=float, default=1.0,
                    help="multiply all tuned thresholds (lower = more detections)")
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = {}
    if CFG_PATH.exists():
        cfg = json.loads(CFG_PATH.read_text())
        print(f"loaded {CFG_PATH.name}")
    weights = a.weights or cfg.get("weights") or str(DEFAULT_WEIGHTS)
    model = YOLO(weights)
    NAMES = [model.names[i] for i in range(len(model.names))]
    CONF = {n: float(cfg.get("conf", {}).get(n, 0.25)) * a.conf_scale for n in NAMES}
    print(f"classes: {NAMES}")
    print("thresholds: " + ", ".join(f"{k}={v:.2f}" for k, v in CONF.items()))

    src = int(a.source) if str(a.source).isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")
    W = int(cap.get(3)) or 1280
    H = int(cap.get(4)) or 720
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"source {W}x{H} @ {fps_src:.1f} | mode={a.mode} imgsz={a.imgsz}")

    disp_scale = 1.0
    if a.display_width and W > a.display_width:
        disp_scale = a.display_width / W
    dw, dh = int(round(W * disp_scale)), int(round(H * disp_scale))

    writer = None
    if a.save:
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_src, (dw, dh))

    rng = np.random.default_rng(0)
    colors = {i: tuple(int(v) for v in rng.integers(70, 255, 3))
              for i in range(len(NAMES))}
    times = deque(maxlen=30)
    hist = deque(maxlen=max(1, a.smooth))

    win = "surgical segmentation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    min_conf = max(0.01, min(CONF.values()))

    # crop box is computed once -- the scope does not move within a clip
    fixed_bb = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        if a.mode == "crop":
            if fixed_bb is None:
                fixed_bb = scope_bbox(frame)
            if fixed_bb is None:
                inp, meta, mode = (cv2.resize(*pad_square_black(frame)[:1],
                                              (a.imgsz, a.imgsz)),
                                   pad_square_black(frame)[1], "lb")
            else:
                x0, y0, s = fixed_bb
                inp = cv2.resize(frame[y0:y0 + s, x0:x0 + s], (a.imgsz, a.imgsz),
                                 interpolation=cv2.INTER_AREA)
                meta, mode = fixed_bb, "crop"
        else:
            sq, m = pad_square_black(frame)
            inp = cv2.resize(sq, (a.imgsz, a.imgsz), interpolation=cv2.INTER_AREA)
            meta, mode = m, "lb"

        res = model.predict(inp, imgsz=a.imgsz, conf=min_conf, verbose=False,
                            augment=a.tta, max_det=a.max_det)[0]

        disp = (cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
                if disp_scale != 1.0 else frame.copy())
        layer = np.zeros_like(disp)
        labels = []
        present = Counter()

        if res.masks is not None and len(res.masks.data):
            data = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cnf = res.boxes.conf.cpu().numpy()
            for m_, c, s in zip(data, cls, cnf):
                if c >= len(NAMES) or s < CONF.get(NAMES[c], 0.25):
                    continue
                polys = masks_to_display_polys(m_ > 0.5, mode, meta, disp_scale)
                if not polys:
                    continue
                present[NAMES[c]] += 1
                cv2.fillPoly(layer, polys, colors[c])
                p = max(polys, key=cv2.contourArea)
                labels.append((f"{NAMES[c]} {s:.2f}",
                               int(p[:, 0].min()), int(p[:, 1].min()), colors[c]))

        # real temporal smoothing: average the colour layer over N frames
        if a.smooth:
            hist.append(layer.astype(np.float32))
            layer = (np.mean(hist, axis=0)).astype(np.uint8)

        cv2.addWeighted(layer, a.alpha, disp, 1.0, 0, dst=disp)
        for text, x, y, col in labels:
            pt = (max(2, x), max(18, y - 6))
            cv2.putText(disp, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        col, 1, cv2.LINE_AA)

        times.append(time.time() - t0)
        fps = 1 / (sum(times) / len(times)) if times else 0
        cv2.putText(disp, f"{fps:.1f} FPS  [{a.mode} {a.imgsz}]", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, ", ".join(f"{k}" for k, _ in present.most_common(5)),
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        if writer:
            writer.write(disp)
        cv2.imshow(win, disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
