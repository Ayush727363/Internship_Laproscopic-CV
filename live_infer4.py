#!/usr/bin/env python3
r"""
live_infer4.py -- detect, replace, hold, fade. No tracker library, fast.

    python live_infer4.py --source test2.avi --mode lb --imgsz 512

BEHAVIOUR (exactly what was asked for, nothing extra)
-------------------------------------------------------
- New detection for a class this frame -> its mask is FULLY REPLACED by the
  new one immediately. No blending old+new, no carrying old shape forward.
- No detection for a class this frame -> keep showing its LAST mask:
    phase A (--hold-sec, default 1.5s): fully opaque, unchanged, un-shrunk.
    phase B (--fade-sec, default 1.5s): opacity ramps 1.0 -> 0.0 AND the
        mask shrinks toward its own center (a cheap affine scale -- not a
        tracker, not motion-following, just a shrink-toward-center so it
        visually "closes in" rather than snapping off).
    After hold+fade elapses with nothing new -> gone.
- ONE label per class, placed once at that class's largest blob, even if a
  class currently has multiple separate blobs on screen. No stacked/
  duplicate text.

WHY THIS IS FAST
------------------
The previous version's slowdown came from creating a CSRT tracker object
PER detected instance, every time a track needed to keep going -- CSRT is a
heavy correlation-filter tracker, not built for many-instances-per-frame use.
This version tracks nothing. Between-detection frames do a cheap numpy
alpha-multiply and an affine warp on an already-small mask buffer -- both
are trivial compared to a model forward pass, so cost is dominated by
inference alone, same as your original 25-30 FPS script.

USAGE
-----
    python live_infer4.py --source test2.avi
    python live_infer4.py --source test2.avi --hold-sec 1.0 --fade-sec 2.0
    python live_infer4.py --source 0                          # live capture card
    python live_infer4.py --source test2.avi --no-scores --no-threshold-print
    python live_infer4.py --source test2.avi --save demo.mp4
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\Study\CDC Project 1\Project")
CFG_PATH = PROJECT_ROOT / "tuned_config.json"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "segment" / "surgical" / "weights" / "best.pt"

# High-contrast palette against red/pink tissue. BGR order.
PALETTE_BGR = [
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (60, 255, 60),    # bright green
    (255, 0, 255),    # magenta
    (0, 165, 255),    # orange
    (255, 200, 0),    # sky blue
    (0, 255, 140),    # spring green
    (255, 0, 140),    # violet-pink
]


def pad_square_black(img):
    h, w = img.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    out = cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, (t, l, s)


def mask_to_display_polys(mask, meta, disp_scale, min_area=40):
    mh, mw = mask.shape[:2]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    top, left, s = meta
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        p = c.reshape(-1, 2).astype(np.float32)
        p[:, 0] = (p[:, 0] * (s / mw) - left) * disp_scale
        p[:, 1] = (p[:, 1] * (s / mh) - top) * disp_scale
        polys.append(p.astype(np.int32))
    return polys


def shrink_polys(polys, factor):
    """Scale each polygon toward ITS OWN centroid by `factor` (1.0=no shrink,
    0.0=a point). Pure numpy, no tracker, negligible cost."""
    if factor >= 0.999:
        return polys
    out = []
    for p in polys:
        c = p.mean(axis=0)
        out.append(((p.astype(np.float32) - c) * factor + c).astype(np.int32))
    return out


class ClassState:
    """One slot per CLASS (not per blob) -- holds whatever is currently
    being displayed for that class: its polygons, color, best confidence
    among its blobs, and how long ago it was last actually detected."""

    __slots__ = ("polys", "color", "conf", "frames_since_seen")

    def __init__(self):
        self.polys = None
        self.color = None
        self.conf = 0.0
        self.frames_since_seen = 10 ** 9  # effectively "never seen"

    def replace(self, polys, color, conf):
        self.polys = polys
        self.color = color
        self.conf = conf
        self.frames_since_seen = 0

    def age(self):
        self.frames_since_seen += 1

    def visible(self, hold_frames, fade_frames):
        return self.polys is not None and self.frames_since_seen < hold_frames + fade_frames

    def alpha_and_scale(self, hold_frames, fade_frames):
        """(opacity_multiplier, shrink_factor) for the current age."""
        k = self.frames_since_seen
        if k <= hold_frames:
            return 1.0, 1.0
        t = (k - hold_frames) / max(1, fade_frames)   # 0 -> 1 across fade phase
        t = min(1.0, max(0.0, t))
        return (1.0 - t), (1.0 - 0.35 * t)  # fades to 0 opacity, shrinks ~35% at most


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--save", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="base mask opacity when fully held (before any fade)")
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--conf-scale", type=float, default=1.0)
    ap.add_argument("--hold-sec", type=float, default=1.5)
    ap.add_argument("--fade-sec", type=float, default=1.5)
    ap.add_argument("--no-scores", action="store_true")
    ap.add_argument("--no-threshold-print", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--half", action="store_true",
                    help="FP16 inference -- try this if FPS is still low, "
                         "usually a free ~1.3-1.5x on a laptop GPU")
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = {}
    if CFG_PATH.exists():
        cfg = json.loads(CFG_PATH.read_text())
    weights = a.weights or cfg.get("weights") or str(DEFAULT_WEIGHTS)
    model = YOLO(weights)
    NAMES = [model.names[i] for i in range(len(model.names))]
    CONF = {n: float(cfg.get("conf", {}).get(n, 0.25)) * a.conf_scale for n in NAMES}
    COLORS = {i: PALETTE_BGR[i % len(PALETTE_BGR)] for i in range(len(NAMES))}

    if not a.no_threshold_print:
        print("thresholds: " + ", ".join(f"{k}={v:.2f}" for k, v in CONF.items()))

    src = int(a.source) if str(a.source).isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")
    W = int(cap.get(3)) or 1280
    H = int(cap.get(4)) or 720
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    hold_frames = max(1, int(round(a.hold_sec * fps_src)))
    fade_frames = max(1, int(round(a.fade_sec * fps_src)))

    disp_scale = 1.0
    if a.display_width and W > a.display_width:
        disp_scale = a.display_width / W
    dw, dh = int(round(W * disp_scale)), int(round(H * disp_scale))

    writer = None
    if a.save:
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_src, (dw, dh))

    win = "surgical segmentation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    times = deque(maxlen=30)
    min_conf = max(0.01, min(CONF.values()))

    states = {i: ClassState() for i in range(len(NAMES))}

    print(f"hold={a.hold_sec}s ({hold_frames}f)  fade={a.fade_sec}s ({fade_frames}f)  "
          f"source_fps={fps_src:.1f}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        sq, meta = pad_square_black(frame)
        inp = cv2.resize(sq, (a.imgsz, a.imgsz), interpolation=cv2.INTER_AREA)

        disp = (cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
                if disp_scale != 1.0 else frame.copy())

        res = model.predict(inp, imgsz=a.imgsz, conf=min_conf, verbose=False,
                            max_det=a.max_det, half=a.half)[0]

        # gather this frame's detections, best (largest-area) blob set per class
        seen_this_frame = set()
        by_class_polys = {}
        by_class_conf = {}
        if res.masks is not None and len(res.masks.data):
            data = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cnf = res.boxes.conf.cpu().numpy()
            for m_, c, s in zip(data, cls, cnf):
                if c >= len(NAMES) or s < CONF.get(NAMES[c], 0.25):
                    continue
                polys = mask_to_display_polys(m_ > 0.5, meta, disp_scale)
                if not polys:
                    continue
                by_class_polys.setdefault(c, []).extend(polys)
                by_class_conf[c] = max(by_class_conf.get(c, 0.0), float(s))
                seen_this_frame.add(c)

        # REPLACE: any class detected this frame fully overwrites its old mask
        for c in seen_this_frame:
            states[c].replace(by_class_polys[c], COLORS[c], by_class_conf[c])
        # everything else just ages (hold/fade countdown ticks forward)
        for c, st in states.items():
            if c not in seen_this_frame:
                st.age()

        layer = np.zeros_like(disp)
        labels = []
        for c, st in states.items():
            if not st.visible(hold_frames, fade_frames):
                continue
            op, shrink = st.alpha_and_scale(hold_frames, fade_frames)
            if op <= 0.02:
                continue
            polys = shrink_polys(st.polys, shrink) if shrink < 0.999 else st.polys
            col = tuple(int(v * op) for v in st.color)
            cv2.fillPoly(layer, polys, col)
            biggest = max(polys, key=cv2.contourArea)
            txt = NAMES[c] if a.no_scores else f"{NAMES[c]} {st.conf:.2f}"
            labels.append((txt, int(biggest[:, 0].min()), int(biggest[:, 1].min()),
                          st.color, op))

        cv2.addWeighted(layer, a.alpha, disp, 1.0, 0, dst=disp)
        if not a.no_labels:
            for text, x, y, col, op in labels:
                pt = (max(2, x), max(18, y - 6))
                cv2.putText(disp, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 3, cv2.LINE_AA)
                fcol = tuple(int(v * op) for v in col)
                cv2.putText(disp, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            fcol, 1, cv2.LINE_AA)

        times.append(time.time() - t0)
        fps = 1 / (sum(times) / len(times)) if times else 0
        cv2.putText(disp, f"{fps:.1f} FPS  [lb {a.imgsz}]", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, ", ".join(NAMES[c] for c in states
                                    if states[c].visible(hold_frames, fade_frames)),
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
