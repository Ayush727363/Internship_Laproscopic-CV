#!/usr/bin/env python3
r"""
live_infer5.py -- fixed fill bug, hardcoded colors, clean text, vertical list.

    python live_infer5.py --source test2.avi

FIXES vs live_infer4.py
--------------------------
1. HOLLOW-MASK BUG: cv2.fillPoly(layer, [poly1, poly2, ...], color) fills using
   even-odd style logic across ALL polygons passed together -- if
   findContours returns more than one contour for a blob (a real outer
   boundary plus a tiny noise contour, or the shrink-during-fade transform
   nudging point order), the fill can cancel out the middle of a shape,
   producing the hollow/border-only look you saw. Fixed by filling each
   contour with its own cv2.fillPoly call instead of batching them.

2. Text has NO background/shadow stroke now -- just the colored text, single
   draw call, no black outline underneath.

3. Text is visible ONLY during the hold phase (first --hold-sec seconds
   after last detection). The instant fade starts, the label disappears
   entirely -- only the shrinking/fading mask remains. Previously the label
   faded together with the mask; now it's binary: shown during hold, gone
   during fade.

4. Hardcoded, logical per-class colors (not random, not auto-generated):
     external iliac artery -> red        (artery)
     external iliac vein    -> blue        (vein)
     uterine artery         -> orange-red  (artery family, distinct from #1)
     ovary                  -> cream/white
     uterus                 -> magenta/pink
     ureter                 -> yellow      (classic "find the yellow ureter")
     obturator nerve        -> bright green (nerve)
     instruments             -> cyan        (metallic, clearly non-anatomical)

5. The "currently visible" list (below the FPS line) is now a VERTICAL
   stacked list, one class per line, instead of a comma-separated row.
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

# ---------------------------------------------------------------------------
# Hardcoded logical colors, BGR order (OpenCV). Keyed by class NAME so the
# mapping survives even if class index order ever changes.
# ---------------------------------------------------------------------------
CLASS_COLOR_BGR = {
    "external iliac artery": (60, 60, 220),    # red
    "external iliac vein":   (220, 100, 40),    # blue
    "uterine artery":        (0, 100, 255),     # orange-red (artery family)
    "ovary":                 (235, 245, 250),   # cream / off-white
    "uterus":                (200, 60, 210),    # magenta / pink
    "ureter":                (0, 230, 255),     # yellow
    "obturator nerve":       (60, 220, 60),     # bright green
    "instruments":           (255, 220, 60),    # cyan
}
FALLBACK_COLORS = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255)]


def pad_square_black(img):
    h, w = img.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    out = cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, (t, l, s)


def mask_to_display_polys(mask, meta, disp_scale, min_area=40):
    """One clean outer contour per separate blob. RETR_EXTERNAL already
    excludes inner/hole contours, so each returned poly is a simple solid
    shape -- safe to fill individually."""
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
        poly = p.astype(np.int32)
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def shrink_polys(polys, factor):
    """Scale each polygon toward its OWN centroid. factor=1 -> unchanged."""
    if factor >= 0.999:
        return polys
    out = []
    for p in polys:
        c = p.mean(axis=0)
        shrunk = ((p.astype(np.float32) - c) * factor + c).astype(np.int32)
        if len(shrunk) >= 3:
            out.append(shrunk)
    return out


def fill_polys_solid(layer, polys, color):
    """Fill each polygon SEPARATELY -- fixes the hollow/border-only bug that
    batched cv2.fillPoly(layer, polys, color) can produce."""
    for p in polys:
        cv2.fillPoly(layer, [p], color)


class ClassState:
    __slots__ = ("polys", "color", "conf", "frames_since_seen")

    def __init__(self):
        self.polys = None
        self.color = None
        self.conf = 0.0
        self.frames_since_seen = 10 ** 9

    def replace(self, polys, color, conf):
        self.polys = polys
        self.color = color
        self.conf = conf
        self.frames_since_seen = 0

    def age(self):
        self.frames_since_seen += 1

    def visible(self, hold_frames, fade_frames):
        return self.polys is not None and self.frames_since_seen < hold_frames + fade_frames

    def in_hold_phase(self, hold_frames):
        return self.frames_since_seen <= hold_frames

    def alpha_and_scale(self, hold_frames, fade_frames):
        k = self.frames_since_seen
        if k <= hold_frames:
            return 1.0, 1.0
        t = (k - hold_frames) / max(1, fade_frames)
        t = min(1.0, max(0.0, t))
        return (1.0 - t), (1.0 - 0.35 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--save", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--conf-scale", type=float, default=1.0)
    ap.add_argument("--hold-sec", type=float, default=1.5)
    ap.add_argument("--fade-sec", type=float, default=1.5)
    ap.add_argument("--no-scores", action="store_true")
    ap.add_argument("--no-threshold-print", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--half", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = {}
    if CFG_PATH.exists():
        cfg = json.loads(CFG_PATH.read_text())
    weights = a.weights or cfg.get("weights") or str(DEFAULT_WEIGHTS)
    model = YOLO(weights)
    NAMES = [model.names[i] for i in range(len(model.names))]
    CONF = {n: float(cfg.get("conf", {}).get(n, 0.25)) * a.conf_scale for n in NAMES}

    fallback_i = 0
    COLORS = {}
    for i, n in enumerate(NAMES):
        if n in CLASS_COLOR_BGR:
            COLORS[i] = CLASS_COLOR_BGR[n]
        else:
            COLORS[i] = FALLBACK_COLORS[fallback_i % len(FALLBACK_COLORS)]
            fallback_i += 1

    if not a.no_threshold_print:
        print("thresholds: " + ", ".join(f"{k}={v:.2f}" for k, v in CONF.items()))
    print("colors: " + ", ".join(f"{n}={COLORS[i]}" for i, n in enumerate(NAMES)))

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

        for c in seen_this_frame:
            states[c].replace(by_class_polys[c], COLORS[c], by_class_conf[c])
        for c, st in states.items():
            if c not in seen_this_frame:
                st.age()

        layer = np.zeros_like(disp)
        label_lines = []   # (text, color) -- vertical list, hold-phase only
        for c, st in states.items():
            if not st.visible(hold_frames, fade_frames):
                continue
            op, shrink = st.alpha_and_scale(hold_frames, fade_frames)
            if op <= 0.02:
                continue
            polys = shrink_polys(st.polys, shrink) if shrink < 0.999 else st.polys
            col = tuple(int(v * op) for v in st.color)
            fill_polys_solid(layer, polys, col)

            if not a.no_labels and st.in_hold_phase(hold_frames):
                txt = NAMES[c] if a.no_scores else f"{NAMES[c]} {st.conf:.2f}"
                label_lines.append((txt, st.color))

        cv2.addWeighted(layer, a.alpha, disp, 1.0, 0, dst=disp)

        # per-instance on-mask labels -- ONLY during hold phase, no shadow
        for c, st in states.items():
            if a.no_labels or not st.in_hold_phase(hold_frames) or not st.visible(hold_frames, fade_frames):
                continue
            polys = st.polys
            biggest = max(polys, key=cv2.contourArea)
            txt = NAMES[c] if a.no_scores else f"{NAMES[c]} {st.conf:.2f}"
            pt = (max(2, int(biggest[:, 0].min())), max(18, int(biggest[:, 1].min()) - 6))
            cv2.putText(disp, txt, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        st.color, 2, cv2.LINE_AA)

        times.append(time.time() - t0)
        fps = 1 / (sum(times) / len(times)) if times else 0
        cv2.putText(disp, f"{fps:.1f} FPS  [lb {a.imgsz}]", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # vertical stacked list of currently-visible (hold-phase) classes
        y = 56
        for text, col in label_lines:
            name_only = text.rsplit(" ", 1)[0] if not a.no_scores else text
            cv2.putText(disp, name_only, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            y += 22

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
