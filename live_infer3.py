#!/usr/bin/env python3
r"""
live_infer3.py -- high-contrast colors, track-between-detections, clean labels.

    python live_infer3.py --source test2.avi --mode lb --imgsz 640
    python live_infer3.py --source test2.avi --mode lb --imgsz 640 --no-scores
    python live_infer3.py --source test2.avi --mode lb --imgsz 640 --detect-every 3
    python live_infer3.py --source test2.avi --mode lb --imgsz 640 --save demo.mp4

WHAT'S NEW vs live_infer2.py
-----------------------------
1. COLORS -- fixed, hand-picked high-contrast palette (cyan, yellow, bright
   green, magenta, orange...) instead of random pastel colors. Chosen to
   stand out against red/pink tissue specifically.

2. TRACKING BETWEEN DETECTIONS (--detect-every N, default 1 = off)
   Real object detection is the expensive part. This adds a light tracker
   (OpenCV CSRT) per detected instance: run full detection every Nth frame,
   and on the frames in between, let the tracker follow where that shape
   moved instead of re-running the model. If a tracked object isn't
   reconfirmed by a fresh detection within --forget-after seconds, its mask
   fades out smoothly instead of vanishing or flickering.
   This is standard "detect + track" -- it does not invent new shape
   information between detections, it carries the last known shape forward
   smoothly. Also usually raises your effective FPS since tracking is much
   cheaper than a full model pass.

3. --no-scores hides the "name 0.83" confidence number after each label,
   leaving just the class name. --no-threshold-print silences the startup
   terminal line listing every class's confidence cutoff.
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

# ---------------------------------------------------------------------------
# High-contrast palette, chosen against red/pink tissue. BGR order (OpenCV).
# Cycles if you have more classes than colors; your 8 classes fit exactly.
# ---------------------------------------------------------------------------
PALETTE_BGR = [
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (60, 255, 60),    # bright green
    (255, 0, 255),    # magenta
    (0, 165, 255),    # orange
    (255, 200, 0),    # sky blue
    (0, 255, 140),    # spring green
    (255, 0, 140),    # violet/pink-blue
    (0, 220, 255),    # gold
    (200, 100, 255),  # light purple
]


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


def mask_to_display_polys(mask, mode, meta, disp_scale, min_area=40):
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
        else:
            x0, y0, s = meta
            p[:, 0] = (p[:, 0] * (s / mw) + x0) * disp_scale
            p[:, 1] = (p[:, 1] * (s / mh) + y0) * disp_scale
        polys.append(p.astype(np.int32))
    return polys


def poly_bbox(poly):
    x, y, w, h = cv2.boundingRect(poly)
    return (x, y, w, h)


# ---------------------------------------------------------------------------
# One tracked instance: a detection followed across frames by a CSRT tracker
# until either a fresh detection reconfirms it, or it goes unconfirmed for
# too long and fades out.
# ---------------------------------------------------------------------------
class TrackedInstance:
    _next_id = 0

    def __init__(self, cls_idx, name, color, polys, frame, conf):
        self.id = TrackedInstance._next_id
        TrackedInstance._next_id += 1
        self.cls_idx = cls_idx
        self.name = name
        self.color = color
        self.polys = polys
        self.conf = conf
        self.age_since_seen = 0
        self.tracker = None
        self._init_tracker(frame)

    def _init_tracker(self, frame):
        try:
            biggest = max(self.polys, key=cv2.contourArea)
            bbox = poly_bbox(biggest)
            if bbox[2] < 4 or bbox[3] < 4:
                self.tracker = None
                return
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, bbox)
        except Exception:
            self.tracker = None  # opencv-contrib not installed -- degrade gracefully

    def update_from_detection(self, polys, conf, frame):
        self.polys = polys
        self.conf = conf
        self.age_since_seen = 0
        self._init_tracker(frame)  # re-anchor tracker to the fresh shape

    def advance_by_tracking(self, frame):
        """No fresh detection this frame -- shift the existing shape using
        the tracker's estimated motion, rather than freezing it in place."""
        self.age_since_seen += 1
        if self.tracker is None:
            return True  # keep showing the frozen last shape
        ok, bbox = self.tracker.update(frame)
        if not ok:
            return True
        old_bbox = poly_bbox(max(self.polys, key=cv2.contourArea))
        dx = (bbox[0] + bbox[2] / 2) - (old_bbox[0] + old_bbox[2] / 2)
        dy = (bbox[1] + bbox[3] / 2) - (old_bbox[1] + old_bbox[3] / 2)
        if abs(dx) > 2 or abs(dy) > 2:
            self.polys = [p + np.array([dx, dy]) for p in self.polys]
            self.polys = [p.astype(np.int32) for p in self.polys]
        return True

    def fade_alpha(self, forget_frames):
        """1.0 when freshly seen, fading to 0 as age approaches forget_frames."""
        if forget_frames <= 0:
            return 1.0
        return max(0.0, 1.0 - self.age_since_seen / forget_frames)


def match_detections_to_tracks(dets, tracks, iou_thr=0.15):
    """Greedy IoU matching (by class) between this frame's fresh detections
    and existing tracked instances, so a re-detection updates the SAME track
    instead of spawning a duplicate."""
    used_tracks, used_dets = set(), set()
    pairs = []
    for di, (cls_idx, _n, polys, _c) in enumerate(dets):
        best_t, best_iou = None, iou_thr
        db = poly_bbox(max(polys, key=cv2.contourArea))
        for ti, t in enumerate(tracks):
            if ti in used_tracks or t.cls_idx != cls_idx:
                continue
            tb = poly_bbox(max(t.polys, key=cv2.contourArea))
            x1, y1 = max(db[0], tb[0]), max(db[1], tb[1])
            x2 = min(db[0] + db[2], tb[0] + tb[2])
            y2 = min(db[1] + db[3], tb[1] + tb[3])
            if x2 <= x1 or y2 <= y1:
                continue
            inter = (x2 - x1) * (y2 - y1)
            union = db[2] * db[3] + tb[2] * tb[3] - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou, best_t = iou, ti
        if best_t is not None:
            pairs.append((di, best_t))
            used_tracks.add(best_t)
            used_dets.add(di)
    return pairs, used_dets, used_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--mode", choices=["lb", "crop"], default="lb")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--conf-scale", type=float, default=1.0)

    # tracking between detections
    ap.add_argument("--detect-every", type=int, default=1,
                    help="run full detection every Nth frame; 1 = every frame "
                         "(tracking off). Try 3-5 for a speed boost.")
    ap.add_argument("--forget-after", type=float, default=2.5,
                    help="seconds an unconfirmed track is kept (fading out) "
                         "before being dropped")

    # display toggles
    ap.add_argument("--no-scores", action="store_true",
                    help="hide the confidence number after each label")
    ap.add_argument("--no-threshold-print", action="store_true",
                    help="silence the startup terminal line listing thresholds")
    ap.add_argument("--no-labels", action="store_true",
                    help="hide text labels entirely, masks only")
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

    has_tracker = hasattr(cv2, "TrackerCSRT_create")
    if a.detect_every > 1 and not has_tracker:
        print("NOTE: cv2.TrackerCSRT_create unavailable (need opencv-contrib-python).")
        print("      pip install opencv-contrib-python")
        print("      Falling back to frozen-shape hold (no motion following).")

    src = int(a.source) if str(a.source).isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")
    W = int(cap.get(3)) or 1280
    H = int(cap.get(4)) or 720
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    forget_frames = int(a.forget_after * fps_src)

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
    tracks = []
    frame_idx = 0
    fixed_bb = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        frame_idx += 1
        run_detection = (frame_idx % max(1, a.detect_every)) == 1 or a.detect_every <= 1

        if a.mode == "crop":
            if fixed_bb is None:
                fixed_bb = scope_bbox(frame) or (0, 0, min(W, H))
            x0, y0, s = fixed_bb
            inp = cv2.resize(frame[y0:y0 + s, x0:x0 + s], (a.imgsz, a.imgsz),
                             interpolation=cv2.INTER_AREA)
            meta, mode = fixed_bb, "crop"
        else:
            sq, m = pad_square_black(frame)
            inp = cv2.resize(sq, (a.imgsz, a.imgsz), interpolation=cv2.INTER_AREA)
            meta, mode = m, "lb"

        disp = (cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
                if disp_scale != 1.0 else frame.copy())

        if run_detection:
            res = model.predict(inp, imgsz=a.imgsz, conf=min_conf, verbose=False,
                                max_det=a.max_det)[0]
            dets = []
            if res.masks is not None and len(res.masks.data):
                data = res.masks.data.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                cnf = res.boxes.conf.cpu().numpy()
                for m_, c, s in zip(data, cls, cnf):
                    if c >= len(NAMES) or s < CONF.get(NAMES[c], 0.25):
                        continue
                    polys = mask_to_display_polys(m_ > 0.5, mode, meta, disp_scale)
                    if polys:
                        dets.append((int(c), NAMES[c], polys, float(s)))

            pairs, used_dets, used_tracks = match_detections_to_tracks(dets, tracks)
            for di, ti in pairs:
                cls_idx, name, polys, s = dets[di]
                tracks[ti].update_from_detection(polys, s, disp)
            for di, (cls_idx, name, polys, s) in enumerate(dets):
                if di in used_dets:
                    continue
                tracks.append(TrackedInstance(cls_idx, name, COLORS[cls_idx],
                                              polys, disp, s))
            for ti, t in enumerate(tracks):
                if ti not in used_tracks:
                    t.advance_by_tracking(disp) if a.detect_every > 1 else None
                    if a.detect_every <= 1:
                        t.age_since_seen += 1
        else:
            for t in tracks:
                t.advance_by_tracking(disp)

        tracks = [t for t in tracks if t.fade_alpha(forget_frames) > 0.02]

        layer = np.zeros_like(disp)
        labels = []
        present = Counter()
        for t in tracks:
            a_mult = t.fade_alpha(forget_frames)
            col = tuple(int(c * a_mult) for c in t.color)
            cv2.fillPoly(layer, t.polys, col)
            present[t.name] += 1
            if not a.no_labels:
                p = max(t.polys, key=cv2.contourArea)
                txt = t.name if a.no_scores else f"{t.name} {t.conf:.2f}"
                labels.append((txt, int(p[:, 0].min()), int(p[:, 1].min()),
                              t.color, a_mult))

        cv2.addWeighted(layer, a.alpha, disp, 1.0, 0, dst=disp)
        for text, x, y, col, a_mult in labels:
            pt = (max(2, x), max(18, y - 6))
            cv2.putText(disp, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 3, cv2.LINE_AA)
            fcol = tuple(int(c * a_mult + 255 * (1 - a_mult) * 0) for c in col)
            cv2.putText(disp, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        fcol, 1, cv2.LINE_AA)

        times.append(time.time() - t0)
        fps = 1 / (sum(times) / len(times)) if times else 0
        tag = f"{a.mode} {a.imgsz}" + (f"  track x{a.detect_every}" if a.detect_every > 1 else "")
        cv2.putText(disp, f"{fps:.1f} FPS  [{tag}]", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, ", ".join(present.keys()), (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

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
