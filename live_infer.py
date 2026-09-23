# #!/usr/bin/env python3
# """
# live_infer.py -- real-time overlay for surgical video.

#     python live_infer.py --source 0                  # webcam / capture card
#     python live_infer.py --source clip.mp4
#     python live_infer.py --source clip.mp4 --save out.mp4

# Preprocessing here MUST match training: the model was trained on square
# letterboxed 512px frames, so live frames are letterboxed the same way and the
# masks are mapped back onto the original 16:9 for display. Feeding raw 16:9
# frames instead silently degrades accuracy.

# Detections are smoothed over a short window because frame-independent
# segmentation flickers, and a structure blinking on and off reads as far less
# trustworthy to a clinician than a slightly imperfect boundary.
# """

# import argparse, time
# from collections import deque, defaultdict
# import cv2, numpy as np
# from ultralytics import YOLO

# WEIGHTS = r"D:\Study\CDC Project 1\Project\runs\segment\surgical\weights\best.pt"
# IMGSZ = 512
# NAMES = [
#     "external iliac artery",
#     "external iliac vein",
#     "obturator nerve",
#     "ovary",
#     "ureter",
#     "uterine artery",
#     "uterus",
#     "instruments",
# ]
# # Per-class confidence. Rare structures get a lower bar on purpose: a missed
# # ureter matters more than an extra candidate the surgeon can dismiss. Tune
# # these on your val split.
# CONF = {
#     "external iliac artery": 0.01,
#     "external iliac vein": 0.01,
#     "obturator nerve": 0.01,
#     "ovary": 0.01,
#     "ureter": 0.01,
#     "uterine artery": 0.01,
#     "uterus": 0.01,
#     "instruments": 0.01,
# }
# SMOOTH = 5


# def letterbox_square(img, size):
#     h, w = img.shape[:2]
#     s = max(h, w)
#     top, left = (s - h) // 2, (s - w) // 2
#     sq = cv2.copyMakeBorder(
#         img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
#     )
#     return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA), (top, left, s)


# def unletterbox_mask(mask, meta, out_hw):
#     top, left, s = meta
#     m = cv2.resize(mask.astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
#     return m[top : top + out_hw[0], left : left + out_hw[1]]


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--source", default="0")
#     ap.add_argument("--save", default=None)
#     ap.add_argument("--no-smooth", action="store_true")
#     a = ap.parse_args()

#     model = YOLO(WEIGHTS)
#     src = int(a.source) if a.source.isdigit() else a.source
#     cap = cv2.VideoCapture(src)
#     if not cap.isOpened():
#         raise SystemExit(f"cannot open source: {a.source}")

#     writer = None
#     if a.save:
#         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
#         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
#         fps = cap.get(cv2.CAP_PROP_FPS) or 25
#         writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#     rng = np.random.default_rng(0)
#     colors = {
#         i: tuple(int(v) for v in rng.integers(60, 255, 3)) for i in range(len(NAMES))
#     }
#     history = defaultdict(lambda: deque(maxlen=SMOOTH))
#     times = deque(maxlen=30)

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         t0 = time.time()
#         H, W = frame.shape[:2]
#         sq, meta = letterbox_square(frame, IMGSZ)
#         res = model.predict(
#             sq, imgsz=IMGSZ, verbose=False, conf=min(CONF.values()) if CONF else 0.25
#         )[0]

#         overlay = frame.copy()
#         present = set()
#         if res.masks is not None:
#             data = res.masks.data.cpu().numpy()
#             cls = res.boxes.cls.cpu().numpy().astype(int)
#             cnf = res.boxes.conf.cpu().numpy()
#             for m, c, s in zip(data, cls, cnf):
#                 if s < CONF.get(NAMES[c], 0.25):
#                     continue
#                 present.add(c)
#                 full = unletterbox_mask(m > 0.5, meta, (H, W))
#                 overlay[full.astype(bool)] = colors[c]
#                 ys, xs = np.where(full)
#                 if len(xs):
#                     cv2.putText(
#                         frame,
#                         f"{NAMES[c]} {s:.2f}",
#                         (int(xs.min()), max(18, int(ys.min()) - 6)),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         colors[c],
#                         2,
#                     )
#         frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

#         if not a.no_smooth:
#             for i in range(len(NAMES)):
#                 history[i].append(i in present)
#             stable = [
#                 NAMES[i]
#                 for i in range(len(NAMES))
#                 if sum(history[i]) > len(history[i]) / 2
#             ]
#         else:
#             stable = [NAMES[i] for i in present]

#         times.append(time.time() - t0)
#         fps_now = 1 / (sum(times) / len(times)) if times else 0
#         cv2.putText(
#             frame,
#             f"{fps_now:.1f} FPS",
#             (10, 28),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2,
#         )
#         cv2.putText(
#             frame,
#             ", ".join(stable[:5]),
#             (10, 56),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             (255, 255, 255),
#             1,
#         )

#         if writer:
#             writer.write(frame)
#         cv2.imshow("surgical segmentation", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     if writer:
#         writer.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
live_infer.py -- real-time overlay for surgical video.

    python live_infer.py --source 0
    python live_infer.py --source test2.avi
    python live_infer.py --source test2.avi --save out.mp4
    python live_infer.py --source test2.avi --smooth-masks   # temporal EMA on overlay

Preprocessing matches training: square letterbox to 512px with GRAY (114) padding,
which is what Ultralytics uses internally. Masks are mapped back to the original
frame via contour transforms rather than full-resolution mask upscaling -- the
latter costs O(max(H,W)^2) per detection and is what tanks FPS on 1080p+ sources.

Display is decoupled from inference: the window is downscaled to --display-width
so a 1080p/4K source doesn't open a window larger than the screen.
"""

import argparse, time
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS = r"D:\Study\CDC Project 1\Project\runs\segment\surgical\weights\best.pt"
IMGSZ = 512
PAD_VALUE = (114, 114, 114)  # must match Ultralytics' training-time letterbox

# Per-class confidence. Rare structures get a lower bar on purpose: a missed
# ureter matters more than an extra candidate the surgeon can dismiss.
# NOTE: setting these to ~0.0 floods NMS and fills max_det with junk. Keep a
# real floor while debugging.
CONF = {
    "external iliac artery": 0.01,
    "external iliac vein": 0.01,
    "obturator nerve": 0.01,
    "ovary": 0.01,
    "ureter": 0.01,
    "uterine artery": 0.01,
    "uterus": 0.01,
    "instruments": 0.01,
}
DEFAULT_CONF = 0.15
SMOOTH = 5


def letterbox_square(img, size):
    """Pad to square with gray, then resize. Returns (square_img, meta)."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(
        img,
        top,
        s - h - top,
        left,
        s - w - left,
        cv2.BORDER_CONSTANT,
        value=PAD_VALUE,
    )
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA), (top, left, s)


def mask_to_display_polys(mask, meta, disp_scale, min_area=40):
    """
    Convert a small (model-resolution) binary mask into polygons in DISPLAY
    coordinates. Avoids upscaling the mask to max(H,W)^2, which is the main
    source of frame-time cost on high-resolution sources.
    """
    top, left, s = meta
    mh, mw = mask.shape[:2]
    kx, ky = s / mw, s / mh

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = (pts[:, 0] * kx - left) * disp_scale
        pts[:, 1] = (pts[:, 1] * ky - top) * disp_scale
        polys.append(pts.astype(np.int32))
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--save", default=None)
    ap.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Downscale the preview window to this width. 0 = native.",
    )
    ap.add_argument(
        "--smooth-masks",
        action="store_true",
        help="Temporal EMA on the overlay layer (reduces flicker).",
    )
    ap.add_argument(
        "--ema",
        type=float,
        default=0.6,
        help="EMA weight for --smooth-masks. Higher = more damping.",
    )
    ap.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smoothing of the class-presence text line.",
    )
    ap.add_argument("--alpha", type=float, default=0.4, help="Mask opacity.")
    ap.add_argument("--device", default=None, help="e.g. 0, cpu")
    ap.add_argument("--max-det", type=int, default=30)
    a = ap.parse_args()

    model = YOLO(WEIGHTS)

    # Trust the checkpoint, not a hardcoded list. A mismatched class order
    # silently mislabels everything and applies the wrong per-class threshold.
    NAMES = [model.names[i] for i in range(len(model.names))]
    print(f"model classes ({len(NAMES)}): {NAMES}")

    try:
        import torch

        dev = (
            a.device
            if a.device is not None
            else ("0" if torch.cuda.is_available() else "cpu")
        )
        print(f"cuda available: {torch.cuda.is_available()} | using device: {dev}")
    except Exception:
        dev = a.device

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"source: {src_w}x{src_h} @ {src_fps:.1f} fps")

    disp_scale = 1.0
    if a.display_width and src_w > a.display_width:
        disp_scale = a.display_width / src_w
    disp_w, disp_h = int(round(src_w * disp_scale)), int(round(src_h * disp_scale))
    print(f"display: {disp_w}x{disp_h} (scale {disp_scale:.3f})")

    writer = None
    if a.save:
        writer = cv2.VideoWriter(
            a.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (disp_w, disp_h)
        )

    rng = np.random.default_rng(0)
    colors = {
        i: tuple(int(v) for v in rng.integers(60, 255, 3)) for i in range(len(NAMES))
    }
    history = defaultdict(lambda: deque(maxlen=SMOOTH))
    times = deque(maxlen=30)
    ema_layer = None

    win = "surgical segmentation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)

    min_conf = min(CONF.values()) if CONF else DEFAULT_CONF

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        sq, meta = letterbox_square(frame, IMGSZ)

        disp = (
            cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            if disp_scale != 1.0
            else frame.copy()
        )

        res = model.predict(
            sq,
            imgsz=IMGSZ,
            verbose=False,
            conf=min_conf,
            max_det=a.max_det,
            device=dev,
        )[0]

        mask_layer = np.zeros_like(disp)
        labels = []
        present = set()

        if res.masks is not None and len(res.masks.data):
            data = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cnf = res.boxes.conf.cpu().numpy()

            for m, c, s in zip(data, cls, cnf):
                if c >= len(NAMES):
                    continue
                if s < CONF.get(NAMES[c], DEFAULT_CONF):
                    continue
                polys = mask_to_display_polys(m > 0.5, meta, disp_scale)
                if not polys:
                    continue
                present.add(c)
                cv2.fillPoly(mask_layer, polys, colors[c])
                p = max(polys, key=cv2.contourArea)
                x, y = p[:, 0].min(), p[:, 1].min()
                labels.append((f"{NAMES[c]} {s:.2f}", int(x), int(y), colors[c]))

        if a.smooth_masks:
            f = mask_layer.astype(np.float32)
            ema_layer = f if ema_layer is None else a.ema * ema_layer + (1 - a.ema) * f
            mask_layer = ema_layer.astype(np.uint8)

        # Blend masks first, THEN draw text -- otherwise labels get washed out.
        cv2.addWeighted(mask_layer, a.alpha, disp, 1.0, 0, dst=disp)
        for text, x, y, col in labels:
            cv2.putText(
                disp,
                text,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                disp,
                text,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                1,
                cv2.LINE_AA,
            )

        if not a.no_smooth:
            for i in range(len(NAMES)):
                history[i].append(i in present)
            stable = [
                NAMES[i]
                for i in range(len(NAMES))
                if len(history[i]) == SMOOTH and sum(history[i]) > SMOOTH / 2
            ]
        else:
            stable = [NAMES[i] for i in sorted(present)]

        times.append(time.time() - t0)
        fps_now = 1 / (sum(times) / len(times)) if times else 0
        cv2.putText(
            disp,
            f"{fps_now:.1f} FPS",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            ", ".join(stable[:5]),
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

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
