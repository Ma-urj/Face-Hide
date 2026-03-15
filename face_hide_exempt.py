#!/usr/bin/env python3
"""
FaceHide Exempt — mosaic all faces EXCEPT those registered in the faces/ folder.

Place one or more clear face photos in faces/ before running.
Recognition runs once per new face; subsequent frames use IoU tracking.

Usage:
    python face_hide_exempt.py input_video.mp4
    python face_hide_exempt.py input_video.mp4 -f my_faces/
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")

IMG_EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COSINE_THRESHOLD = 0.363
DETECT_MAX_DIM   = 640
DETECT_EVERY     = 5
MAX_MISSED       = 15
IOU_THRESHOLD    = 0.25


def download(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved → {dest}")


def apply_mosaic(frame: np.ndarray, x: int, y: int, w: int, h: int, block_size: int) -> None:
    if w <= 0 or h <= 0:
        return
    roi = frame[y : y + h, x : x + w]
    small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def clamp_bbox(x, y, w, h, fw, fh, pad):
    px, py = int(w * pad), int(h * pad)
    return (max(0, x - px), max(0, y - py),
            min(fw, x + w + px) - max(0, x - px),
            min(fh, y + h + py) - max(0, y - py))


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter)


# ── Face tracker ──────────────────────────────────────────────────────────────

class FaceTrack:
    _counter = 0

    def __init__(self, bbox: tuple, should_mosaic: bool):
        FaceTrack._counter += 1
        self.id            = FaceTrack._counter
        self.bbox          = bbox
        self.should_mosaic = should_mosaic
        self.missed        = 0


class FaceTracker:
    def __init__(self):
        self.tracks: list[FaceTrack] = []

    def update(self, detections: list, frame: np.ndarray,
               recognizer, registered: list) -> None:
        """
        detections: list of (bbox, face_row)
        Recognition only fires for unmatched (new) detections.
        """
        matched_t, matched_d = set(), set()

        pairs = sorted(
            ((bbox_iou(t.bbox, d[0]), ti, di)
             for ti, t in enumerate(self.tracks)
             for di, d in enumerate(detections)
             if bbox_iou(t.bbox, d[0]) >= IOU_THRESHOLD),
            reverse=True,
        )
        for score, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            self.tracks[ti].bbox   = detections[di][0]
            self.tracks[ti].missed = 0
            matched_t.add(ti)
            matched_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.missed += 1

        for di, (bbox, face_row) in enumerate(detections):
            if di in matched_d:
                continue
            should_mosaic = True
            if recognizer is not None and registered:
                try:
                    feat = recognizer.feature(recognizer.alignCrop(frame, face_row))
                    for _, reg_feat in registered:
                        if recognizer.match(feat, reg_feat,
                                            cv2.FaceRecognizerSF_FR_COSINE) >= COSINE_THRESHOLD:
                            should_mosaic = False
                            break
                except Exception:
                    pass
            self.tracks.append(FaceTrack(bbox, should_mosaic))

        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]


# ── Models ────────────────────────────────────────────────────────────────────

def load_yunet(threshold: float):
    path = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        download(YUNET_URL, path)
    return cv2.FaceDetectorYN.create(
        str(path), "", (320, 320),
        score_threshold=threshold, nms_threshold=0.3, top_k=5000,
    )


def yunet_detect(detector, frame: np.ndarray) -> list:
    """Returns list of (bbox, face_row) in original frame coords."""
    h, w = frame.shape[:2]
    scale = min(DETECT_MAX_DIM / max(h, w), 1.0)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
    sh, sw = small.shape[:2]
    detector.setInputSize((sw, sh))
    _, faces = detector.detect(small)
    if faces is None:
        return []
    if scale < 1.0:
        faces = faces.copy()
        faces[:, :4]   /= scale
        faces[:, 5:15] /= scale
    return [((int(f[0]), int(f[1]), int(f[2]), int(f[3])), f) for f in faces]


def load_sface():
    path = Path(__file__).parent / "face_recognition_sface_2021dec.onnx"
    if not path.exists():
        download(SFACE_URL, path)
    return cv2.FaceRecognizerSF.create(str(path), "")


def load_registered(faces_dir: str, detector, recognizer) -> list:
    folder = Path(faces_dir)
    folder.mkdir(exist_ok=True)

    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not images:
        print("  faces/ is empty — all faces will be mosaiced.")
        return []

    registered = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] cannot read {img_path.name}")
            continue
        dets = yunet_detect(detector, img)
        if not dets:
            print(f"  [skip] no face detected in {img_path.name}")
            continue
        _, best_row = max(dets, key=lambda d: d[1][14])
        try:
            feat = recognizer.feature(recognizer.alignCrop(img, best_row))
            registered.append((img_path.stem, feat))
            print(f"  [ok]   {img_path.name}")
        except Exception as e:
            print(f"  [skip] {img_path.name}: {e}")

    return registered


# ── Live playback ─────────────────────────────────────────────────────────────

def run_live(input_path, yunet, recognizer, registered,
             block_size: int, padding: float) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open '{input_path}'", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay  = max(1, int(1000 / fps))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nPlaying : {input_path}  ({width}x{height} @ {fps:.1f} fps)")
    print(f"Exempt  : {len(registered)} registered face(s)")
    print("Q/ESC = quit   SPACE = pause\n")

    window    = "FaceHide Exempt — Q/ESC quit  |  SPACE pause"
    win_scale = min(1280 / width, 720 / height, 1.0)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int(width * win_scale), int(height * win_scale))

    tracker   = FaceTracker()
    paused    = False
    frame_idx = 0
    frame     = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                tracker = FaceTracker()
                ret, frame = cap.read()
                if not ret:
                    break
            frame_idx += 1

            if frame_idx % DETECT_EVERY == 1:
                dets = yunet_detect(yunet, frame)
                padded = [(clamp_bbox(*bbox, width, height, padding), row)
                          for bbox, row in dets]
                tracker.update(padded, frame, recognizer, registered)

            for track in tracker.tracks:
                if track.should_mosaic:
                    apply_mosaic(frame, *track.bbox, block_size)

            pct   = f"{frame_idx / total * 100:.1f}%" if total > 0 else f"#{frame_idx}"
            label = f"Faces: {len(tracker.tracks)}   {pct}   [SPACE] pause  [Q] quit"
            cv2.putText(frame, label, (10, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (10, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if frame is not None:
            cv2.imshow(window, frame)

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FaceHide Exempt — mosaic all faces except registered ones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Video file path (or 0 for webcam)")
    parser.add_argument("-f", "--faces-dir", default="faces",
                        help="Folder of face photos to exempt from mosaic")
    parser.add_argument("-b", "--block-size", type=int, default=15)
    parser.add_argument("-p", "--padding",    type=float, default=0.1)
    parser.add_argument("--threshold",        type=float, default=0.4)
    args = parser.parse_args()

    input_path = 0 if args.input == "0" else args.input
    if isinstance(input_path, str) and not Path(input_path).exists():
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print("Loading models…")
    yunet      = load_yunet(args.threshold)
    recognizer = load_sface()
    print("Detector: YuNet  |  Recognizer: SFace")
    print(f"Loading registered faces from '{args.faces_dir}/'…")
    registered = load_registered(args.faces_dir, yunet, recognizer)

    run_live(input_path, yunet, recognizer, registered,
             args.block_size, args.padding)


if __name__ == "__main__":
    main()
