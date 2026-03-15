#!/usr/bin/env python3
"""FaceHide - Live face detection with mosaic effect over all detected faces."""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SSD_PROTO = ("https://raw.githubusercontent.com/opencv/opencv/master/"
             "samples/dnn/face_detector/deploy.prototxt")
SSD_MODEL = ("https://github.com/opencv/opencv_3rdparty/raw/"
             "dnn_samples_face_detector_20170830/"
             "res10_300x300_ssd_iter_140000.caffemodel")

DETECT_MAX_DIM = 640


def download(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved → {dest}")


def apply_mosaic(frame: np.ndarray, x: int, y: int, w: int, h: int, block_size: int) -> None:
    if w <= 0 or h <= 0:
        return
    # Scale block size with face area so large close-up faces are equally obscured
    adaptive_block = max(block_size, max(w, h) // 10)
    roi = frame[y : y + h, x : x + w]
    small = cv2.resize(roi, (max(1, w // adaptive_block), max(1, h // adaptive_block)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def clamp_bbox(x, y, w, h, fw, fh, pad):
    px, py = int(w * pad), int(h * pad)
    return (max(0, x - px), max(0, y - py),
            min(fw, x + w + px) - max(0, x - px),
            min(fh, y + h + py) - max(0, y - py))


def load_yunet(threshold: float):
    path = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        download(YUNET_URL, path)
    return cv2.FaceDetectorYN.create(
        str(path), "", (320, 320),
        score_threshold=threshold, nms_threshold=0.3, top_k=5000,
    )


def yunet_detect(detector, frame: np.ndarray) -> list:
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
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


def load_ssd(threshold: float):
    proto = Path(__file__).parent / "deploy.prototxt"
    model = Path(__file__).parent / "res10_300x300_ssd.caffemodel"
    if not proto.exists():
        download(SSD_PROTO, proto)
    if not model.exists():
        download(SSD_MODEL, model)
    net = cv2.dnn.readNetFromCaffe(str(proto), str(model))

    def detect(frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        dets = net.forward()
        out = []
        for i in range(dets.shape[2]):
            if float(dets[0, 0, i, 2]) >= threshold:
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                out.append((x1, y1, x2 - x1, y2 - y1))
        return out

    return detect


def run_live(input_path, detect, block_size: int, padding: float) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open '{input_path}'", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay  = max(1, int(1000 / fps))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Playing: {input_path}  ({width}x{height} @ {fps:.1f} fps)")
    print("Q/ESC = quit   SPACE = pause   D = start/stop recording")

    window    = "FaceHide — Q/ESC quit  |  SPACE pause"
    win_scale = min(1280 / width, 720 / height, 1.0)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int(width * win_scale), int(height * win_scale))

    paused    = False
    frame_idx = 0
    frame     = None
    writer    = None
    out_path  = None

    def start_recording():
        nonlocal writer, out_path
        p = Path(input_path) if isinstance(input_path, str) else Path("webcam")
        out_path = str(p.parent / f"{p.stem}_hidden{p.suffix if isinstance(input_path, str) else '.mp4'}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Recording → {out_path}")

    def stop_recording():
        nonlocal writer, out_path
        if writer:
            writer.release()
            writer = None
            print(f"Saved: {out_path}")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    break
            frame_idx += 1

            raw = detect(frame)
            boxes = [clamp_bbox(x, y, w, h, width, height, padding)
                     for x, y, w, h in raw]

            for (x, y, w, h) in boxes:
                apply_mosaic(frame, x, y, w, h, block_size)

            if writer:
                writer.write(frame)

            pct   = f"{frame_idx / total * 100:.1f}%" if total > 0 else f"#{frame_idx}"
            rec   = "  ⏺ REC" if writer else ""
            label = f"Faces: {len(boxes)}   {pct}   [D] record  [SPACE] pause  [Q] quit{rec}"
            cv2.putText(frame, label, (10, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (10, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255) if writer else (255, 255, 255), 1, cv2.LINE_AA)

        if frame is not None:
            cv2.imshow(window, frame)

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key in (ord("d"), ord("D")):
            if writer:
                stop_recording()
            else:
                start_recording()

    stop_recording()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="FaceHide — mosaic all detected faces in a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Video file path (or 0 for webcam)")
    parser.add_argument("-b", "--block-size", type=int, default=15)
    parser.add_argument("-p", "--padding",    type=float, default=0.1)
    parser.add_argument("--threshold",        type=float, default=0.4)
    args = parser.parse_args()

    input_path = 0 if args.input == "0" else args.input
    if isinstance(input_path, str) and not Path(input_path).exists():
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        det = load_yunet(args.threshold)
        print("Detector: YuNet")
        detect = lambda frame: yunet_detect(det, frame)
    except Exception as e:
        print(f"YuNet unavailable ({e}), using SSD fallback.")
        detect = load_ssd(args.threshold)
        print("Detector: SSD res10")

    run_live(input_path, detect, args.block_size, args.padding)


if __name__ == "__main__":
    main()
