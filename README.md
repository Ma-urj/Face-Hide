# FaceHide

A real-time face detection tool that applies a mosaic (pixelation) effect over every face found in a video, with live playback and optional download.

---

## Features

- Live playback with mosaic applied over all detected faces
- Adaptive block size — larger faces get stronger pixelation
- Press `D` to start/stop recording the processed stream to disk
- Webcam support
- Automatic model download on first run (no manual setup)

---

## Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
opencv-python>=4.8.0
numpy>=1.24.0
```

Models are downloaded automatically to the project folder on first run:
- `face_detection_yunet_2023mar.onnx` — face detector (YuNet)

---

## Usage

### Mosaic all faces in a video

```bash
python face_hide.py input_video.mp4
```

### Use webcam

```bash
python face_hide.py 0
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-b`, `--block-size` | `15` | Minimum mosaic block size in pixels |
| `-p`, `--padding` | `0.1` | Fractional padding around each detected face |
| `--threshold` | `0.4` | Detection confidence threshold (0–1) |

### Controls

| Key | Action |
|-----|--------|
| `D` | Start / stop recording to `<input>_hidden.mp4` |
| `SPACE` | Pause / resume |
| `Q` / `ESC` | Quit |

---

## Exempt specific faces

A separate tool (`face_hide_exempt.py`) lets you register faces that should **not** be mosaiced. Drop one or more clear face photos into the `faces/` folder, then run:

```bash
python face_hide_exempt.py input_video.mp4
```

Additional models downloaded automatically:
- `face_recognition_sface_2021dec.onnx` — face recognizer (SFace)

Recognition runs **once per new face** when it first appears. Subsequent frames use lightweight IoU tracking — no repeated model calls.

---

## Project Structure

```
FaceHide/
├── face_hide.py              # Main tool — mosaic all faces
├── face_hide_exempt.py       # Exempt registered faces from mosaic
├── faces/                    # Place face photos here for exemption
├── requirements.txt
└── README.md
```

---

## Future Improvements

- **Detection accuracy** — small and distant faces in crowd scenes can be missed; a multi-scale or tiled detection pass would improve coverage
- **Face tracking** — replace IoU matching with a dedicated tracker (e.g. DeepSORT) for smoother mosaic persistence across frames
- **GPU acceleration** — offload YuNet and SFace inference to CUDA/OpenCL for real-time performance on high-resolution footage
- **Batch / offline mode** — process and save a full video without live playback
- **Confidence tuning UI** — adjust detection threshold interactively with a slider during playback
