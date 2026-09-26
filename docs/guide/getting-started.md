# Installation & quickstart

This guide walks you through installing `vigilo-stream`, setting up ONNX model weights, and running your first vision pipeline.

## Installation

Install the package from PyPI:

::: code-group
```bash [pip]
pip install vigilo-stream
```

```bash [uv]
uv add vigilo-stream
```

```bash [poetry]
poetry add vigilo-stream
```
:::

You can import the package using either `vigilo_stream` or the legacy alias `rustream`:

```python
import vigilo_stream
# or
import rustream as vigilo_stream
```

## System requirements

- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Operating systems**:
  - Windows 10 / 11 (x86_64)
  - Ubuntu 20.04+ / Debian 11+ (x86_64)
  - macOS 12+ (Apple Silicon arm64 or Intel x86_64)
- **Hardware acceleration (optional)**:
  - **Integrated GPUs (iGPUs / APUs)**: Supported via DirectML. For real-time performance, **Intel 11th Gen Core and newer** (Iris Xe / Intel Arc) or **AMD Ryzen 6000 series and newer** (RDNA 2 / RDNA 3, e.g. Radeon 680M / 780M) are recommended. Older integrated graphics run best on the default CPU engine.
  - **Discrete GPUs**: NVIDIA GeForce/RTX, AMD Radeon RX, or Intel Arc.
  - **Apple Silicon**: M1, M2, M3, M4 (CoreML / Metal).
  - *See full details in the [Hardware & GPU acceleration guide](/hardware/acceleration).*
- **Optional**: OpenCV (`opencv-python` or `opencv-python-headless`) for video display and HUD rendering.

## Model weights

The neural pipeline uses four ONNX Runtime models:

| Task | Architecture | Default filename | Size |
| :--- | :--- | :--- | :--- |
| **Face detection** | YuNet (320x320) | `face_detection_yunet_2023mar.onnx` | ~230 KB |
| **Head pose** | MobileNetV3 Small | `headpose_mobilenetv3_small.onnx` | ~6.1 MB |
| **Gaze estimation** | MobileOne-S0 Gaze | `mobileone_s0_gaze.onnx` | ~5.0 MB |
| **Object detection** | YOLOX-Nano | `yolox_nano.onnx` | ~3.7 MB |

### Automatic download

By default, `Pipeline(models_dir="models", auto_download=True)` checks for missing models and downloads them automatically on first use.

You can also download them explicitly ahead of time:

```python
import vigilo_stream

# Downloads all missing models into the ./models directory
vigilo_stream.download_models("models")
```

### Manual download

If you prefer to download weights manually:

```bash
mkdir -p models
curl -sSL -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sSL -o models/headpose_mobilenetv3_small.onnx https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx
curl -sSL -o models/mobileone_s0_gaze.onnx https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx
curl -sSL -o models/yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
```

## Quickstart

Here is a minimal script running camera capture, neural detection, and temporal event polling:

```python
import time
import numpy as np
import vigilo_stream

# 1. Initialize pipeline with auto-download
pipe = vigilo_stream.Pipeline(models_dir="models", auto_download=True)
pipe.start("camera:0")

print("Pipeline started. Press Ctrl+C to stop.")

try:
    while pipe.is_running():
        # Poll the latest frame (zero-copy RGB8)
        frame = pipe.poll_frame()
        if frame is not None:
            # Direct pointer exposure into NumPy array
            img = np.asarray(frame)
            print(f"Frame seq={frame.seq} shape={img.shape}", end="\r")

        # Poll latest detection signals
        snapshot = pipe.snapshot()
        if snapshot and snapshot.faces:
            face = snapshot.faces[0]
            print(f"\nFace detected! Score: {face.score:.2f} BBox: {face.bbox}")

        # Drain new temporal violation events
        for event in pipe.events():
            print(f"\nAlert: {event.event_type} - {event.violation}")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    pipe.stop()
    print("Pipeline stopped.")
```
