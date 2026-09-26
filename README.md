# vigilo-stream

[![PyPI](https://img.shields.io/badge/pypi-vigilo--stream-blue)](https://pypi.org/project/vigilo-stream/)
[![Documentation](https://img.shields.io/badge/docs-vigilo--stream-orange?logo=vitepress)](https://abdullah-masood-05.github.io/vigilo-stream/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.80+-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Maturin](https://img.shields.io/badge/Maturin-1.15-purple)](https://github.com/PyO3/maturin)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

Zero-copy multi-modal stream fusion engine for real-time AI pipelines in Python.

`vigilo-stream` provides Python bindings for the stream fusion engine in [`vigilo-core`](https://github.com/Abdullah-Masood-05/vigilo-core). It gives Python vision and proctoring pipelines direct access to video frames and temporal rule evaluation without copying memory across the FFI boundary.

- Zero-copy buffer sharing: Frame memory allocated in Rust is exposed directly to NumPy and PyTorch through `__array_interface__` and the buffer protocol.
- Lock-free frame exchange: Capture workers publish frames through `ArcSwap` slots, discarding stale frames automatically instead of building queues.
- Deterministic temporal fusion: The `FusionEngine` processes detection signals through configurable hysteresis bands, hold timers, and score accumulators. Given the same input, replay produces identical events.
- Multimodal detection: Wraps the `vigilo-core` inference pipeline for face detection (YuNet), head pose (MobileNetV3), gaze estimation (MobileGaze), object detection (YOLOX-Nano), and identity matching (ArcFace).

## Documentation

Comprehensive usage guides and API references are available in the **[Documentation](https://abdullah-masood-05.github.io/vigilo-stream/)**:

- [Getting started and installation](https://abdullah-masood-05.github.io/vigilo-stream/guide/getting-started)
- [Zero-copy memory sharing](https://abdullah-masood-05.github.io/vigilo-stream/guide/zero-copy)
- [Pipeline lifecycle](https://abdullah-masood-05.github.io/vigilo-stream/guide/pipeline)
- [Fusion engine and replay](https://abdullah-masood-05.github.io/vigilo-stream/guide/fusion-engine)
- [Live OpenCV HUD](https://abdullah-masood-05.github.io/vigilo-stream/guide/opencv-hud)
- [Python API reference](https://abdullah-masood-05.github.io/vigilo-stream/api/pipeline)
- [Hardware and GPU acceleration](https://abdullah-masood-05.github.io/vigilo-stream/hardware/acceleration)

## Installation

```bash
pip install vigilo-stream
```

You can import the library using either `vigilo_stream` or the `rustream` alias.

## Quick start

```python
import vigilo_stream
import numpy as np

# 1. Zero-copy frame operations (no neural model files required)
frame = vigilo_stream.create_synthetic_frame(1280, 720, seq=1, r=255, g=0, b=0)
print(frame.width, frame.height, frame.shape) # 1280 720 (720, 1280, 3)

# Expose Rust memory directly as a NumPy array without copying
arr = np.asarray(frame)
assert arr.__array_interface__["data"][0] == frame.__array_interface__["data"][0]

# 2. Vision and proctoring pipeline
# Pipeline automatically downloads default model weights on first run
with vigilo_stream.Pipeline(models_dir="models") as pipe:
    pipe.start("camera:0")  # Accepts "camera:0", "file:clip.mp4", or "dir:frames/"

    while pipe.is_running():
        frame = pipe.poll_frame()
        if frame:
            img = np.asarray(frame)

        snapshot = pipe.snapshot()
        if snapshot:
            print(f"Faces: {snapshot.face_count}, Pose: {snapshot.head_pose}")

        events = pipe.events()
        for event in events:
            print(f"Violation: {event}")

# 3. Headless deterministic stream fusion (no neural models or camera required)
engine = vigilo_stream.FusionEngine()
events = engine.replay("recorded_session.jsonl")
print(f"Replayed session produced {len(events)} events.")
```

## Real-time OpenCV visualization with HUD

You can stream frames directly into OpenCV with zero-copy buffer access, overlay bounding boxes with confidence scores, draw 5 facial landmarks, project a 3D head pose orientation gizmo, render gaze direction rays, tag prohibited objects, and display proctoring telemetry cards and violation status pills matching the desktop viewer.

![Live OpenCV HUD](examples/sample_hud_output.jpg)

### Complete example

```python
import cv2
import math
import time
import numpy as np
import vigilo_stream

# Initialize pipeline with automatic model download
pipe = vigilo_stream.Pipeline(models_dir="models", auto_download=True)
pipe.start("camera:0")  # Accepts "camera:0", "file:clip.mp4", or "dir:frames/"

cv2.namedWindow("Vigilo Stream Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vigilo Stream Viewer", 1280, 720)

active_violations = set()
held_objects = []
last_object_time = 0.0

try:
    while pipe.is_running():
        # 1. Zero-copy frame access: raw pointer shared directly with NumPy
        frame = pipe.poll_frame()
        if frame is None:
            continue

        rgb = np.asarray(frame)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 2. Instantaneous model snapshot and temporal fusion events
        snap = pipe.snapshot()
        for ev in pipe.events():
            if ev.event_type == "ViolationStarted" and ev.violation:
                active_violations.add(ev.violation.kind)
            elif ev.event_type == "ViolationEnded" and ev.violation:
                active_violations.discard(ev.violation.kind)

        # 3. Draw face bounding boxes, 5 landmarks, 3D pose, and gaze rays
        if snap:
            for i, face in enumerate(snap.faces):
                bx = int(face.bbox.x)
                by = int(face.bbox.y)
                bw = int(face.bbox.w)
                bh = int(face.bbox.h)

                # Bounding box with score badge
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (94, 197, 34), 2)
                cv2.putText(
                    img,
                    f"FACE {int(face.score * 100)}%",
                    (bx, max(18, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (140, 255, 100),
                    1,
                    cv2.LINE_AA,
                )

                # 5 Facial landmarks (right eye, left eye, nose, right mouth, left mouth)
                palette = [
                    (94, 197, 34),
                    (94, 197, 34),
                    (21, 204, 250),
                    (113, 113, 248),
                    (113, 113, 248),
                ]
                for pt, col in zip(face.landmarks, palette):
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 4, col, -1, cv2.LINE_AA)

                # 3D Head pose orientation axes (Euler projection)
                if i == 0 and snap.head_pose:
                    cx, cy = int(bx + bw * 0.5), int(by + bh * 0.5)
                    s = min(bw, bh) * 0.45
                    rad = math.pi / 180.0
                    y_rad = -snap.head_pose.yaw_deg * rad
                    p_rad = snap.head_pose.pitch_deg * rad
                    r_rad = snap.head_pose.roll_deg * rad

                    cyaw, syaw = math.cos(y_rad), math.sin(y_rad)
                    cpit, spit = math.cos(p_rad), math.sin(p_rad)
                    crol, srol = math.cos(r_rad), math.sin(r_rad)

                    x_end = (
                        int(cx + s * (cyaw * crol)),
                        int(cy + s * (cpit * srol + crol * spit * syaw)),
                    )
                    y_end = (
                        int(cx + s * (-cyaw * srol)),
                        int(cy + s * (cpit * crol - spit * syaw * srol)),
                    )
                    z_end = (int(cx + s * syaw), int(cy + s * (-cyaw * spit)))

                    cv2.arrowedLine(img, (cx, cy), x_end, (68, 68, 239), 2, tipLength=0.2)
                    cv2.arrowedLine(img, (cx, cy), y_end, (94, 197, 34), 2, tipLength=0.2)
                    cv2.arrowedLine(img, (cx, cy), z_end, (250, 165, 96), 2, tipLength=0.2)

                # Gaze direction ray originating between the eyes
                if i == 0 and snap.gaze and len(face.landmarks) >= 2:
                    ox = int((face.landmarks[0][0] + face.landmarks[1][0]) * 0.5)
                    oy = int((face.landmarks[0][1] + face.landmarks[1][1]) * 0.5)
                    glen = bw * 1.1
                    gdx = -glen * math.sin(snap.gaze.yaw_rad) * math.cos(snap.gaze.pitch_rad)
                    gdy = -glen * math.sin(snap.gaze.pitch_rad)
                    cv2.arrowedLine(
                        img,
                        (ox, oy),
                        (int(ox + gdx), int(oy + gdy)),
                        (252, 171, 240),
                        2,
                        tipLength=0.15,
                    )

            # Prohibited objects (phones, books, secondary devices)
            # The object worker runs at 1 Hz to save compute, while face models run at 30 Hz.
            # Hold the latest detected objects for 1.2s to render a steady bounding box.
            if snap.objects:
                held_objects = snap.objects
                last_object_time = time.time()
            elif "prohibited_object" not in active_violations and (time.time() - last_object_time > 1.2):
                held_objects = []

            for obj in held_objects:
                ox = int(obj.bbox.x)
                oy = int(obj.bbox.y)
                ow = int(obj.bbox.w)
                oh = int(obj.bbox.h)
                cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (68, 68, 239), 2)
                cv2.putText(
                    img,
                    f"{obj.label.upper()} {int(obj.score * 100)}%",
                    (ox, max(18, oy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

        # 4. Display frame and handle interactive keys
        cv2.imshow("Vigilo Stream Viewer", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("e"):
            pipe.enrol()

finally:
    pipe.stop()
    cv2.destroyAllWindows()
```

You can also run the full modular viewer script with HUD telemetry cards and violation status pills in [`examples/live_opencv_hud.py`](examples/live_opencv_hud.py):

```bash
python examples/live_opencv_hud.py --source camera:0
```


## Model weights

The neural pipeline uses ONNX Runtime models:
- Face detection: YuNet (`face_detection_yunet_2023mar.onnx`)
- Head pose: MobileNetV3 (`headpose_mobilenetv3_small.onnx`)
- Gaze estimation: MobileGaze (`mobileone_s0_gaze.onnx`)
- Object detection: YOLOX-Nano (`yolox_nano.onnx`)

By default, `Pipeline(models_dir="models")` downloads missing models on first use. You can also download them explicitly:

```python
import vigilo_stream
vigilo_stream.download_models("models")
```

Alternatively, download them using curl:

```bash
mkdir -p models
curl -sSL -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sSL -o models/headpose_mobilenetv3_small.onnx https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx
curl -sSL -o models/mobileone_s0_gaze.onnx https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx
curl -sSL -o models/yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
```

## Hardware & GPU acceleration

`vigilo-stream` ships with a fast, multithreaded CPU engine by default (~18 MB wheel). For hardware acceleration, it provides on-demand dynamic execution providers for Windows, Linux, and macOS:

| Platform | Execution Provider | Hardware Support |
| :--- | :--- | :--- |
| **Windows** | **DirectML** (DirectX 12) | NVIDIA GeForce/RTX, AMD Radeon, Intel Arc / Iris Xe, Qualcomm |
| **Linux** | **CUDA** | NVIDIA discrete GPUs (Driver + CUDA runtime) |
| **macOS** | **CoreML** (Metal / ANE) | Apple Silicon (M1, M2, M3, M4) |

### Integrated GPU (iGPU) recommendations

GPU acceleration is fully functional on integrated graphics via DirectML. For real-time inference throughput:
- **Intel iGPUs**: **Recommended for 11th Gen Core and newer** (Tiger Lake, Alder Lake, Raptor Lake, Core Ultra / Meteor Lake, Lunar Lake) with **Intel Iris Xe Graphics** or **Intel Arc Graphics**. Earlier 10th Gen and older processors (Gen 9/9.5 UHD 620/630) lack DP4a deep learning instructions, so the default multi-threaded CPU engine performs equal to or faster than the iGPU.
- **AMD APUs**: **Recommended for AMD Ryzen 6000 series and newer** (Zen 3+, Zen 4, Zen 5) featuring **RDNA 2, RDNA 3, or RDNA 3.5** integrated graphics (e.g., Radeon 680M, 780M, 890M) with hardware WMMA / matrix acceleration. Earlier Vega-based APUs (Ryzen 2000–5000) are supported, but CPU mode is typically comparable.
- **Memory Tip**: Ensure system RAM is configured in **dual-channel** mode, as integrated GPUs share unified memory bandwidth with the CPU.

```python
import vigilo_stream

# Run on GPU (downloads and loads platform backend on first use if missing)
pipeline = vigilo_stream.Pipeline(device="gpu")

# Automatic selection: uses GPU if already cached/available, falls back to CPU
pipeline = vigilo_stream.Pipeline(device="auto")
```

For full details, benchmarks, and advanced configuration, see the [Hardware Acceleration Documentation](https://abdullah-masood-05.github.io/vigilo-stream/hardware/acceleration).

## Architecture

```
Camera / Video File / Image Directory
                 │
                 ▼
  FrameSource (DirectShow / FFmpeg)
                 │
                 ▼
     ArcSwap Latest-Frame Slot  ◄── Zero-copy pointer sharing with NumPy
        ┌────────┴────────┐
        ▼                 ▼
   Face Worker      Object Worker
  YuNet+Pose+Gaze     YOLOX-Nano
        └────────┬────────┘
                 ▼
              Signals ──► FusionEngine ──► Events / Violations
```

## Building from source

Requirements:
- Rust 1.80 or newer
- Python 3.9 or newer
- C++ build tools (MSVC on Windows, GCC/Clang on Linux and macOS)

```bash
# Set up a virtual environment and install build tools
uv venv
uv pip install maturin pytest numpy

# Build and install the extension into the active environment
uv run maturin develop

# Run the test suite
uv run pytest -v tests/
```

## Release notes

### v1.0.1

- **Cross-Platform On-Demand GPU Acceleration**:
  - **Windows**: Microsoft DirectML (DirectX 12) acceleration for NVIDIA, AMD, Intel Arc, and Qualcomm GPUs.
  - **Linux**: NVIDIA CUDA acceleration.
  - **macOS**: Apple CoreML / Metal acceleration for Apple Silicon (M1–M4).
- **Tested Integrated GPU (iGPU) Compatibility**: Verified support on integrated graphics; optimal performance recommended on Intel 11th Gen Core+ (Iris Xe / Arc) and AMD Ryzen 6000+ (RDNA 2/3 / Radeon 680M/780M).
- **Dynamic Hardware Detection & Lazy Download**: Keeps default PyPI package lightweight (~18 MB); downloads platform GPU runtimes on first use when requested.
- **Multi-Device Pipeline Support**: Added `device="auto"`, `device="gpu"`, and `device="cpu"` parameters to `Pipeline`.
- **Runtime Provider Inspection**: Added `device_info()`, `detect_gpu_support()`, and `enable_gpu()` APIs.
- **Official Documentation Site**: Full VitePress documentation website with dark/light themes published to GitHub Pages.


### v0.1.1

- Added `download_models()` helper to fetch default ONNX model weights automatically.
- Enhanced `Pipeline` to download missing model files automatically on first use (`auto_download=True`).
- Added `MODEL_URLS` mapping and updated documentation.

### v0.1.0

- Initial release of `vigilo-stream` (with `rustream` backward-compatibility alias) targeting Python 3.9 through 3.13.
- Implemented `Frame` with `__array_interface__` and `memoryview()` support for zero-copy NumPy interop.
- Implemented `FusionEngine` with single-frame stepping and deterministic JSONL log replay.
- Implemented `Pipeline` context manager wrapping camera capture, detection workers, and event polling.
- Added data bindings for `BBox`, `FaceDetection`, `HeadPose`, `Gaze`, `ObjectDetection`, `Signals`, `Violation`, and `Event`.
- Multi-platform CI testing across Windows, Ubuntu, and macOS.

## License

AGPL-3.0. See [LICENSE](LICENSE) for details.
