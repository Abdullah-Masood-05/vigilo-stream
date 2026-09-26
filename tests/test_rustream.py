import os
import tempfile
import numpy as np
import pytest

import rustream
from rustream import (
    BBox,
    FaceDetection,
    Frame,
    FusionEngine,
    Gaze,
    HeadPose,
    ObjectDetection,
    Pipeline,
    Signals,
    create_synthetic_frame,
)


def test_version():
    assert rustream.__version__ == "1.0.1"


def test_vigilo_stream_import():
    import vigilo_stream
    assert vigilo_stream.__version__ == "1.0.1"
    assert vigilo_stream.Frame is rustream.Frame
    assert vigilo_stream.FusionEngine is rustream.FusionEngine
    assert vigilo_stream.Pipeline is rustream.Pipeline
    assert "face_detection_yunet_2023mar.onnx" in vigilo_stream.MODEL_URLS
    assert callable(vigilo_stream.download_models)
    assert callable(rustream.download_models)


def test_synthetic_frame_zero_copy():
    # Create 640x480 RGB8 frame with known color (R=200, G=100, B=50)
    w, h = 640, 480
    frame = create_synthetic_frame(w, h, seq=100, r=200, g=100, b=50)

    assert frame.width == w
    assert frame.height == h
    assert frame.seq == 100
    assert frame.shape == (h, w, 3)
    assert frame.size == w * h * 3
    assert not frame.is_empty

    # Verify memoryview
    mv = memoryview(frame.as_bytes())
    assert len(mv) == w * h * 3
    assert mv[0] == 200
    assert mv[1] == 100
    assert mv[2] == 50

    # Verify zero-copy NumPy array via __array_interface__
    arr = np.asarray(frame)
    assert arr.shape == (h, w, 3)
    assert arr.dtype == np.uint8
    assert arr[0, 0, 0] == 200
    assert arr[0, 0, 1] == 100
    assert arr[0, 0, 2] == 50

    # Verify to_numpy() method
    arr2 = frame.to_numpy()
    assert np.array_equal(arr, arr2)

    # Check that memory is shared directly without copying
    frame_ptr = frame.__array_interface__["data"][0]
    arr_ptr = arr.__array_interface__["data"][0]
    assert frame_ptr == arr_ptr, "NumPy array must share memory pointer with Rust Frame"


def test_types_and_serialization():
    bbox = BBox(10.0, 20.0, 100.0, 150.0)
    assert bbox.x == 10.0
    assert bbox.area == 15000.0
    assert bbox.center == (60.0, 95.0)

    d = bbox.to_dict()
    assert d["x"] == 10.0
    assert d["area"] == 15000.0

    face = FaceDetection(bbox, score=0.98, landmarks=[(15.0, 25.0), (35.0, 25.0), (25.0, 40.0), (18.0, 60.0), (32.0, 60.0)])
    assert face.score == pytest.approx(0.98, abs=1e-4)
    assert len(face.landmarks) == 5
    assert face.to_dict()["score"] == pytest.approx(0.98, abs=1e-4)

    pose = HeadPose(15.5, -5.0, 2.0)
    assert pose.yaw_deg == pytest.approx(15.5)
    assert pose.pitch_deg == pytest.approx(-5.0)
    assert pose.roll_deg == pytest.approx(2.0)
    assert pose.to_dict()["yaw_deg"] == pytest.approx(15.5)

    gaze = Gaze(yaw_rad=0.3, pitch_rad=-0.1, eye_yaw_rad=0.05, eye_pitch_rad=-0.02)
    assert gaze.yaw_rad == pytest.approx(0.3, abs=1e-4)
    assert gaze.to_dict()["yaw_rad"] == pytest.approx(0.3, abs=1e-4)

    obj = ObjectDetection("cell phone", 0.88, bbox, class_id=67)
    assert obj.label == "cell phone"
    assert obj.score == pytest.approx(0.88, abs=1e-4)
    assert obj.to_dict()["label"] == "cell phone"


def test_fusion_engine_step():
    engine = FusionEngine()
    assert engine.active() == []

    # Step with an empty frame (no face)
    signals = Signals(
        seq=1,
        t_ms=100,
        faces=[],
        head_pose=None,
        gaze=None,
        objects=[],
        identity_match=None,
    )
    events = engine.step(signals, 100)
    assert isinstance(events, list)

    # Finish at end of session
    fin_events = engine.finish(200)
    assert isinstance(fin_events, list)


def test_fusion_engine_replay_and_config():
    engine = FusionEngine()

    # Create a minimal JSONL recording
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"seq":1,"t_ms":100,"faces":[],"head_pose":null,"gaze":null,"objects":[],"identity_match":null}\n')
        f.write('{"seq":2,"t_ms":200,"faces":[],"head_pose":null,"gaze":null,"objects":[],"identity_match":null}\n')
        temp_path = f.name

    try:
        events = engine.replay(temp_path)
        assert isinstance(events, list)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Test hot-reload config
    custom_toml = """
    [thresholds.face]
    no_face_hold_ms = 4000
    no_face_clear_ms = 500
    never_seen_ms = 10000
    multi_face_hold_ms = 1000
    multi_face_clear_ms = 500
    """
    engine.update_config(custom_toml)
    assert engine.active() == []


def test_pipeline_instantiation_and_context_manager():
    with Pipeline(auto_download=False, device="cpu") as pipeline:
        assert not pipeline.is_running()
        assert pipeline.snapshot() is None
        assert pipeline.poll_frame() is None
        assert pipeline.events() == []
        assert not pipeline.is_enrolled()

    with Pipeline(auto_download=False, device="auto") as pipeline:
        assert not pipeline.is_running()


def test_gpu_detection_and_device_handling():
    import vigilo_stream

    is_supported, reason = vigilo_stream.detect_gpu_support()
    assert isinstance(is_supported, bool)
    assert isinstance(reason, str)

    provider, is_gpu = vigilo_stream.device_info()
    assert isinstance(provider, str)
    assert isinstance(is_gpu, bool)
    assert provider == "CPU"
    assert is_gpu is False

    is_cached = vigilo_stream.is_gpu_cached(vigilo_stream.__version__)
    assert isinstance(is_cached, bool)

    guidance = vigilo_stream.get_hardware_guidance()
    assert isinstance(guidance, dict)
    assert "intel_igpu" in guidance
    assert "amd_apu" in guidance
    assert "11th Gen" in guidance["intel_igpu"]["recommended_generations"]
    assert "Ryzen 6000" in guidance["amd_apu"]["recommended_generations"]

