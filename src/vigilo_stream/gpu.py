"""
GPU acceleration and dynamic execution provider management for vigilo-stream.

Handles:
- Cross-platform GPU capability detection (DirectML on Windows, CUDA on Linux, CoreML on macOS).
- On-demand downloading and caching of GPU binaries from GitHub Releases.
- Dynamic backend switching between CPU and GPU native modules.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Tuple


def detect_gpu_support() -> Tuple[bool, str]:
    """Detect if the current machine has compatible GPU hardware.

    Note on Integrated GPUs (iGPUs / APUs):
        DirectML supports DirectX 12 compatible integrated GPUs as well as discrete cards.
        For optimal inference throughput:
        - Intel iGPUs: Recommended on 11th Gen Core and newer (Iris Xe / Intel Arc).
        - AMD APUs: Recommended on Ryzen 6000 series and newer (RDNA 2/3 / Radeon 680M/780M).
        Older iGPUs (Intel UHD 620/630, AMD Vega) run best using the default CPU engine.

    Returns:
        (is_supported, backend_name_or_reason)
    """

    sys_name = platform.system()

    if sys_name == "Windows":
        # DirectML runs on any DirectX 12 capable GPU (NVIDIA, AMD, Intel, Qualcomm)
        try:
            d3d12 = ctypes.windll.d3d12
            if d3d12 is not None:
                return True, "DirectML (Windows DirectX 12)"
        except Exception as e:
            return False, f"DirectX 12 not available: {e}"
        return False, "DirectX 12 (d3d12.dll) not found"

    elif sys_name == "Linux":
        # CUDA requires NVIDIA drivers
        if shutil.which("nvidia-smi") is not None:
            return True, "CUDA (Linux NVIDIA)"
        for lib in ("libcuda.so.1", "libcuda.so"):
            try:
                ctypes.cdll.LoadLibrary(lib)
                return True, "CUDA (Linux NVIDIA)"
            except OSError:
                pass
        return False, "NVIDIA GPU driver / CUDA not found"

    elif sys_name == "Darwin":
        # macOS: CoreML works with Apple Silicon Neural Engine / Metal
        machine = platform.machine()
        if machine == "arm64":
            return True, "CoreML (Apple Silicon)"
        return True, "CoreML (macOS Metal)"

    return False, f"Unsupported OS for GPU acceleration: {sys_name}"


def get_hardware_guidance() -> dict[str, Any]:
    """Return hardware compatibility and performance recommendations for CPU vs GPU inference.

    Returns structured guidance covering:
    - Intel integrated GPUs (recommended: 11th Gen Core and newer with Iris Xe / Intel Arc).
    - AMD integrated GPUs (recommended: Ryzen 6000 series and newer with RDNA 2/3 / Radeon 680M/780M).
    - Discrete GPUs (NVIDIA, AMD Radeon, Intel Arc).
    - Apple Silicon (M1-M4).
    """
    return {
        "intel_igpu": {
            "recommended_generations": "11th Gen Core (Tiger Lake, Alder Lake, Raptor Lake, Core Ultra / Meteor Lake, Lunar Lake) and newer",
            "recommended_gpus": ["Intel Iris Xe Graphics (80-96 EUs)", "Intel Arc Graphics (Xe-LPG / Xe2)"],
            "notes": "Features DP4a hardware instructions for accelerated INT8/FP16 dot products. On 10th Gen and older (Gen 9/9.5 UHD 620/630), CPU mode performs equal to or faster than iGPU.",
        },
        "amd_apu": {
            "recommended_generations": "Ryzen 6000, 7000, 8000, and Ryzen AI 300 series (Zen 3+, Zen 4, Zen 5) and newer",
            "recommended_gpus": ["Radeon 660M / 680M (RDNA 2)", "Radeon 740M / 760M / 780M (RDNA 3)", "Radeon 880M / 890M (RDNA 3.5)"],
            "notes": "Features RDNA 2/3/3.5 with WMMA matrix acceleration and high-bandwidth LPDDR5/DDR5. Older Vega APUs (Ryzen 2000-5000) are supported via DirectX 12, but CPU mode is typically comparable.",
        },
        "discrete_gpus": {
            "nvidia": "GeForce GTX 1060+, RTX 20/30/40/50 series, RTX Ada / Quadro (DirectML on Windows, CUDA on Linux)",
            "amd": "Radeon RX 5000, 6000, 7000 series (DirectML on Windows)",
            "intel": "Intel Arc A380, A580, A750, A770, B-series (DirectML on Windows)",
        },
        "apple_silicon": {
            "models": "M1, M2, M3, M4 (Base, Pro, Max, Ultra)",
            "provider": "CoreML (Apple Neural Engine + Metal GPU)",
        },
        "best_practices": [
            "Use dual-channel RAM on laptops/APUs for maximum shared memory bandwidth.",
            "Keep official GPU drivers updated (Intel Arc/Iris Xe or AMD Adrenalin).",
            "Set Windows power profile to Balanced or Best Performance to avoid power-saving throttling.",
        ],
    }


def get_platform_tag() -> str:

    """Return the release archive platform tag."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()

    if sys_name == "windows":
        return "windows-x86_64"
    elif sys_name == "linux":
        return "linux-x86_64" if machine in ("x86_64", "amd64") else f"linux-{machine}"
    elif sys_name == "darwin":
        return "macos-arm64" if machine == "arm64" else "macos-x86_64"
    return f"{sys_name}-{machine}"


def get_gpu_cache_dir(version: str) -> Path:
    """Return local cache directory for the GPU backend binaries."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base / "vigilo_stream" / "backends" / f"v{version}"


def is_gpu_cached(version: str) -> bool:
    """Check if the GPU backend is already downloaded and present in local cache."""
    cache_dir = get_gpu_cache_dir(version)
    if not cache_dir.exists():
        return False
    exts = (".pyd",) if sys.platform == "win32" else (".so", ".dylib")
    return any(f.suffix in exts and "_core" in f.name for f in cache_dir.rglob("*"))


def download_gpu_backend(version: str, verbose: bool = True) -> Path:
    """Download the platform GPU backend archive from GitHub Releases.

    Args:
        version: Package version (e.g. '1.0.0')
        verbose: Whether to print download progress

    Returns:
        Path to the extracted cache directory
    """
    cache_dir = get_gpu_cache_dir(version)
    cache_dir.mkdir(parents=True, exist_ok=True)

    platform_tag = get_platform_tag()
    filename = f"vigilo-stream-gpu-{platform_tag}.zip"
    url = f"https://github.com/Abdullah-Masood-05/vigilo-stream/releases/download/v{version}/{filename}"

    temp_zip = cache_dir / f"download_{filename}"

    if verbose:
        print(f"Downloading GPU backend ({platform_tag}) from {url}...")

    def reporthook(count, block_size, total_size):
        if total_size > 0 and verbose:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\rDownloading GPU backend: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, temp_zip, reporthook=reporthook)
        if verbose:
            print("\nExtracting GPU backend...")

        with zipfile.ZipFile(temp_zip, "r") as z:
            z.extractall(cache_dir)

        temp_zip.unlink(missing_ok=True)
        if verbose:
            print(f"GPU backend ready at: {cache_dir}")
        return cache_dir

    except Exception as e:
        temp_zip.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download GPU backend for {platform_tag} from {url}: {e}\n"
            "You can continue using the default CPU backend or check your internet connection."
        ) from e


def load_gpu_backend(version: str) -> Any:
    """Dynamically load the GPU-enabled _core native module from the cache directory.

    Args:
        version: Package version (e.g. '1.0.0')

    Returns:
        The loaded GPU _core module
    """
    cache_dir = get_gpu_cache_dir(version)
    if not cache_dir.exists():
        raise FileNotFoundError(f"GPU cache directory does not exist: {cache_dir}")

    exts = (".pyd",) if sys.platform == "win32" else (".so", ".dylib")
    candidates = [f for f in cache_dir.rglob("*") if f.suffix in exts and "_core" in f.name]
    if not candidates:
        raise FileNotFoundError(f"No GPU native binary (*_core* in {exts}) found in {cache_dir}")

    gpu_lib_path = candidates[0]

    # On Windows, add both cache_dir and the library directory to DLL search path
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for d in {cache_dir.resolve(), gpu_lib_path.parent.resolve()}:
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass

    module_name = "vigilo_stream._core_gpu"

    spec = importlib.util.spec_from_file_location(module_name, str(gpu_lib_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {gpu_lib_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

