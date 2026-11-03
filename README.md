# ardy-cpp

**English** | [简体中文](README_zh.md)

[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![Backend](https://img.shields.io/badge/backend-ONNX%20Runtime%20%7C%20DirectML-green.svg)]()

A lightweight, standalone C++17 text-to-motion engine built on [ARDY](https://github.com/nv-tlabs/ardy) & [ARDY Mini](https://github.com/intsuc/ardy-mini).

Designed in the spirit of `sam.cpp` and `llama.cpp`: **zero Python, zero PyTorch, zero external tokenization or math dependencies**. Everything from BERT WordPiece tokenization, 10-step DDIM diffusion sampling, autoregressive sliding-window state management, to world-space coordinate recentering and BVH export runs in plain, readable C++.

```
"A person walks forward."  ──▶  walk.bvh (Core27 Standard Humanoid Skeleton)
```

---

## Key Features

- **100% Pure C++17**: No Python interpreter, no PyTorch runtime, no CUDA Toolkit requirement (runs via DirectML on Windows or CPU).
- **Zero Third-Party Dependency Hell**:
  - Built-in header-only WordPiece tokenizer (greedy longest-match with BERT vocabulary).
  - Built-in header-only JSON parser (no `nlohmann/json` or RapidJSON dependency).
  - Built-in reproducible Mulberry32 + Marsaglia polar RNG.
- **Real-Time Streaming & Interactive Live-Prompt Replanning**:
  - Autoregressive 40-frame (2-second) generation window.
  - Seamless in-flight prompt switching: change text prompts mid-stream without resetting motion history (e.g., smoothly blend from walking to jumping).
- **Blazing Fast**:
  - **GPU (DirectML)**: ~80 ms per 2-second motion chunk (~19x faster than real-time playback).
  - **CPU (INT8)**: ~500 ms per 2-second motion chunk (~4x faster than real-time playback).
- **Standard Motion Capture Output**:
  - Direct export to industry-standard `.bvh` files (27 humanoid joints, ZXY Euler rotations).
  - Directly loadable in Blender, Maya, Unity, and **Godot 4**.

---

## 1. Download Model Assets

The model weights are hosted separately on Hugging Face:

| Model Set | Size | Recommended For | Notes |
| :--- | ---: | :--- | :--- |
| **`models/int8`** | **235 MB** | **CPU (Default)** | Block-wise INT8 with FP32 head protection; visually indistinguishable from FP32 |
| **`models/fp16`** | **372 MB** | **GPU (DirectML)** | Half-precision weight storage; runs at ~80 ms/window on GPUs |
| **`models/int4`** | **167 MB** | Embedded / Minimal | 4-bit block quantization; lowest disk footprint |

Download and place the extracted model directory under `models/` (e.g. `models/int8` or `models/fp16`).

Each model folder contains:
- `text_encoder.onnx`: Distilled MiniLM text conditioning student.
- `denoiser.onnx`: Autoregressive diffusion backbone with classifier-free guidance (CFG).
- `decoder.onnx`: FSQ latent detokenizer and forward kinematics (FK).
- `model.json` & `vocab.txt`: Tensor dimensions, schedule, and vocabulary.

---

## 2. Build from Source

Requirements:
- A C++17 compliant compiler (MSVC, GCC, or Clang).
- CMake 3.16+.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (CPU or DirectML package).

```bash
# 1. Configure with ONNX Runtime directory
cmake -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime

# 2. Build release binary
cmake --build build --config Release
```

On Windows, `onnxruntime.dll` and `DirectML.dll` (if available) will automatically be copied next to the generated `ardy-cli` executable.

---

## 3. Usage

### Command-Line Generation

```bash
# Generate 6 seconds of motion using CPU (INT8 model)
./build/ardy-cli "A person walks forward." --seconds 6 --out walk.bvh

# Generate using GPU via DirectML (requires models/fp16)
./build/ardy-cli "A person is running." --seconds 10 --device dml --model-dir models/fp16 --out run.bvh
```

### Continuous Prompt Script

To generate a continuous, seamless sequence with prompt changes at every 2-second window:

```bash
# prompts.txt (one prompt per 2-second window):
# A person walks forward.
# A person jumps in place.
# A person waves with the right hand.

./build/ardy-cli --script prompts.txt --out sequence.bvh
```

### Interactive Live Prompting

Type instructions on the fly; the character continuously transitions into each new instruction:

```bash
./build/ardy-cli --interactive --seconds 2 --out interactive.bvh
```

### CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `[prompt]` | English text describing the desired motion | — |
| `--seconds N` | Duration in seconds (multiples of 2 recommended) | `4.0` |
| `--seed N` | Random seed for deterministic generation | `0` |
| `--cfg-weight W` | Classifier-free guidance scale | `3.5` |
| `--out PATH` | Path to save the `.bvh` file | `output.bvh` |
| `--script PATH` | File containing one prompt per 2-second window | — |
| `--interactive` | Read prompts from stdin interactively | `false` |
| `--model-dir DIR` | Directory containing model graphs and config | `models/int8` |
| `--device DEV` | Compute device: `cpu`, `dml` (DirectML GPU), or `auto` | `cpu` |
| `--device-id N` | Force specific DXGI adapter index for DirectML | Best Discrete GPU |

---

## 4. Integration (Godot / Game Engines / C++)

`ardy-cpp` is designed to be embedded directly into game engines and custom pipelines.

```cpp
#include "ardy_engine.hpp"

// 1. Initialize runtime
ardy::EngineOptions options;
options.device = "dml"; // or "cpu"
ardy::ArdyRuntime runtime("models/fp16", options);

// 2. Start a continuous streaming session
ardy::ArdySession session(runtime, /*seed=*/42);

// 3. Generate 40 frames (2 seconds)
std::vector<ardy::MotionChunk> chunks = session.step("A person jumps in place.", 40);

for (const auto& chunk : chunks) {
    // chunk.posed_joints: [frames * 27 * 3] 3D world joint positions
    // chunk.local_rotations: [frames * 27 * 9] 3x3 local rotation matrices
    // chunk.root_positions: [frames * 3] Root trajectory translation
}

// 4. In-flight instruction change (smooth transition from jumping to waving)
std::vector<ardy::MotionChunk> next_chunks = session.step("A person waves right hand.", 40);
```

In **Godot 4**, wrap `ardy_engine.hpp` inside a GDExtension node to directly drive `Skeleton3D` via `set_bone_pose_rotation` without writing files to disk.

---

## Repository Structure

```text
src/
├── main.cpp            # CLI entry point
├── ardy_engine.hpp/.cpp# Core inference session & autoregressive loop
├── bvh.hpp             # Zero-dependency BVH exporter (Core27 skeleton)
├── ddim.hpp            # Deterministic DDIM sampler (eta = 0)
├── dxgi.hpp            # Windows DXGI discrete GPU adapter selection
├── json.hpp            # Lightweight recursive-descent JSON parser
├── manifest.hpp        # Model manifest & tensor contract mapping
├── random.hpp          # Bit-exact Mulberry32 + Marsaglia polar RNG
├── tokenizer.hpp       # Plain C++ BERT WordPiece tokenizer
└── window.hpp          # Autoregressive windowing, recentering & FSQ requantization
CMakeLists.txt          # Standard CMake build script
README.md               # Documentation
```

---

## Acknowledgments

- [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) for the original ARDY autoregressive motion diffusion model.
- [intsuc/ardy-mini](https://github.com/intsuc/ardy-mini) for the distilled MiniLM conditioning model and browser export pipeline.
