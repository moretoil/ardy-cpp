# ardy-cpp (中文说明)

[English](README.md) | **简体中文**

基于 [ARDY](https://github.com/nv-tlabs/ardy) 与 [ARDY Mini](https://github.com/intsuc/ardy-mini) 构建的轻量级、自包含 C++17 文本生成人体 3D 动作（Text-to-Motion）推理引擎。

遵循 `sam.cpp` 与 `llama.cpp` 的极客设计哲学：**零 Python、零 PyTorch、零复杂的外部依赖环境**。从 BERT WordPiece 文本分词、10 步 DDIM 扩散采样计算、自回归滑动窗口状态管理，到空间坐标重定心与标准 BVH 动捕动画文件导出，全部由现代化、易读的纯 C++17 实现。

```
"A person walks forward."  ──▶  walk.bvh (Core27 标准人形骨骼)
```

---

## 核心特性

- **100% 纯血 C++17**：
  - 无需安装 Python 解释器、无需配置数 GB 的 PyTorch 运行库。
  - Windows 平台原生支持 DirectML 显卡加速，**无需配置复杂的 NVIDIA CUDA Toolkit 或 cuDNN**（独显、核显、A 卡均可即插即用）。
- **零第三方依赖地狱**：
  - 内置手写单头文件 WordPiece 分词器（直接基于 BERT 词表进行贪心最长匹配，与官方分词 100% 吻合）。
  - 内置手写单头文件递归下降 JSON 解析器（无需引入几十兆的 `nlohmann/json`）。
  - 内置逐位可复现的 Mulberry32 + Marsaglia 极坐标正态随机数发生器。
- **真正的流式自回归与中途换词实时变招（Interactive Live-Prompting）**：
  - 以 40 帧（2 秒）为一个自回归生成窗口。
  - 支持中途随时下发新的文本指令，动作无缝顺滑过渡（例如：角色正在前行时，下发跳跃指令，角色直接在迈步中顺势起跳，无需重置历史）。
- **极速低延迟**：
  - **GPU (DirectML)**：生成一段 2 秒动作仅需 **~80 毫秒**（比实时播放快约 19 倍）。
  - **CPU (INT8)**：生成一段 2 秒动作仅需 **~500 毫秒**（比实时播放快约 4 倍）。
- **通用动捕格式输出**：
  - 直接输出行业通用的标准 `.bvh` 骨骼动画文件（Core27 标准人类 27 关节，ZXY 欧拉角通道）。
  - 导出结果可直接拖入 Blender、Maya、Unity 以及 **Godot 4** 中驱动任意 3D 人形角色。

---

## 1. 下载模型资产

模型权重托管在 Hugging Face 仓库中，下载后解压放置在 `models/` 目录下即可：

| 模型集 | 存储大小 | 推荐运行设备 | 特点说明 |
| :--- | ---: | :--- | :--- |
| **`models/int8`** | **235 MB** | **CPU (默认推荐)** | 手术级块量化 (Block-wise 32) + 保护 6 个敏感 Head，与 FP32 视觉完全零差别 |
| **`models/fp16`** | **372 MB** | **GPU (DirectML 推荐)** | 半精度权重存储 + FP32 纯显卡执行，生成仅需 ~80 毫秒/窗口 |
| **`models/int4`** | **167 MB** | 极简端侧设备 | 4-bit 块量化，体积最小，低能量动作略软 |

每个模型子文件夹内均包含：
- `text_encoder.onnx`：蒸馏版轻量 MiniLM 文本编码器（代替原版 16GB 的 LLaMA-3-8B）。
- `denoiser.onnx`：自回归去噪扩散主干网络（含文本引导 CFG 分支）。
- `decoder.onnx`：FSQ 潜变量解码与正向运动学（FK 已打平进图）。
- `model.json` 与 `vocab.txt`：张量维度定义、采样时间表与分词词表。

---

## 2. 编译构建

### 环境要求
- 支持 C++17 的编译器（MSVC、GCC 或 Clang）。
- CMake 3.16+。
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) 预编译包（CPU 或 DirectML 包）。

### 标准 CMake 构建流程

```bash
# 1. 配置工程并指定 ONNX Runtime 目录
cmake -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime

# 2. 编译 Release 可执行文件
cmake --build build --config Release
```

在 Windows 平台上，构建脚本会自动将 `onnxruntime.dll` 和 `DirectML.dll` 复制到生成的 `ardy-cli.exe` 旁边。

---

## 3. 使用方法

### 基础命令行生成

```bash
# 使用 CPU 生成 6 秒向前走动作（默认使用 INT8 模型）
./build/ardy-cli "A person walks forward." --seconds 6 --out walk.bvh

# 使用 DirectML 显卡加速生成 10 秒跑步动作（需搭配 models/fp16）
./build/ardy-cli "A person is running." --seconds 10 --device dml --model-dir models/fp16 --out run.bvh
```

### 连续动作序列脚本模式

每行输入一句指令，每句生成一个 2 秒窗口，动作自动首尾相接、丝滑过渡：

```bash
# prompts.txt 文件内容:
# A person walks forward.
# A person jumps in place.
# A person waves with the right hand.

./build/ardy-cli --script prompts.txt --out sequence.bvh
```

### 实时交互变招模式

在控制台逐行打字输入指令，小人会在当前姿势上顺势演进：

```bash
./build/ardy-cli --interactive --seconds 2 --out interactive.bvh
```

### 命令行参数一览

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `[prompt]` | 描述动作的英文文本提示词 | — |
| `--seconds N` | 生成的动作时长（秒，推荐 2 的整数倍） | `4.0` |
| `--seed N` | 随机种子（同种子必得完全相同的动作） | `0` |
| `--cfg-weight W` | 文本引导强度（CFG scale） | `3.5` |
| `--out PATH` | 输出 `.bvh` 文件的保存路径 | `output.bvh` |
| `--script PATH` | 指定包含多行文本提示词的脚本文件 | — |
| `--interactive` | 开启交互模式（从终端标准输入实时读入提示词） | `false` |
| `--model-dir DIR` | 模型资产所在的目录 | `models/int8` |
| `--device DEV` | 计算硬件设备：`cpu`、`dml` (DirectML GPU) 或 `auto` | `cpu` |
| `--device-id N` | 强制指定 DirectML 运行的具体显卡设备号 | 自动优选独立显卡 |

---

## 4. 二次开发与游戏引擎嵌入（Godot / Unreal / C++）

`ardy-cpp` 的架构被设计为极易嵌入各种大型工程与游戏引擎中。

```cpp
#include "ardy_engine.hpp"

// 1. 初始化引擎运行时
ardy::EngineOptions options;
options.device = "dml"; // 或 "cpu"
ardy::ArdyRuntime runtime("models/fp16", options);

// 2. 开启一个可持续生成的状态会话
ardy::ArdySession session(runtime, /*seed=*/42);

// 3. 生成 40 帧（2 秒）动作
std::vector<ardy::MotionChunk> chunks = session.step("A person jumps in place.", 40);

for (const auto& chunk : chunks) {
    // chunk.posed_joints: [帧数 × 27 × 3] 关节的三维世界空间坐标
    // chunk.local_rotations: [帧数 × 27 × 9] 每个关节的 3x3 局部旋转矩阵
    // chunk.root_positions: [帧数 × 3] 根节点的绝对位移轨迹
}

// 4. 实时变招（无需重置历史，直接从起跳动作自然过渡到挥手）
std::vector<ardy::MotionChunk> next_chunks = session.step("A person waves right hand.", 40);
```

在 **Godot 4** 中，你可以将 `ardy_engine.hpp` 封装为 GDExtension 节点，把旋转矩阵直接转为 Godot 原生的 `Quaternion`，通过 `Skeleton3D.set_bone_pose_rotation` 直接在游戏内实时驱动角色，无需写入中间文件。

---

## 源码目录结构

```text
src/
├── main.cpp            # 命令行入口与参数解析
├── ardy_engine.hpp/.cpp# 核心推理会话管理与自回归生成循环
├── bvh.hpp             # 纯 C++ 实现的 Core27 骨骼 BVH 导出器
├── ddim.hpp            # 10 步确定性 DDIM 采样数学更新公式
├── dxgi.hpp            # Windows DXGI 独立显卡智能检测与打分
├── json.hpp            # 递归下降 JSON 轻量解析器
├── manifest.hpp        # model.json 元数据映射结构
├── random.hpp          # 逐位可复现的 Mulberry32 + Marsaglia polar 随机数生成器
├── tokenizer.hpp       # 纯 C++ WordPiece 英文分词器（含 BERT 文本规范化）
└── window.hpp          # 自回归滑动窗口维护、Root 重定心与 FSQ 重量化
CMakeLists.txt          # 标准跨平台构建文件
README.md               # 英文说明文档
README_zh.md            # 中文说明文档
```

---

## 致谢与引用

- [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy)：原始 ARDY 自回归动作扩散模型。
- [intsuc/ardy-mini](https://github.com/intsuc/ardy-mini)：蒸馏版 MiniLM 文本编码器与浏览器导出管线。
