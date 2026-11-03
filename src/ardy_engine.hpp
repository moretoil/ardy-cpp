// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  ARDY Mini 文本生成动作引擎
// =============================================================================
//
// 分工很明确：
//
//   * 神经网络相关的重活全部在三个 ONNX 图里：
//       text_encoder —— MiniLM 学生模型，把提示词变成 2048 维条件向量
//       denoiser     —— 带 CFG 的扩散去噪主干，一次前向 = 一个 DDIM 步
//       decoder      —— FSQ 反量化 + 身体解码 + 正向运动学（FK 已烘焙进图里）
//
//   * 这个类负责“胶水”：扩散循环、自回归窗口状态、世界坐标重定心、时间统计。
//
// 两层结构：
//   ArdyRuntime  —— 持有一个模型目录对应的三个 ONNX 会话，可被多次复用。
//                   加载代价主要在它（fp32 模型 3~4 秒），所以进程内应只建一个。
//   ArdySession  —— 一次“生成会话”，持有随机数状态和历史 token。
//                   窗口之间换提示词是无缝的，因此同一个 session 就支持
//                   交互式实时变招（walk → jump → wave 连续生成）。
//
// 线程安全：本类不加锁。一个 ArdySession 只能在一个线程里顺序调用；
// 多个 session 共享同一个 runtime 时，ONNX Runtime 的会话本身是线程安全的。

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "manifest.hpp"
#include "random.hpp"
#include "window.hpp"

namespace ardy {

/// 一次 step() 产出的动作片段。
///
/// 所有数组都是“帧优先”的连续内存（第 f 帧的数据紧跟第 f-1 帧），
/// 方便直接拷给渲染器或写文件。
struct MotionChunk {
    int start_frame = 0;   ///< 这一段第一帧在整段动作里的序号
    int frame_count = 0;   ///< 这一段包含多少帧

    std::vector<float> motion;                ///< [frame_count × motion_dim] 归一化动作特征
    std::vector<float> posed_joints;          ///< [frame_count × 27 × 3] 关节世界坐标
    std::vector<float> local_rotations;       ///< [frame_count × 27 × 9] 局部旋转矩阵（行主序）
    std::vector<float> root_positions;        ///< [frame_count × 3] 根节点世界坐标
    std::vector<std::uint8_t> foot_contacts;  ///< [frame_count × 4] 四只脚/脚趾的着地标记

    double denoise_ms = 0.0;  ///< 这一段花在扩散上的时间（含 10 步 × 窗口数）
    double decode_ms = 0.0;   ///< 花在解码器上的时间
};

/// 引擎加载选项。
struct EngineOptions {
    std::string device = "cpu";  ///< cpu | dml（DirectML）| cuda | auto
    int adapter_id = -1;         ///< DirectML 用哪块 DXGI 显卡；-1 = 自动挑独显
    int intra_op_threads = 0;    ///< ONNX Runtime 线程数；0 = 交给它自己决定
    double cfg_weight = -1.0;    ///< 文本引导强度；<0 表示用 manifest 里的默认值
    bool verbose = false;        ///< 打印更多诊断信息
};

/// 运行时：持有三个 ONNX 会话 + 一份配置。
///
/// 典型用法：
///     ArdyRuntime runtime("models/fp16", options);
///     ArdySession session(runtime, seed);
///     auto chunks = session.step("A person walks forward.", 40);
class ArdyRuntime {
 public:
    ArdyRuntime(const std::string& model_dir, const EngineOptions& options);
    ~ArdyRuntime();

    ArdyRuntime(const ArdyRuntime&) = delete;
    ArdyRuntime& operator=(const ArdyRuntime&) = delete;

    const Config& config() const { return config_; }
    /// 实际生效的 provider 列表，例如 {"DmlExecutionProvider", "CPUExecutionProvider"}。
    const std::vector<std::string>& providers() const { return providers_; }
    const std::string& model_dir() const { return model_dir_; }
    /// 给人看的设备描述，例如 "NVIDIA GeForce RTX 3050 Laptop GPU (3.9 GB)"。
    const std::string& device_label() const { return device_label_; }

    /// 分词 + 文本编码，返回 2048 维条件向量。
    ///
    /// 每次调用都会真正跑一次 text_encoder；ArdySession 内部会按提示词缓存结果，
    /// 所以同一个 prompt 连续生成时只会编码一次。
    std::vector<float> encode_text(const std::string& prompt);

    /// 跑一次去噪网络的单步前向，输出 pred_x0（形状 [max_tokens × hybrid_dim]）。
    ///
    /// 注意：这里**不包括** DDIM 的数值更新，更新由 ddim.hpp 在 CPU 上做。
    /// 这样设计的好处是 ONNX 图保持纯函数，便于和浏览器/Python 逐位比对。
    void denoise_step(const ArWindow& window, const std::vector<float>& text_conditions,
                      double first_heading_angle, int timestep, std::vector<float>& prediction);

    /// 解码器输出。长度固定为整个窗口（max_frames 帧），有效帧由 pad_mask 决定。
    struct Decoded {
        std::vector<float> normalized_motion;     ///< [max_frames × motion_dim]
        std::vector<float> posed_joints;          ///< [max_frames × j × 3]
        std::vector<float> local_rotations;       ///< [max_frames × j × 9]
        std::vector<float> root_positions;        ///< [max_frames × 3]
        std::vector<std::uint8_t> foot_contacts;  ///< [max_frames × 4]
    };
    void decode(const std::vector<float>& hybrid_tokens, int valid_tokens,
                const std::array<float, 3>& global_translation, Decoded& out);

    int prompt_token_limit() const { return config_.max_prompt_tokens; }

 private:
    struct Impl;   ///< ONNX Runtime 相关细节藏在 .cpp 里，头文件不暴露 onnxruntime
    std::unique_ptr<Impl> impl_;
    Config config_;
    std::string model_dir_;
    std::vector<std::string> providers_;
    std::string device_label_;
    double cfg_weight_ = 2.0;
    bool verbose_ = false;
};

/// 一次生成会话：持有随机数状态与历史 token，因此可以“接着往下写”。
///
/// 关键属性：同一个 session 里，前后两次 step() 生成的动作是连续的；
/// 中途换提示词只会影响之后生成的窗口，不会让角色跳变
/// —— 这正是“交互式”的实质。
class ArdySession {
 public:
    ArdySession(ArdyRuntime& runtime, std::uint32_t seed,
                const std::array<float, 3>& initial_translation = {0.0f, 0.0f, 0.0f},
                double initial_heading = 0.0);

    /// 继续生成 duration_frames 帧。
    ///
    /// 内部按 40 帧一个窗口循环；不足一个窗口时最后一个窗口会少取几帧。
    /// 返回的分段按时间顺序排列。
    std::vector<MotionChunk> step(const std::string& prompt, int duration_frames);

    /// 便捷重载：按秒数生成（内部换算成帧）。
    std::vector<MotionChunk> generate(const std::string& prompt, double seconds) {
        const int frames = static_cast<int>(std::lround(seconds * runtime_.config().dims.fps));
        return step(prompt, frames);
    }

    int frame_count() const { return frame_count_; }
    std::uint32_t seed() const { return rng_.seed(); }

 private:
    /// 交给 create_ar_window 的“上一段历史”。
    struct PreparedHistory {
        std::vector<float> history;                                 ///< 已重定心的历史 token
        int history_tokens = 0;
        int history_frames = 0;
        std::array<float, 3> global_translation{0.0f, 0.0f, 0.0f};  ///< 之前累计的世界位移
        double first_heading_angle = 0.0;                           ///< 历史首帧的朝向
    };

    /// 从 global_hybrid_ 里裁出最近的历史 token，并把 root 轨迹重新定心。
    PreparedHistory prepare_history(int requested_frames) const;

    /// 追加新生成的 token，并只保留最近 history_tokens 个（更早的再也用不到）。
    void append_global_hybrid(const std::vector<float>& tokens);

    /// 把会话截断到某个帧号（按 token 对齐向下取整）。
    void branch_to(int frame);

    ArdyRuntime& runtime_;
    const Config& config_;
    PortableRandom rng_;
    std::vector<float> global_hybrid_;  ///< 滚动缓冲：只保留最近的历史 token
    int frame_count_ = 0;               ///< 会话累计生成的帧数
    std::array<float, 3> initial_translation_;
    double initial_heading_ = 0.0;
    std::string cached_prompt_;         ///< 上一次的提示词（用于避免重复编码）
    std::vector<float> cached_conditions_;
};

}  // namespace ardy
