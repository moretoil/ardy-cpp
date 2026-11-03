// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  自回归窗口与“世界坐标重置”
// =============================================================================
//
// ARDY 的生成方式是“一次出 2 秒”，然后靠循环接着往下写。每一轮都构造一个固定
// 20 个 token 的窗口喂给去噪网络：
//
//     token 0 ──────────────────────────── token 19
//     ┌───────────────────┬───────────────┐
//     │   历史（≤10 个）   │  待生成（10 个）│
//     │   已确定、原样传入  │  这里填高斯噪声 │
//     └───────────────────┴───────────────┘
//
// 每个 token 承载 4 帧，所以 20 token = 80 帧，但最终只交付后半段的 40 帧；
// 前半段只是为了给网络提供“动作连续性”的上下文。
//
// ---------------------------------------------------------------------------
//  两个必须理解的预处理：重定心（recenter）与潜变量重量化（requantize）
// ---------------------------------------------------------------------------
//
//  1) 重定心
//     如果直接沿用世界坐标，角色走得越远，坐标数值越大，模型没见过这种分布，
//     生成质量会崩。所以每轮结束后把“最后一帧的 x/z”减掉，让坐标回到原点附近。
//     被减掉的那部分存进 global_translation，解码时再加回去，世界轨迹不会丢。
//
//  2) 潜变量重量化
//     身体部分是 FSQ（有限标量量化）编出来的 128 维离散潜变量，每维只有 64 级。
//     扩散采样输出的是连续值，喂回下一轮历史前必须重新吸附到那 64 级网格上，
//     否则历史里的值会落在“训练时从未出现过”的位置。
//     舍入要用“四舍六入五成双”（ties-to-even），这是 PyTorch 的行为；
//     JavaScript 的 Math.round 是四舍五入，会在这里产生系统性偏差。

#include <algorithm>
#include <cmath>
#include <utility>
#include <cstddef>
#include <vector>

#include "manifest.hpp"
#include "random.hpp"

namespace ardy {

/// 一次去噪所需的全部张量（都已按 ONNX 输入形状组织）。
struct ArWindow {
    std::vector<float> x;                    ///< [max_tokens × hybrid_dim]，历史 + 噪声
    std::vector<float> history_mask;         ///< [max_frames] 历史帧位置为 1
    std::vector<float> generation_mask;      ///< [max_frames] 待生成帧位置为 1
    std::vector<float> history_token_mask;   ///< [max_tokens] 历史 token 位置为 1
    std::vector<float> generation_token_mask;///< [max_tokens] 待生成 token 位置为 1
    int history_frames = 0;
    int generation_frames = 0;
    int history_tokens = 0;
    int generation_tokens = 0;
    int generation_token_offset = 0;         ///< 待生成区在 token 轴上的起点
};

/// 组装一个 AR 窗口。
///
/// history 为空表示这是第一轮（没有任何上下文），此时整个窗口都是待生成区。
/// 噪声只填待生成区，历史区原样保留 —— 这一点很关键：
/// 如果把历史也重新加噪，模型看到的就不是“已经发生的动作”了。
inline ArWindow create_ar_window(const Dimensions& dims, PortableRandom& rng, const std::vector<float>& history) {
    ArWindow window;
    const std::size_t history_tokens =
        history.empty() ? 0 : history.size() / static_cast<std::size_t>(dims.hybrid_dim);
    window.history_tokens = static_cast<int>(history_tokens);
    window.history_frames = window.history_tokens * dims.num_frames_per_token;
    window.generation_tokens = dims.generation_tokens;
    window.generation_frames = dims.generation_frames;
    window.generation_token_offset = window.history_tokens;

    window.x.assign(static_cast<std::size_t>(dims.max_tokens) * dims.hybrid_dim, 0.0f);
    std::copy(history.begin(), history.end(), window.x.begin());
    // 只有“生成槽”填标准正态噪声，其余保持 0（掩码会告诉网络哪些位置有效）。
    rng.fill_normal(window.x,
                    static_cast<std::size_t>(window.generation_token_offset) * dims.hybrid_dim,
                    static_cast<std::size_t>(window.generation_token_offset + window.generation_tokens) *
                        dims.hybrid_dim);

    window.history_mask.assign(dims.max_frames, 0.0f);
    window.generation_mask.assign(dims.max_frames, 0.0f);
    window.history_token_mask.assign(dims.max_tokens, 0.0f);
    window.generation_token_mask.assign(dims.max_tokens, 0.0f);
    for (int i = 0; i < window.history_frames; ++i) window.history_mask[static_cast<std::size_t>(i)] = 1.0f;
    for (int i = 0; i < window.generation_frames; ++i) {
        window.generation_mask[static_cast<std::size_t>(window.history_frames + i)] = 1.0f;
    }
    for (int i = 0; i < window.history_tokens; ++i) window.history_token_mask[static_cast<std::size_t>(i)] = 1.0f;
    for (int i = 0; i < window.generation_tokens; ++i) {
        window.generation_token_mask[static_cast<std::size_t>(window.generation_token_offset + i)] = 1.0f;
    }
    return window;
}

/// 解码器的帧级掩码：前 valid_tokens×4 帧为 1，其余为 0（padding）。
inline std::vector<float> create_motion_pad_mask(const Dimensions& dims, int valid_tokens) {
    std::vector<float> mask(static_cast<std::size_t>(dims.max_frames), 0.0f);
    const int frames = valid_tokens * dims.num_frames_per_token;
    for (int i = 0; i < frames; ++i) mask[static_cast<std::size_t>(i)] = 1.0f;
    return mask;
}

/// 银行家舍入（四舍六入五成双）：0.5 时向偶数一侧靠。
///
/// 为什么不用 std::round？因为 std::round 是“远离零”舍入，ECMAScript 的
/// Math.round 是“向上”舍入，而 PyTorch（也就是模型训练/导出时用的）是
/// ties-to-even。FSQ 的量化网格边界上，这个差别会让同一个值落到相邻两级，
/// 进而让动作出现细微但可见的差异。
inline double round_ties_to_even(double value) {
    if (!std::isfinite(value)) return value;
    const double floor_value = std::floor(value);
    const double fraction = value - floor_value;
    if (fraction < 0.5) return floor_value;
    if (fraction > 0.5) return floor_value + 1.0;
    return std::fmod(floor_value, 2.0) == 0.0 ? floor_value : floor_value + 1.0;
}

/// 重定心/重量化之后，下一轮需要继承的两样东西。
struct RecenterResult {
    std::array<float, 3> global_translation{0.0f, 0.0f, 0.0f};  ///< 累计的世界坐标偏移
    double first_heading_angle = 0.0;                            ///< 下一段起始朝向（弧度）
};

/// 原地处理一个窗口：root 轨迹重定心 + 身体潜变量重量化。
///
/// 参数 hybrid 是 [max_tokens × hybrid_dim] 的扁平数组，每 148 维的结构：
///     [0  .. 19]  4 帧 × 5 个 root 特征（x, y, z, cosθ, sinθ）
///     [20 ..147]  128 维 FSQ 身体潜变量（每个 token 一份，4 帧共享）
///
/// 返回值给出“这一轮被减掉的世界位移”和“下一轮的历史起始朝向”。
inline RecenterResult recenter_and_requantize(std::vector<float>& hybrid, int valid_tokens,
                                              const Dimensions& dims, const RecenterConfig& recenter,
                                              const LatentQuantization& quantization,
                                              const std::array<float, 3>& global_translation,
                                              int next_history_start_token) {
    const int hybrid_dim = dims.hybrid_dim;
    const int root_token_dim = dims.nframe_root_dim;
    const int root_frame_dim = dims.root_features_per_frame;
    const int frames_per_token = dims.num_frames_per_token;
    const int valid_frames = valid_tokens * frames_per_token;
    const int position_x = recenter.position_indices[0];
    const int position_z = recenter.position_indices[2];
    const int heading_cos = recenter.heading_indices[0];
    const int heading_sin = recenter.heading_indices[1];
    const auto& root_mean = recenter.root_mean;
    const auto& root_std = recenter.root_std;

    // 帧索引 → 扁平下标。root 特征在 token 内部是按帧紧凑排列的。
    auto root_offset = [&](int frame, int feature) -> std::size_t {
        const int token = frame / frames_per_token;
        const int frame_in_token = frame % frames_per_token;
        return static_cast<std::size_t>(token) * hybrid_dim +
               static_cast<std::size_t>(frame_in_token) * root_frame_dim + static_cast<std::size_t>(feature);
    };
    // 取反归一化后的“真实”root 值（乘以 std 加 mean）。
    auto raw_root = [&](int frame, int feature) -> double {
        return static_cast<double>(hybrid[root_offset(frame, feature)]) *
                   root_std[static_cast<std::size_t>(feature)] +
               root_mean[static_cast<std::size_t>(feature)];
    };

    // ---- 第一步：把整段轨迹平移到以最后一帧为原点 ----
    // y 不平移：模型只关心水平位移，高度本来就是绝对量。
    const int center_frame = valid_frames - 1;
    const double center_x = raw_root(center_frame, position_x);
    const double center_z = raw_root(center_frame, position_z);
    for (int frame = 0; frame < valid_frames; ++frame) {
        for (const auto& [feature, center] : {std::pair<int, double>{position_x, center_x},
                                              std::pair<int, double>{position_z, center_z}}) {
            const double translated = raw_root(frame, feature) - center;
            hybrid[root_offset(frame, feature)] = static_cast<float>(
                (translated - root_mean[static_cast<std::size_t>(feature)]) /
                root_std[static_cast<std::size_t>(feature)]);
        }
    }

    // ---- 第二步：把 128 维身体潜变量吸附回 FSQ 网格 ----
    // 每一维的级数是 levels[f]（这里是 64），值域 [-1, 1]，
    // 所以 half_width = 32，量化就是 round(x * 32) / 32。
    for (int token = 0; token < valid_tokens; ++token) {
        const std::size_t token_offset = static_cast<std::size_t>(token) * hybrid_dim;
        for (int feature = 0; feature < dims.latent_dim; ++feature) {
            const std::size_t index = token_offset + static_cast<std::size_t>(root_token_dim) +
                                      static_cast<std::size_t>(feature);
            const double raw = static_cast<double>(hybrid[index]) *
                                   quantization.stddev[static_cast<std::size_t>(feature)] +
                               quantization.mean[static_cast<std::size_t>(feature)];
            const double half_width =
                std::floor(quantization.levels[static_cast<std::size_t>(feature)] / 2.0);
            const double clamped = std::max(-1.0, std::min(1.0, raw));
            const double discrete = round_ties_to_even(clamped * half_width) / half_width;
            hybrid[index] = static_cast<float>(
                (discrete - quantization.mean[static_cast<std::size_t>(feature)]) /
                quantization.stddev[static_cast<std::size_t>(feature)]);
        }
    }

    // ---- 第三步：输出给下一轮的两样信息 ----
    // 世界位移累加；朝向取“下一段历史的第一帧”的方向，
    // 这样模型在窗口交界处不会突然扭头。
    const int first_history_frame = next_history_start_token * frames_per_token;
    RecenterResult result;
    result.global_translation = {
        static_cast<float>(global_translation[0] + center_x),
        global_translation[1],   // y 是绝对高度，不累加
        static_cast<float>(global_translation[2] + center_z),
    };
    result.first_heading_angle =
        std::atan2(raw_root(first_history_frame, heading_sin), raw_root(first_history_frame, heading_cos));
    return result;
}

}  // namespace ardy
