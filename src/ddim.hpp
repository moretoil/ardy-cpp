// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  DDIM 去噪采样（确定性，eta = 0）
// =============================================================================
//
// ARDY 的扩散过程把“动作潜变量”从纯噪声逐步还原成干净样本。DDIM 的每一步
// 都做同一件事：网络预测出“这一步眼中的最终结果 x0”，再按时间表把它和当前
// 带噪样本混合，得到上一步的样本。
//
// 我们没有用学习式的 scheduler，而是照搬 ARDY 发布时量化的推理时间表：
// 只跑 10 步（timesteps = [9,8,…,0]），比起标准的 1000 步调度，这是“蒸馏”
// 之后的结果 —— 用 10 次网络前向换一段 2 秒动作。
//
// 注意两个容易踩坑的地方：
//
//  1) 只更新“生成区”，历史区必须逐字节保持不变。
//     窗口里同时装着历史 token（前若干 token）和待生成 token（后 10 个），
//     历史部分是已知条件，如果连它也一起混合，自回归就会漂移。
//
//  2) 所有中间运算都用 double，落盘时才转 float32。
//     JavaScript 读 Float32Array 得到的是 double 参与运算，只在写回时
//     Math.fround 舍入一次。如果这里用 float 做中间量，就会多一次舍入，
//     结果和浏览器对不上（也会与 Python 参考实现产生 1e-4 量级偏差）。

#include <cmath>
#include <cstddef>
#include <vector>

namespace ardy {

/// 扩散时间表。
///
/// alphas_cumprod 按“基础模型的 timestep”索引（这里恰好是 0..9），
/// 数值单调不增：下标 0 最干净（alpha≈最大值），下标 9 最噪。
/// alphas_cumprod_prev 存储上一步的同一个量，其中 prev[0] 约定为 1.0。
struct DdimSchedule {
    std::vector<int> timesteps;              ///< 推理顺序（从最噪到最干净）
    std::vector<double> alphas_cumprod;      ///< ᾱ_t
    std::vector<double> alphas_cumprod_prev; ///< ᾱ_{t-1}

    std::size_t steps() const { return timesteps.size(); }
};

/// 单步所需的三个数。
struct DdimStep {
    int timestep;          ///< 传给去噪网络的 timestep 值
    double alpha;          ///< ᾱ_t
    double alpha_previous; ///< ᾱ_{t-1}
};

/// 取出第 index 个推理步的时间参数（index 是“推理序号”，timestep 才是网络输入）。
inline DdimStep ddim_step(const DdimSchedule& schedule, std::size_t index) {
    const int timestep = schedule.timesteps.at(index);
    return DdimStep{timestep, schedule.alphas_cumprod.at(static_cast<std::size_t>(timestep)),
                    schedule.alphas_cumprod_prev.at(static_cast<std::size_t>(timestep))};
}

/// 原地更新 sample[start, end)。
///
/// DDIM(eta=0) 的更新公式（x_t 为当前带噪样本，x0 为网络预测的干净样本）：
///
///     ε   = (x_t / sqrt(ᾱ_t) − x0) / sqrt(1/ᾱ_t − 1)     // 反解出预估噪声
///     x_{t-1} = x0 · sqrt(ᾱ_{t-1}) + ε · sqrt(1 − ᾱ_{t-1})
///
/// 当 ᾱ_t 等于 1 时（第一步），除数为 0，此时 ε 直接取 0，
/// 避免出现 0/0 的 NaN。
inline void apply_ddim_step(std::vector<float>& sample, const std::vector<float>& predicted_x0,
                            const DdimStep& step, std::size_t start, std::size_t end) {
    // 先把四个与下标无关的系数算出来，循环里只做乘加。
    const double sqrt_alpha = std::sqrt(step.alpha);
    const double sqrt_reciprocal_minus_one = std::sqrt(1.0 / step.alpha - 1.0);
    const double sqrt_previous = std::sqrt(step.alpha_previous);
    const double sqrt_one_minus_previous = std::sqrt(1.0 - step.alpha_previous);
    for (std::size_t index = start; index < end; ++index) {
        const double predicted = predicted_x0[index];
        const double epsilon = (sqrt_reciprocal_minus_one == 0.0)
                                   ? 0.0
                                   : (static_cast<double>(sample[index]) / sqrt_alpha - predicted) /
                                         sqrt_reciprocal_minus_one;
        sample[index] = static_cast<float>(predicted * sqrt_previous + epsilon * sqrt_one_minus_previous);
    }
}

}  // namespace ardy
