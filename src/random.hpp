// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  可复现的高斯噪声发生器
// =============================================================================
//
// 这个文件是整个移植“逐位对齐”的地基：扩散模型的起点是一整块标准正态噪声，
// 只要噪声有一个 bit 不同，最终动作就会完全不一样。所以我们没有用 C++ 标准库
// 的 std::normal_distribution（实现随意、跨编译器不保证一致），而是把浏览器端
// 的 PortableRandom 一比一搬了过来：
//
//   1) Mulberry32 —— 32 位整数状态的伪随机数发生器，只用到加法、异或和乘法，
//      因此任何平台都能得到完全相同的 uint32 序列。
//   2) Marsaglia 极坐标法 —— 把两个 [0,1) 均匀数变成一对标准正态数。
//      一次调用产生两个值，第二个用“备用值”缓存起来，下次直接返回。
//   3) 落盘边界统一转 float32 —— 浏览器把噪声写进 Float32Array 时会做一次
//      舍入，这里在返回前 static_cast<float>，保证存进去的位模式一模一样。
//
// 与浏览器实现一致意味着：同一个 --seed，网页演示和 C++ 引擎生成的是同一段
// 动作。（受 glibc/MSVCRT 的 log/sqrt 末位差异影响，极端情况下噪声可能相差
// 1 ULP 以内的双精度值，但转换成 float32 后几乎总是完全相同。）

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ardy {

class PortableRandom {
 public:
    /// 2^32，用于把 uint32 映射成 [0,1) 的均匀分布。
    static constexpr std::uint64_t kRange = 0x100000000ull;

    explicit PortableRandom(std::uint32_t seed) : seed_(seed), state_(seed) {}

    /// 把字符串种子散列成 uint32（FNV-1a）。
    ///
    /// 这样命令行可以直接写 --seed 2（数字）或 --seed hello（字符串），
    /// 两者都会映射到一个确定的初始状态。
    static std::uint32_t seed_from_string(const std::string& seed) {
        std::uint32_t hash = 0x811C9DC5u;   // FNV offset basis
        for (unsigned char byte : seed) {
            hash ^= byte;
            hash *= 0x01000193u;            // FNV prime
        }
        return hash;
    }

    std::uint32_t seed() const { return seed_; }

    /// Mulberry32 的核心一步，返回下一个 32 位无符号随机数。
    ///
    /// 逐行对应 JavaScript 版本：
    ///     state = (state + 0x6d2b79f5) >>> 0;
    ///     let v = state;
    ///     v = Math.imul(v ^ (v >>> 15), v | 1);
    ///     v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    ///     return (v ^ (v >>> 14)) >>> 0;
    ///
    /// 关键点：JS 的整型位运算都会先按 ToInt32/ToUint32 截断成 32 位，
    /// 这里的每次运算也只用 uint32（乘法天然按 2^32 取模），因此结果完全一致。
    std::uint32_t next_uint32() {
        state_ = state_ + 0x6D2B79F5u;              // 以 2^32 回绕的加法
        std::uint32_t value = state_;
        value = multiply(value ^ (value >> 15), value | 1u);
        // JS 里 v ^= v + imul(...) 左边的 v 是 int32，右边是 ToInt32(和)；
        // 对两个 32 位模式做异或，转成无符号结果一样，所以这里直接按位运算。
        const std::int32_t signed_value = static_cast<std::int32_t>(value);
        const std::uint32_t combined = value + multiply(value ^ (value >> 7), value | 61u);
        value = static_cast<std::uint32_t>(signed_value) ^ combined;
        return value ^ (value >> 14);
    }

    /// 映射到 (0,1) 开区间：加 1 再除以 2^32+1，避免出现精确的 0 或 1。
    double next_float() {
        return (static_cast<double>(next_uint32()) + 1.0) / (static_cast<double>(kRange) + 1.0);
    }

    /// 生成一个标准正态 N(0,1) 样本（Marsaglia 极坐标法）。
    ///
    /// 原理：在单位圆内均匀取点 (x,y)，则
    ///     s = x² + y²
    ///     factor = sqrt(-2·ln(s) / s)
    ///     x·factor 与 y·factor 分别服从标准正态且相互独立
    /// 因此一次循环可以产出两个样本，第二个存进 spare_ 供下次使用。
    ///
    /// 循环条件里排除 s <= DBL_EPSILON（会 log(0)）和 s >= 1（不在单位圆内）。
    float next_normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double x = 0.0;
        double y = 0.0;
        double radius_squared = 0.0;
        do {
            x = 2.0 * next_float() - 1.0;   // 映射到 (-1,1)
            y = 2.0 * next_float() - 1.0;
            radius_squared = x * x + y * y;
        } while (radius_squared <= 2.220446049250313e-16 || radius_squared >= 1.0);
        const double scale = std::sqrt((-2.0 * std::log(radius_squared)) / radius_squared);
        spare_ = static_cast<float>(y * scale);   // 缓存第二个样本
        has_spare_ = true;
        return static_cast<float>(x * scale);     // 在这里完成 float32 舍入
    }

    /// 用正态噪声填充 target[start, end)。
    ///
    /// 只填“生成本窗口”那一段，历史 token 保持原样不动 —— 这一点和浏览器的
    /// createArWindow 完全对应。
    void fill_normal(std::vector<float>& target, std::size_t start, std::size_t end) {
        for (std::size_t index = start; index < end; ++index) target[index] = next_normal();
    }

 private:
    /// 对应 JavaScript 的 Math.imul：把两个操作数当作 32 位补码做乘法，
    /// 结果按 2^32 回绕。C++ 的无符号乘法本来就满足这个语义。
    static std::uint32_t multiply(std::uint32_t a, std::uint32_t b) { return a * b; }

    std::uint32_t seed_;               ///< 初始种子（用于回显）
    std::uint32_t state_;              ///< 递推状态
    float spare_ = 0.0f;               ///< 极坐标法暂存的第二个样本
    bool has_spare_ = false;           ///< spare_ 是否有效
};

}  // namespace ardy
