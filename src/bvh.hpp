// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  BVH 动画导出
// =============================================================================
//
// BVH 是最通用的动捕交换格式：Blender / Maya / MotionBuilder / Unity 的 BVH
// 导入器都能直接读。文件分两段：
//
//   HIERARCHY   骨骼树：每个关节的名字、相对父节点的 OFFSET、通道顺序
//   MOTION      逐帧数据：每帧按 HIERARCHY 声明顺序排列的通道值
//
// 这里有一个容易搞错的地方：BVH 的旋转不是矩阵，是欧拉角，而且“顺序”由
// CHANNELS 声明决定。我们声明 Zrotation Xrotation Yrotation，意味着每个关节
// 的旋转矩阵应当按 R = Rz · Rx · Ry 组合，反解时也必须是这个顺序 —— 否则
// 三方（我们的写、工具的读、视觉结果）会对不上。
//
// 校验方式：tools/verify_bvh.py 把导出的欧拉角重建回矩阵，再和引擎内部的
// 局部旋转矩阵逐元素比对，误差在 3e-7 以内（纯浮点舍入）。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "manifest.hpp"

namespace ardy {

/// 一组 ZXY 欧拉角（单位：度）。字段顺序刻意按“先 Z 后 X 再 Y”书写，
/// 但实际组合顺序见上方说明。
struct EulerZxy {
    double z = 0.0;  ///< 绕 Z 轴旋转（度）
    double x = 0.0;  ///< 绕 X 轴旋转（度）
    double y = 0.0;  ///< 绕 Y 轴旋转（度）
};

/// 行主序 3×3 旋转矩阵 → ZXY 欧拉角。
///
/// 设 R = Rz(z)·Rx(x)·Ry(y)，展开后：
///     R[2][1] = sin(x)
///     R[2][0] = −cos(x)·sin(y),  R[2][2] = cos(x)·cos(y)
///     R[0][1] = −sin(z)·cos(x),  R[1][1] = cos(z)·cos(x)
/// 于是
///     x = asin(R21)
///     y = atan2(−R20, R22)
///     z = atan2(−R01, R11)
/// 用 atan2 而不是 asin/acos，可以自动处理四个象限。
///
/// 当 cos(x) ≈ 0 时进入万向锁，y 和 z 无法分离，此时约定 z = 0，
/// 把全部自由度塞给 y（这是 DCC 工具的常规做法）。
inline EulerZxy matrix_to_euler_zxy(const float* m) {
    // m 为行主序 3x3。
    const double sx = std::max(-1.0, std::min(1.0, static_cast<double>(m[7])));
    const double x = std::asin(sx);
    double y;
    double z;
    if (std::abs(sx) < 0.999999) {
        y = std::atan2(-static_cast<double>(m[6]), static_cast<double>(m[8]));
        z = std::atan2(-static_cast<double>(m[1]), static_cast<double>(m[4]));
    } else {
        // 万向锁：把剩余旋转全部归到 y，z 固定为 0。
        y = std::atan2(static_cast<double>(m[2]), static_cast<double>(m[0]));
        z = 0.0;
    }
    constexpr double kRadToDeg = 57.29577951308232;
    return EulerZxy{z * kRadToDeg, x * kRadToDeg, y * kRadToDeg};
}

/// ZXY 欧拉角 → 旋转矩阵（行主序）。
///
/// 只用于自检（tools/verify_bvh.py 的往返验证），推理路径不需要。
inline void euler_zxy_to_matrix(const EulerZxy& euler, double* out) {
    constexpr double kDegToRad = 0.017453292519943295;
    const double cz = std::cos(euler.z * kDegToRad), sz = std::sin(euler.z * kDegToRad);
    const double cx = std::cos(euler.x * kDegToRad), sx = std::sin(euler.x * kDegToRad);
    const double cy = std::cos(euler.y * kDegToRad), sy = std::sin(euler.y * kDegToRad);
    const double r[9] = {
        cz * cy - sz * sx * sy, -sz * cx, cz * sy + sz * sx * cy,
        sz * cy + cz * sx * sy, cz * cx,  sz * sy - cz * sx * cy,
        -cx * sy,               sx,       cx * cy,
    };
    for (int i = 0; i < 9; ++i) out[i] = r[i];
}

class BvhWriter {
 public:
    explicit BvhWriter(const Config& config) : config_(config) {}

    struct Frame {
        const std::vector<float>* local_rotations;  ///< [num_joints * 9]
        const std::vector<float>* root_positions;   ///< [3]
    };

    /// 写出整个文件。
    ///
    /// rotations 是连续内存：frame_count × num_joints × 9；
    /// root_positions 是 frame_count × 3。
    /// 返回 false 表示尺寸对不上或文件打不开（调用方据此报错）。
    bool write(const std::string& path, const std::vector<float>& rotations,
               const std::vector<float>& root_positions, double fps) const {
        const std::size_t joints = config_.skeleton.parents.size();
        const std::size_t stride = joints * 9;
        if (stride == 0 || rotations.size() % stride != 0) return false;
        const std::size_t frame_count = rotations.size() / stride;
        if (root_positions.size() != frame_count * 3) return false;

        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        out << build_header(frame_count, fps);
        std::vector<float> frame_rotations(stride);
        std::vector<float> frame_root(3);
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            std::copy_n(rotations.begin() + static_cast<std::ptrdiff_t>(frame * stride), stride,
                        frame_rotations.begin());
            std::copy_n(root_positions.begin() + static_cast<std::ptrdiff_t>(frame * 3), 3, frame_root.begin());
            out << format_frame(frame_rotations, frame_root) << "\n";
        }
        return static_cast<bool>(out);
    }

 private:
    /// 生成 HIERARCHY 段 + MOTION 头部（帧数与帧间隔）。
    ///
    /// 先把 parents 数组展开成 children 列表，然后从根节点深度优先递归输出。
    std::string build_header(std::size_t frame_count, double fps) const {
        const Skeleton& skeleton = config_.skeleton;
        std::vector<std::vector<int>> children(skeleton.parents.size());
        for (std::size_t joint = 0; joint < skeleton.parents.size(); ++joint) {
            const int parent = skeleton.parents[joint];
            if (parent >= 0) children[static_cast<std::size_t>(parent)].push_back(static_cast<int>(joint));
        }

        std::string text = "HIERARCHY\n";
        emit_joint(0, 0, children, text);
        char motion[128];
        std::snprintf(motion, sizeof(motion), "MOTION\nFrames: %zu\nFrame Time: %.8f\n", frame_count, 1.0 / fps);
        text += motion;
        return text;
    }

    /// 递归输出一个关节节点（含 OFFSET / CHANNELS / 子节点 / 叶节点 End Site）。
    ///
    /// OFFSET 用的是 T-pose 下“子相对父”的位移，对根节点固定为 0
    /// （根节点的位移由动画数据里的 Xposition 等通道提供）。
    void emit_joint(int joint, int depth, const std::vector<std::vector<int>>& children,
                    std::string& out) const {
        const Skeleton& skeleton = config_.skeleton;
        const std::string indent(static_cast<std::size_t>(depth), '\t');
        const bool is_root = skeleton.parents[static_cast<std::size_t>(joint)] < 0;
        out += indent + (is_root ? "ROOT " : "JOINT ") + skeleton.joint_names[static_cast<std::size_t>(joint)] + "\n";
        out += indent + "{\n";

        double offset[3] = {0.0, 0.0, 0.0};
        if (!is_root) {
            const int parent = skeleton.parents[static_cast<std::size_t>(joint)];
            for (int axis = 0; axis < 3; ++axis) {
                offset[axis] =
                    skeleton.neutral_joints[static_cast<std::size_t>(joint)][static_cast<std::size_t>(axis)] -
                    skeleton.neutral_joints[static_cast<std::size_t>(parent)][static_cast<std::size_t>(axis)];
            }
        }
        char line[256];
        std::snprintf(line, sizeof(line), "%s\tOFFSET %.6f %.6f %.6f\n", indent.c_str(), offset[0], offset[1], offset[2]);
        out += line;
        // 根节点多出三个位置通道（世界位移），其余关节只有旋转。
        out += indent + (is_root ? "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
                                 : "\tCHANNELS 3 Zrotation Xrotation Yrotation\n");

        const auto& child_list = children[static_cast<std::size_t>(joint)];
        if (child_list.empty()) {
            // 叶子关节要显式写一个 End Site，否则有些解析器会报不完整。
            out += indent + "\tEnd Site\n" + indent + "\t{\n" + indent + "\t\tOFFSET 0.0 0.0 0.0\n" + indent + "\t}\n";
        }
        for (int child : child_list) emit_joint(child, depth + 1, children, out);
        out += indent + "}\n";
    }

    /// 把一帧的 27 个关节写成一行数值。
    ///
    /// 顺序必须与上面 CHANNELS 的声明严格一致：根节点是 3 个位置 + 3 个旋转，
    /// 其他节点只有 3 个旋转。
    std::string format_frame(const std::vector<float>& rotations, const std::vector<float>& root) const {
        const Skeleton& skeleton = config_.skeleton;
        std::string line;
        line.reserve(skeleton.parents.size() * 64);
        char buffer[128];
        for (std::size_t joint = 0; joint < skeleton.parents.size(); ++joint) {
            const EulerZxy euler = matrix_to_euler_zxy(&rotations[joint * 9]);
            if (skeleton.parents[joint] < 0) {
                std::snprintf(buffer, sizeof(buffer), "%.6f %.6f %.6f %.6f %.6f %.6f ", root[0], root[1], root[2],
                              euler.z, euler.x, euler.y);
            } else {
                std::snprintf(buffer, sizeof(buffer), "%.6f %.6f %.6f ", euler.z, euler.x, euler.y);
            }
            line += buffer;
        }
        if (!line.empty() && line.back() == ' ') line.pop_back();
        return line;
    }

    Config config_;
};

}  // namespace ardy
