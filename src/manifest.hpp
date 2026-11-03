// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  model.json 的 C++ 映射
// =============================================================================
//
// 模型目录里的 model.json 记录了推理需要的全部元数据（不含权重）：
// 张量尺寸、扩散时间表、root 归一化统计量、FSQ 量化参数、骨骼拓扑、以及
// 每个 ONNX 图的输入输出名字。这个文件把它读进强类型结构体，之后的代码就
// 不用再碰字符串键了。
//
// 之所以把 ONNX 的名字也放进配置：不同版本的导出可能改过输入名
// （例如 constrained / 非 constrained 两套图），硬编码名字很脆弱。
// 这里统一从配置查名字，换一份导出只要改 JSON。

#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ddim.hpp"
#include "json.hpp"

namespace ardy {

/// Core40 的全部张量尺寸。
///
/// 命名里 token 和 frame 是两个不同粒度：1 个 token = 4 帧，
/// 所以 max_tokens=20 对应 max_frames=80。
struct Dimensions {
    int fps = 20;                    ///< 帧率（动作播放速度）
    int num_frames_per_token = 4;    ///< 每个 token 覆盖的帧数
    int max_tokens = 20;             ///< 窗口 token 上限（历史 10 + 生成 10）
    int max_frames = 80;             ///< 窗口帧上限
    int generation_tokens = 10;      ///< 每轮生成的 token 数
    int generation_frames = 40;      ///< 每轮生成的帧数（=2 秒）
    int history_tokens = 10;         ///< 保留的历史 token 数
    int history_frames = 40;         ///< 保留的历史帧数
    int root_features_per_frame = 5; ///< 每帧 root 特征数：x, y, z, cosθ, sinθ
    int nframe_root_dim = 20;        ///< 一个 token 内 root 部分的维度（5×4）
    int latent_dim = 128;            ///< FSQ 身体潜变量维度
    int hybrid_dim = 148;            ///< 混合 token 维度（20 + 128）
    int motion_dim = 330;            ///< 解码器输出的完整动作特征维度
    int body_dim = 325;              ///< 其中身体部分维度
    int text_condition_dim = 2048;   ///< 文本条件向量维度
    int num_joints = 27;             ///< 骨骼关节数（Core27）
};

/// root 轨迹的归一化统计量与字段位置。
///
/// 模型内部把 root 特征标准化成零均值单位方差；重定心时需要还原成真实坐标，
/// 所以这里存着 mean/std，以及“哪个下标是 x/z/朝向”的映射。
struct RecenterConfig {
    std::vector<double> root_mean;                  ///< 每维均值（长度 5）
    std::vector<double> root_std;                   ///< 每维标准差（长度 5）
    std::array<int, 3> position_indices{0, 1, 2};   ///< x/y/z 在 5 维中的位置
    std::array<int, 2> heading_indices{3, 4};       ///< cos/sin 在 5 维中的位置
};

/// FSQ 量化参数：每维的级数，以及量化前后的归一化统计量。
struct LatentQuantization {
    std::vector<int> levels;        ///< 每维离散级数（Core40 全为 64）
    std::vector<double> mean;       ///< 量化后反归一化用
    std::vector<double> stddev;     ///< 同上
};

/// Core27 骨骼：父节点索引 + 关节名 + T-pose 关节位置。
///
/// parents 里 -1 表示根节点（Hips）；其余节点都指向“更早出现的某个节点”，
/// 因此天然构成一棵有序树，正向运动学可以按顺序一遍算完。
struct Skeleton {
    std::string name;                        ///< 如 "cskel27"
    int root_index = 0;                      ///< 根节点下标
    std::vector<int> parents;                ///< 长度 = num_joints
    std::vector<std::string> joint_names;    ///< 长度 = num_joints
    std::vector<std::array<double, 3>> neutral_joints;  ///< T-pose 下的关节世界坐标
};

/// 一个 ONNX 图在 manifest 里的描述：文件名 + 语义名到实际张量名的映射。
struct GraphSpec {
    std::string model;                            ///< 相对模型目录的文件名
    std::map<std::string, std::string> inputs;    ///< 语义名 → ONNX 输入名
    std::map<std::string, std::string> outputs;   ///< 语义名 → ONNX 输出名
};

/// 整个 manifest 的运行时视图。
struct Config {
    Dimensions dims;
    RecenterConfig recenter;
    LatentQuantization quantization;
    Skeleton skeleton;
    DdimSchedule schedule;
    double default_cfg_weight = 2.0;   ///< 模型自带的 CFG 默认值（演示页用 3.5）
    int max_prompt_tokens = 128;       ///< 分词上限
    int max_output_frames = 200;       ///< 单次请求的帧数上限（10 秒）
    std::string model_id;              ///< 用于在启动横幅里显示
    GraphSpec text_encoder;
    GraphSpec denoiser;
    GraphSpec decoder;

    /// 按语义名查 ONNX 输入名；缺失即报错（说明 manifest 与代码不匹配）。
    std::string input_name(const GraphSpec& graph, const std::string& semantic) const {
        auto it = graph.inputs.find(semantic);
        if (it == graph.inputs.end()) throw std::runtime_error("graph is missing input " + semantic);
        return it->second;
    }

    /// 按语义名查 ONNX 输出名。
    std::string output_name(const GraphSpec& graph, const std::string& semantic) const {
        auto it = graph.outputs.find(semantic);
        if (it == graph.outputs.end()) throw std::runtime_error("graph is missing output " + semantic);
        return it->second;
    }
};

/// 一次性读入整个文本文件（JSON 很小，不需要流式解析）。
inline std::string read_text_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

/// 解析 graphs.<name> 这一个节点。
inline GraphSpec parse_graph(const json::Value& value) {
    GraphSpec graph;
    graph.model = value["model"].as_string();
    for (const auto& [key, entry] : value["inputs"].members) graph.inputs[key] = entry.as_string();
    for (const auto& [key, entry] : value["outputs"].members) graph.outputs[key] = entry.as_string();
    return graph;
}

/// 读取并校验 model.json，转换成 Config。
///
/// 这里不做“缺字段就用默认值”的容错：manifest 与代码不匹配时，静默用默认值
/// 会让推理结果莫名其妙地错，不如直接抛异常。
inline Config load_config(const std::string& manifest_path) {
    const json::Value root = json::parse(read_text_file(manifest_path));
    Config config;

    // ---- 张量尺寸 ----
    const json::Value& dims = root["dimensions"];
    config.dims.fps = dims["fps"].as_int();
    config.dims.num_frames_per_token = dims["num_frames_per_token"].as_int();
    config.dims.max_tokens = dims["max_tokens"].as_int();
    config.dims.max_frames = dims["max_frames"].as_int();
    config.dims.generation_tokens = dims["generation_tokens"].as_int();
    config.dims.generation_frames = dims["generation_frames"].as_int();
    config.dims.history_tokens = dims["history_tokens"].as_int();
    config.dims.history_frames = dims["history_frames"].as_int();
    config.dims.root_features_per_frame = dims["root_features_per_frame"].as_int();
    config.dims.nframe_root_dim = dims["nframe_root_dim"].as_int();
    config.dims.latent_dim = dims["latent_dim"].as_int();
    config.dims.hybrid_dim = dims["hybrid_dim"].as_int();
    config.dims.motion_dim = dims["motion_dim"].as_int();
    config.dims.body_dim = dims["body_dim"].as_int();
    config.dims.text_condition_dim = dims["text_condition_dim"].as_int();
    config.dims.num_joints = dims["num_joints"].as_int();

    // ---- root 重定心参数 ----
    const json::Value& recenter = root["recenter"];
    config.recenter.root_mean = recenter["root_mean"].to_doubles();
    config.recenter.root_std = recenter["root_std"].to_doubles();
    const std::vector<int> positions = recenter["position_indices"].to_ints();
    const std::vector<int> headings = recenter["heading_indices"].to_ints();
    for (int i = 0; i < 3; ++i) config.recenter.position_indices[i] = positions[static_cast<std::size_t>(i)];
    for (int i = 0; i < 2; ++i) config.recenter.heading_indices[i] = headings[static_cast<std::size_t>(i)];

    // ---- FSQ 量化参数 ----
    const json::Value& quantization = root["latent_quantization"];
    config.quantization.levels = quantization["levels"].to_ints();
    config.quantization.mean = quantization["mean"].to_doubles();
    config.quantization.stddev = quantization["std"].to_doubles();

    // ---- 骨骼拓扑 ----
    const json::Value& skeleton = root["skeleton"];
    config.skeleton.name = skeleton["name"].as_string();
    config.skeleton.root_index = skeleton["root_index"].as_int();
    config.skeleton.parents = skeleton["parents"].to_ints();
    config.skeleton.joint_names = skeleton["joint_names"].to_strings();
    for (const json::Value& joint : skeleton["neutral_joints"].items) {
        const std::vector<double> values = joint.to_doubles();
        config.skeleton.neutral_joints.push_back({values[0], values[1], values[2]});
    }

    // ---- 扩散时间表 ----
    const json::Value& diffusion = root["diffusion"];
    config.schedule.timesteps = diffusion["timesteps"].to_ints();
    config.schedule.alphas_cumprod = diffusion["alphas_cumprod"].to_doubles();
    config.schedule.alphas_cumprod_prev = diffusion["alphas_cumprod_prev"].to_doubles();

    // ---- 生成与分词限制 ----
    config.default_cfg_weight = root["generation"]["default_cfg_weight"].as_double();
    config.max_output_frames = root["generation"]["max_frames"].as_int();
    if (root.has("tokenizer")) {
        config.max_prompt_tokens = root["tokenizer"]["max_length"].as_int();
    }
    if (root.has("model")) {
        config.model_id = root["model"]["id"].as_string();
    }

    // ---- 三个 ONNX 图的文件与张量名 ----
    config.text_encoder = parse_graph(root["graphs"]["text_encoder"]);
    config.denoiser = parse_graph(root["graphs"]["denoiser"]);
    config.decoder = parse_graph(root["graphs"]["decoder"]);
    return config;
}

}  // namespace ardy
