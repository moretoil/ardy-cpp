// SPDX-License-Identifier: Apache-2.0
//
// ArdyRuntime / ArdySession 的实现。
//
// 这里最容易看错的一点：扩散循环本身不在 ONNX 图里。
//   denoiser.onnx 只做“一次前向 = 一个 timestep 的预测”，
//   真正把预测结果揉回样本的 DDIM 更新在 ddim.hpp（纯 CPU 数学）。
// 这么做的好处是与浏览器/Python 参考实现能逐位对齐 —— 数值运算完全可控。

#include "ardy_engine.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "ddim.hpp"
#include "dxgi.hpp"
#include "tokenizer.hpp"

namespace ardy {
namespace {

/// 单调时钟毫秒数，仅用于统计耗时。
double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

/// 构造 ONNX 张量形状的小工具（避免到处写 {a, b} 的花括号歧义）。
std::vector<int64_t> shape2(int64_t a, int64_t b) { return {a, b}; }
std::vector<int64_t> shape1(int64_t a) { return {a}; }

}  // namespace

// ---------------------------------------------------------------------------
// Impl：所有 ONNX Runtime 的对象都藏在这里，头文件因此不需要 include ORT。
// ---------------------------------------------------------------------------

struct ArdyRuntime::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "ardy"};   ///< 进程级 ORT 环境
    Ort::SessionOptions session_options;             ///< 三个会话共用的选项
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::unique_ptr<Ort::Session> text_session;      ///< text_encoder.onnx
    std::unique_ptr<Ort::Session> denoise_session;   ///< denoiser.onnx
    std::unique_ptr<Ort::Session> decode_session;    ///< decoder.onnx
    Tokenizer tokenizer;                             ///< 内置 WordPiece 分词器

    // 下面这些名字都在构造时从 manifest 里查一次，之后每次推理直接用指针，
    // 避免在热路径里反复构造 std::string。
    std::vector<std::string> text_inputs;   ///< [input_ids, attention_mask, token_type_ids]
    std::string text_output;                ///< text_conditions

    std::vector<std::string> denoise_inputs;  ///< 固定顺序，与 Run() 的实参数组一一对应
    std::string denoise_output;               ///< pred_x0

    std::vector<std::string> decode_inputs;   ///< [hybrid_tokens, motion_pad_mask, global_translation]
    std::vector<std::string> decode_outputs;  ///< [normalized_motion, posed_joints, ...]
    std::vector<const char*> decode_output_ptrs;  ///< 给 Run() 用的 C 字符串数组

    std::vector<std::string> providers;
};

namespace {

/// 打开一个 ONNX 图。
std::unique_ptr<Ort::Session> make_session(Ort::Env& env, const Ort::SessionOptions& options,
                                           const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("model graph not found: " + path.string());
    }
    // Windows 上 ORTCHAR_T 是 wchar_t，直接把 filesystem::path 的本地编码路径交给它。
    return std::make_unique<Ort::Session>(env, path.c_str(), options);
}

/// 根据用户选择配置执行提供者（EP），并记录实际使用的设备。
///
/// 优先级：DirectML（零配置，覆盖 N 卡/A 卡/核显）> CUDA（需要额外运行时）> CPU。
/// 返回 true 表示成功挂上了 GPU。
///
/// 一个必须注意的坑：DirectML 的 device_id 是 DXGI 枚举顺序，而不是
/// “系统里第几块显卡”。笔记本上 0 号常常是核显，甚至是远程桌面的虚拟显示
/// 适配器 —— 绑上去会在第一次推理时报 DXGI_ERROR_DEVICE_REMOVED。
/// 所以这里默认用 best_dxgi_adapter() 自己挑独显（见 dxgi.hpp）。
bool configure_providers(const std::string& device, int adapter_id, Ort::SessionOptions& options,
                         std::vector<std::string>& out, std::string& label, bool verbose) {
    const auto available = Ort::GetAvailableProviders();
    out.clear();
    auto has = [&](const char* name) {
        return std::find(available.begin(), available.end(), std::string(name)) != available.end();
    };
    // DirectML 随包提供；CUDA 只有在用户替换了 third_party/onnxruntime 时才存在。
    if (device != "cpu" && (device == "dml" || device == "auto") && has("DmlExecutionProvider")) {
        const int adapter = adapter_id >= 0 ? adapter_id : best_dxgi_adapter();
        const std::string id = std::to_string(adapter >= 0 ? adapter : 0);
        try {
            options.AppendExecutionProvider("DmlExecutionProvider", {{"device_id", id}});
            out.emplace_back("DmlExecutionProvider");
            label = dxgi_adapter_name(adapter >= 0 ? adapter : 0);
            return true;
        } catch (const std::exception& error) {
            // 显式要求 dml 时报错；auto 模式下静默降级。
            if (device == "dml") throw;
            if (verbose) {
                std::fprintf(stderr, "  DirectML unavailable (%s); using the CPU.\n", error.what());
            }
        }
    }
    if (out.empty() && device != "cpu" && (device == "cuda" || device == "auto") && has("CUDAExecutionProvider")) {
        options.AppendExecutionProvider("CUDAExecutionProvider", {{"device_id", "0"}});
        out.emplace_back("CUDAExecutionProvider");
        label = "CUDA device 0";
        return true;
    }
    if (out.empty()) {
        if (device == "dml" || device == "cuda") {
            throw std::runtime_error("this ONNX Runtime build has no " + device + " provider");
        }
        out.emplace_back("CPUExecutionProvider");
        label = "CPU";
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// ArdyRuntime
// ---------------------------------------------------------------------------

ArdyRuntime::ArdyRuntime(const std::string& model_dir, const EngineOptions& options)
    : impl_(std::make_unique<Impl>()), model_dir_(model_dir), verbose_(options.verbose) {
    config_ = load_config((std::filesystem::path(model_dir) / "model.json").string());
    // CFG 权重：命令行没给就用 manifest 里的默认值（模型自带 2.0，演示页用 3.5）。
    cfg_weight_ = options.cfg_weight >= 0.0 ? options.cfg_weight : config_.default_cfg_weight;

    // 会话选项：开启全部图优化（会把 FP16 权重上的 Cast 折叠成常量，
    // 也会做算子融合）；顺序执行模式对单请求场景延迟更低。
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (options.intra_op_threads > 0) impl_->session_options.SetIntraOpNumThreads(options.intra_op_threads);
    configure_providers(options.device, options.adapter_id, impl_->session_options, impl_->providers,
                        device_label_, verbose_);
    providers_ = impl_->providers;

    const std::filesystem::path root(model_dir);
    auto open_graphs = [&](const Ort::SessionOptions& session_options) {
        impl_->text_session = make_session(impl_->env, session_options, root / config_.text_encoder.model);
        impl_->denoise_session = make_session(impl_->env, session_options, root / config_.denoiser.model);
        impl_->decode_session = make_session(impl_->env, session_options, root / config_.decoder.model);
    };
    try {
        open_graphs(impl_->session_options);
    } catch (const std::exception& error) {
        if (impl_->providers.empty() || impl_->providers.front() == "CPUExecutionProvider") throw;
        // GPU provider 放不下某张图时不能让整个程序挂掉：三条图一起退回 CPU 重开。
        // （典型场景：DML 不支持某个量化算子。）
        std::fprintf(stderr, "  %s failed (%s); falling back to the CPU.\n",
                     impl_->providers.front().c_str(), error.what());
        Ort::SessionOptions cpu_options;
        cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        cpu_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        if (options.intra_op_threads > 0) cpu_options.SetIntraOpNumThreads(options.intra_op_threads);
        open_graphs(cpu_options);
        impl_->providers = {"CPUExecutionProvider"};
        providers_ = impl_->providers;
        device_label_ = "CPU";
    }
    impl_->tokenizer = Tokenizer::from_file((root / "vocab.txt").string(), config_.max_prompt_tokens);

    // ---- 绑定张量名（语义名 → ONNX 里的实际名字）----
    impl_->text_inputs = {
        config_.input_name(config_.text_encoder, "inputIds"),
        config_.input_name(config_.text_encoder, "attentionMask"),
        config_.input_name(config_.text_encoder, "tokenTypeIds"),
    };
    impl_->text_output = config_.output_name(config_.text_encoder, "textConditions");

    // 去噪器的 11 个输入按固定顺序绑定，与下面 denoise_step 里的实参数组一一对应。
    const auto& d = config_.denoiser;
    impl_->denoise_inputs = {
        config_.input_name(d, "x"),
        config_.input_name(d, "historyLength"),
        config_.input_name(d, "generationLength"),
        config_.input_name(d, "historyMask"),
        config_.input_name(d, "generationMask"),
        config_.input_name(d, "historyTokenMask"),
        config_.input_name(d, "generationTokenMask"),
        config_.input_name(d, "textConditions"),
        config_.input_name(d, "firstHeadingAngle"),
        config_.input_name(d, "timestep"),
        config_.input_name(d, "cfgWeight"),
    };
    impl_->denoise_output = config_.output_name(d, "predX0");

    impl_->decode_inputs = {
        config_.input_name(config_.decoder, "hybridTokens"),
        config_.input_name(config_.decoder, "motionPadMask"),
        config_.input_name(config_.decoder, "globalTranslation"),
    };
    impl_->decode_outputs = {
        config_.output_name(config_.decoder, "normalizedMotion"),
        config_.output_name(config_.decoder, "posedJoints"),
        config_.output_name(config_.decoder, "localRotations"),
        config_.output_name(config_.decoder, "rootPositions"),
        config_.output_name(config_.decoder, "footContacts"),
    };
    impl_->decode_output_ptrs.clear();
    for (const auto& name : impl_->decode_outputs) impl_->decode_output_ptrs.push_back(name.c_str());
}

ArdyRuntime::~ArdyRuntime() = default;

/// 文本编码：分词 → 三个 int64 张量 → 一次前向 → 2048 维向量。
///
/// 输入形状是 [1, L]，L 随句子长度变化；这里每次调用都会真的推理一次，
/// 调用方（ArdySession）负责按 prompt 做缓存。
std::vector<float> ArdyRuntime::encode_text(const std::string& prompt) {
    const Tokenizer::Encoding encoding = impl_->tokenizer.encode(prompt);
    const int64_t length = static_cast<int64_t>(encoding.ids.size());

    std::vector<int64_t> ids = encoding.ids;
    std::vector<int64_t> mask = encoding.attention_mask;
    std::vector<int64_t> types = encoding.token_type_ids;
    const std::vector<int64_t> shape = shape2(1, length);

    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<int64_t>(impl_->memory, ids.data(), ids.size(), shape.data(), 2),
        Ort::Value::CreateTensor<int64_t>(impl_->memory, mask.data(), mask.size(), shape.data(), 2),
        Ort::Value::CreateTensor<int64_t>(impl_->memory, types.data(), types.size(), shape.data(), 2),
    };
    std::vector<const char*> input_names;
    for (const auto& name : impl_->text_inputs) input_names.push_back(name.c_str());
    const char* output_names[] = {impl_->text_output.c_str()};

    auto outputs = impl_->text_session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                                            inputs.size(), output_names, 1);
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + config_.dims.text_condition_dim);
}

/// 去噪网络单步前向。
///
/// 这里的 window 只是一块数据，函数不修改它（内部的 x 会拷贝一份传给 ORT，
/// 因为 ORT 需要一个可写的输入缓冲）。真正读-改-写的是 ddim.hpp。
void ArdyRuntime::denoise_step(const ArWindow& window, const std::vector<float>& text_conditions,
                               double first_heading_angle, int timestep,
                               std::vector<float>& prediction) {
    const Dimensions& dims = config_.dims;
    // ONNX Runtime 的输入张量要求非 const 指针，所以这里做一份拷贝。
    std::vector<float> x = window.x;
    std::vector<float> history_mask = window.history_mask;
    std::vector<float> generation_mask = window.generation_mask;
    std::vector<float> history_token_mask = window.history_token_mask;
    std::vector<float> generation_token_mask = window.generation_token_mask;
    std::vector<float> conditions = text_conditions;

    // 标量输入（ORT 要求张量，即便只有一个元素，也要给 shape=[1]）。
    int64_t history_length = window.history_frames;
    int64_t generation_length = window.generation_frames;
    int64_t timestep_value = timestep;
    float heading = static_cast<float>(first_heading_angle);
    float cfg = static_cast<float>(cfg_weight_);

    const std::vector<int64_t> x_shape = {1, dims.max_tokens, dims.hybrid_dim};
    const std::vector<int64_t> frame_shape = shape2(1, dims.max_frames);
    const std::vector<int64_t> token_shape = shape2(1, dims.max_tokens);
    const std::vector<int64_t> condition_shape = {1, 1, dims.text_condition_dim};
    const std::vector<int64_t> scalar_shape = shape1(1);

    // 顺序必须与构造函数里 denoise_inputs 的顺序完全一致。
    std::array<Ort::Value, 11> inputs{
        Ort::Value::CreateTensor<float>(impl_->memory, x.data(), x.size(), x_shape.data(), 3),
        Ort::Value::CreateTensor<int64_t>(impl_->memory, &history_length, 1, scalar_shape.data(), 1),
        Ort::Value::CreateTensor<int64_t>(impl_->memory, &generation_length, 1, scalar_shape.data(), 1),
        Ort::Value::CreateTensor<float>(impl_->memory, history_mask.data(), history_mask.size(),
                                       frame_shape.data(), 2),
        Ort::Value::CreateTensor<float>(impl_->memory, generation_mask.data(), generation_mask.size(),
                                       frame_shape.data(), 2),
        Ort::Value::CreateTensor<float>(impl_->memory, history_token_mask.data(), history_token_mask.size(),
                                       token_shape.data(), 2),
        Ort::Value::CreateTensor<float>(impl_->memory, generation_token_mask.data(),
                                       generation_token_mask.size(), token_shape.data(), 2),
        Ort::Value::CreateTensor<float>(impl_->memory, conditions.data(), conditions.size(),
                                       condition_shape.data(), 3),
        Ort::Value::CreateTensor<float>(impl_->memory, &heading, 1, scalar_shape.data(), 1),
        Ort::Value::CreateTensor<int64_t>(impl_->memory, &timestep_value, 1, scalar_shape.data(), 1),
        Ort::Value::CreateTensor<float>(impl_->memory, &cfg, 1, scalar_shape.data(), 1),
    };
    std::vector<const char*> input_names;
    for (const auto& name : impl_->denoise_inputs) input_names.push_back(name.c_str());
    const char* output_names[] = {impl_->denoise_output.c_str()};

    auto outputs = impl_->denoise_session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                                              inputs.size(), output_names, 1);
    const std::size_t count = static_cast<std::size_t>(dims.max_tokens) * dims.hybrid_dim;
    const float* data = outputs[0].GetTensorData<float>();
    prediction.assign(data, data + count);
}

/// 解码：混合 token → 动作特征 / 关节坐标 / 旋转 / 根位移 / 脚部接触。
///
/// 图里已经包含 FSQ 反量化、Transformer 身体解码和正向运动学，
/// 所以这里拿到的就是“可以直接渲染”的数据，不需要额外的骨架计算。
void ArdyRuntime::decode(const std::vector<float>& hybrid_tokens, int valid_tokens,
                         const std::array<float, 3>& global_translation, Decoded& out) {
    const Dimensions& dims = config_.dims;
    // 窗口固定 20 token，有效部分之外的槽位补 0（由 pad_mask 屏蔽）。
    std::vector<float> tokens = hybrid_tokens;
    tokens.resize(static_cast<std::size_t>(dims.max_tokens) * dims.hybrid_dim, 0.0f);
    std::vector<float> pad_mask = create_motion_pad_mask(dims, valid_tokens);
    float translation[3] = {global_translation[0], global_translation[1], global_translation[2]};

    const std::vector<int64_t> token_shape = {1, dims.max_tokens, dims.hybrid_dim};
    const std::vector<int64_t> frame_shape = shape2(1, dims.max_frames);
    const std::vector<int64_t> translation_shape = shape2(1, 3);

    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<float>(impl_->memory, tokens.data(), tokens.size(), token_shape.data(), 3),
        Ort::Value::CreateTensor<float>(impl_->memory, pad_mask.data(), pad_mask.size(), frame_shape.data(), 2),
        Ort::Value::CreateTensor<float>(impl_->memory, translation, 3, translation_shape.data(), 2),
    };
    std::vector<const char*> input_names;
    for (const auto& name : impl_->decode_inputs) input_names.push_back(name.c_str());

    auto outputs = impl_->decode_session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                                             inputs.size(), impl_->decode_output_ptrs.data(),
                                             impl_->decode_output_ptrs.size());

    const std::size_t frames = static_cast<std::size_t>(dims.max_frames);
    const std::size_t joints = static_cast<std::size_t>(dims.num_joints);

    auto take_floats = [&](std::size_t index, std::size_t count) {
        const float* data = outputs[index].GetTensorData<float>();
        return std::vector<float>(data, data + count);
    };

    // 顺序与 decode_outputs 一致。
    out.normalized_motion = take_floats(0, frames * static_cast<std::size_t>(dims.motion_dim));
    out.posed_joints = take_floats(1, frames * joints * 3);
    out.local_rotations = take_floats(2, frames * joints * 9);
    out.root_positions = take_floats(3, frames * 3);

    // 脚部接触在不同导出里可能是 bool 也可能是 float，这里两种都兼容。
    out.foot_contacts.assign(frames * 4, 0);
    const Ort::Value& contacts = outputs[4];
    const auto info = contacts.GetTensorTypeAndShapeInfo();
    const ONNXTensorElementDataType element_type = info.GetElementType();
    const std::size_t contact_count = frames * 4;
    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        const bool* data = contacts.GetTensorData<bool>();
        for (std::size_t i = 0; i < contact_count; ++i) out.foot_contacts[i] = data[i] ? 1 : 0;
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = contacts.GetTensorData<float>();
        for (std::size_t i = 0; i < contact_count; ++i) out.foot_contacts[i] = data[i] > 0.5f ? 1 : 0;
    } else {
        throw std::runtime_error("unexpected foot contact tensor dtype");
    }
}

// ---------------------------------------------------------------------------
// ArdySession
// ---------------------------------------------------------------------------

ArdySession::ArdySession(ArdyRuntime& runtime, std::uint32_t seed,
                         const std::array<float, 3>& initial_translation, double initial_heading)
    : runtime_(runtime),
      config_(runtime.config()),
      rng_(seed),
      initial_translation_(initial_translation),
      initial_heading_(initial_heading) {
    // 历史缓冲最多装 history_tokens 个 token，提前预留避免反复扩容。
    global_hybrid_.reserve(static_cast<std::size_t>(config_.dims.history_tokens) * config_.dims.hybrid_dim);
}

/// 把会话截到某个帧号。
///
/// 只在“会话帧数不是 4 的倍数”时用到：半截的 token 无法作为历史参与自回归，
/// 所以向下对齐到完整 token。
void ArdySession::branch_to(int frame) {
    const Dimensions& dims = config_.dims;
    const int tokens = frame / dims.num_frames_per_token;
    const std::size_t keep = std::min<std::size_t>(global_hybrid_.size(),
                                                   static_cast<std::size_t>(tokens) * dims.hybrid_dim);
    global_hybrid_.resize(keep);
    frame_count_ = tokens * dims.num_frames_per_token;
}

/// 追加历史 token，并把过老的裁掉。
///
/// 只有最近 history_tokens 个 token 会被 prepare_history 读到，
/// 所以缓冲区维持在这个长度即可，内存占用是常数。
void ArdySession::append_global_hybrid(const std::vector<float>& tokens) {
    global_hybrid_.insert(global_hybrid_.end(), tokens.begin(), tokens.end());
    const std::size_t limit = static_cast<std::size_t>(config_.dims.history_tokens) * config_.dims.hybrid_dim;
    if (global_hybrid_.size() > limit) {
        global_hybrid_.erase(global_hybrid_.begin(),
                             global_hybrid_.begin() + static_cast<std::ptrdiff_t>(global_hybrid_.size() - limit));
    }
}

/// 准备历史窗口：裁剪 + root 重定心 + 计算下一段的起始朝向。
///
/// 两种分支：
///   * requested_tokens == 0（第一轮，或没有可用历史）：
///       历史为空，世界位移直接沿用调用方给的初值；
///       但如果之前生成过（available_tokens > 0），仍要用最后一帧的 x/z 作为
///       起点，否则第二段会从原点重新开始，轨迹会断。
///   * 正常情况：取最后 requested_tokens 个 token，把整段 root 平移到
///       以“历史最后一帧”为原点，并记录被减掉的偏移量。
///
/// 注意 y 轴单独处理：模型只关心水平移动，高度是绝对量；这里用调用方给的
/// initial_translation[1] 作为高度基准，避免角色突然沉到地下或飘在空中。
ArdySession::PreparedHistory ArdySession::prepare_history(int requested_frames) const {
    const Dimensions& dims = config_.dims;
    const RecenterConfig& recenter = config_.recenter;
    const int frames_per_token = dims.num_frames_per_token;
    const int hybrid_dim = dims.hybrid_dim;

    const int available_tokens =
        std::min(frame_count_ / frames_per_token,
                 static_cast<int>(global_hybrid_.size() / static_cast<std::size_t>(hybrid_dim)));
    const int requested_tokens = std::min({dims.history_tokens, requested_frames / frames_per_token,
                                           available_tokens});

    const int position_x = recenter.position_indices[0];
    const int position_y = recenter.position_indices[1];
    const int position_z = recenter.position_indices[2];
    const int heading_cos = recenter.heading_indices[0];
    const int heading_sin = recenter.heading_indices[1];

    // 读某个 root 特征的“反归一化真值”。
    auto raw_root = [&](const std::vector<float>& buffer, int base_token, int frame, int feature) -> double {
        const int token = base_token + frame / frames_per_token;
        const std::size_t index = static_cast<std::size_t>(token) * hybrid_dim +
                                  static_cast<std::size_t>(frame % frames_per_token) *
                                      dims.root_features_per_frame +
                                  static_cast<std::size_t>(feature);
        return static_cast<double>(buffer[index]) * recenter.root_std[static_cast<std::size_t>(feature)] +
               recenter.root_mean[static_cast<std::size_t>(feature)];
    };

    PreparedHistory prepared;
    if (requested_tokens == 0) {
        if (available_tokens > 0) {
            // 没有可用的历史 token，但世界坐标不能丢：用最后一帧继续。
            const int last_frame = available_tokens * frames_per_token - 1;
            prepared.global_translation = {
                static_cast<float>(raw_root(global_hybrid_, 0, last_frame, position_x)),
                initial_translation_[1],
                static_cast<float>(raw_root(global_hybrid_, 0, last_frame, position_z)),
            };
            prepared.first_heading_angle =
                std::atan2(raw_root(global_hybrid_, 0, last_frame, heading_sin),
                           raw_root(global_hybrid_, 0, last_frame, heading_cos));
        } else {
            prepared.global_translation = initial_translation_;
            prepared.first_heading_angle = initial_heading_;
        }
        return prepared;
    }

    // 取最近 requested_tokens 个 token 作为历史。
    const int first_token = available_tokens - requested_tokens;
    const std::size_t offset = static_cast<std::size_t>(first_token) * hybrid_dim;
    prepared.history.assign(global_hybrid_.begin() + static_cast<std::ptrdiff_t>(offset),
                            global_hybrid_.begin() +
                                static_cast<std::ptrdiff_t>(offset + static_cast<std::size_t>(requested_tokens) *
                                                                         hybrid_dim));
    prepared.history_tokens = requested_tokens;
    prepared.history_frames = requested_tokens * frames_per_token;

    // 以历史最后一帧为原点，把整段 root 平移回原点附近。
    const int last_frame = prepared.history_frames - 1;
    const double center_x = raw_root(prepared.history, 0, last_frame, position_x);
    const double center_z = raw_root(prepared.history, 0, last_frame, position_z);
    const double first_cos = raw_root(prepared.history, 0, 0, heading_cos);
    const double first_sin = raw_root(prepared.history, 0, 0, heading_sin);

    for (int frame = 0; frame < prepared.history_frames; ++frame) {
        // y 用 initial_translation[1] 作为基准（去掉外部施加的垂直偏移）。
        const std::pair<int, double> adjustments[3] = {
            {position_x, center_x},
            {position_y, static_cast<double>(initial_translation_[1])},
            {position_z, center_z},
        };
        for (const auto& [feature, center] : adjustments) {
            const std::size_t token_index =
                static_cast<std::size_t>(frame / frames_per_token) * hybrid_dim +
                static_cast<std::size_t>(frame % frames_per_token) * dims.root_features_per_frame +
                static_cast<std::size_t>(feature);
            const double translated = static_cast<double>(prepared.history[token_index]) *
                                          recenter.root_std[static_cast<std::size_t>(feature)] +
                                      recenter.root_mean[static_cast<std::size_t>(feature)] - center;
            prepared.history[token_index] = static_cast<float>(
                (translated - recenter.root_mean[static_cast<std::size_t>(feature)]) /
                recenter.root_std[static_cast<std::size_t>(feature)]);
        }
    }

    prepared.global_translation = {static_cast<float>(center_x), initial_translation_[1],
                                   static_cast<float>(center_z)};
    prepared.first_heading_angle = std::atan2(first_sin, first_cos);
    return prepared;
}

/// 生成入口：按 40 帧一个窗口循环，直到凑够请求的帧数。
///
/// 一轮窗口内的完整流程：
///   1. prepare_history   取出最近 40 帧，root 重定心
///   2. create_ar_window  历史 + 新噪声拼成 20 token
///   3. 10 次 denoise_step + apply_ddim_step（逐步去噪）
///   4. recenter_and_requantize  把结果整理成下一轮可用的历史
///   5. decode            解出关节/旋转/根位移
///   6. 只取后 40 帧交付，并把 token 存进历史
///
/// 第 6 步“只取后 40 帧”是自回归的常规做法：前 40 帧是给网络的对齐上下文，
/// 已经在上一轮交付过了，重复交付会造成动作重叠。
std::vector<MotionChunk> ArdySession::step(const std::string& prompt, int duration_frames) {
    const Dimensions& dims = config_.dims;
    if (duration_frames <= 0) throw std::runtime_error("duration must be positive");
    if (duration_frames > config_.max_output_frames) {
        throw std::runtime_error("duration exceeds the model's window limit");
    }

    // 半截 token 无法参与自回归，先对齐。
    if (frame_count_ % dims.num_frames_per_token != 0) branch_to(frame_count_);

    // 提示词没变就复用上次的条件向量 —— 这省掉了每次窗口都跑一遍 MiniLM。
    // 换词的瞬间这里会重新编码，新指令从下一个窗口开始生效。
    if (cached_prompt_ != prompt) {
        cached_conditions_ = runtime_.encode_text(prompt);
        cached_prompt_ = prompt;
    }

    const int window_count =
        (duration_frames + dims.generation_frames - 1) / dims.generation_frames;
    std::vector<MotionChunk> chunks;
    chunks.reserve(static_cast<std::size_t>(window_count));

    int written = 0;
    for (int window_index = 0; window_index < window_count; ++window_index) {
        // ---- 1 & 2：历史 + 噪声 ----
        const PreparedHistory prepared = prepare_history(dims.history_frames);
        ArWindow window = create_ar_window(dims, rng_, prepared.history);

        std::vector<float> prediction(static_cast<std::size_t>(dims.max_tokens) * dims.hybrid_dim);
        // 只有“生成槽”会被 DDIM 改写，历史槽必须保持不变。
        const std::size_t generation_start =
            static_cast<std::size_t>(window.generation_token_offset) * dims.hybrid_dim;
        const std::size_t generation_end =
            static_cast<std::size_t>(window.generation_token_offset + window.generation_tokens) *
            dims.hybrid_dim;

        // ---- 3：10 步 DDIM ----
        const double denoise_started = now_ms();
        for (std::size_t index = 0; index < config_.schedule.steps(); ++index) {
            const DdimStep step = ddim_step(config_.schedule, index);
            runtime_.denoise_step(window, cached_conditions_, prepared.first_heading_angle,
                                  step.timestep, prediction);
            apply_ddim_step(window.x, prediction, step, generation_start, generation_end);
        }
        const double denoise_ms = now_ms() - denoise_started;

        // ---- 4：整理成下一轮可用的历史 ----
        const int valid_tokens = window.history_tokens + window.generation_tokens;
        std::vector<float> decoder_hybrid(static_cast<std::size_t>(dims.max_tokens) * dims.hybrid_dim, 0.0f);
        std::copy_n(window.x.begin(),
                    static_cast<std::size_t>(valid_tokens) * dims.hybrid_dim,
                    decoder_hybrid.begin());
        const int next_history_start = std::max(0, valid_tokens - dims.history_tokens);
        const RecenterResult recentered =
            recenter_and_requantize(decoder_hybrid, valid_tokens, dims, config_.recenter,
                                    config_.quantization, prepared.global_translation, next_history_start);

        // ---- 5：解码成可渲染数据 ----
        ArdyRuntime::Decoded decoded;
        const double decode_started = now_ms();
        runtime_.decode(decoder_hybrid, valid_tokens, recentered.global_translation, decoded);
        const double decode_ms = now_ms() - decode_started;

        const int frames_to_copy = std::min(dims.generation_frames, duration_frames - written);
        const int generated_frame_offset = window.history_frames;
        const int generated_tokens = (frames_to_copy + dims.num_frames_per_token - 1) / dims.num_frames_per_token;

        // ---- 6a：把这一轮生成的内容写回历史 ----
        // root 部分取“解码后的归一化动作”（它已经带上了重定心后的正确坐标），
        // 身体潜变量取窗口里的值 —— 两者拼起来才是完整的一个 token。
        std::vector<float> tokens(static_cast<std::size_t>(generated_tokens) * dims.hybrid_dim, 0.0f);
        for (int token = 0; token < generated_tokens; ++token) {
            const std::size_t destination = static_cast<std::size_t>(token) * dims.hybrid_dim;
            for (int frame_in_token = 0; frame_in_token < dims.num_frames_per_token; ++frame_in_token) {
                const int source_frame =
                    generated_frame_offset + token * dims.num_frames_per_token + frame_in_token;
                const std::size_t destination_root =
                    destination + static_cast<std::size_t>(frame_in_token) * dims.root_features_per_frame;
                const std::size_t source_motion =
                    static_cast<std::size_t>(source_frame) * dims.motion_dim;
                for (int feature = 0; feature < dims.root_features_per_frame; ++feature) {
                    tokens[destination_root + static_cast<std::size_t>(feature)] =
                        decoded.normalized_motion[source_motion + static_cast<std::size_t>(feature)];
                }
            }
            const std::size_t source_token =
                static_cast<std::size_t>(window.generation_token_offset + token) * dims.hybrid_dim;
            for (int feature = dims.nframe_root_dim; feature < dims.hybrid_dim; ++feature) {
                tokens[destination + static_cast<std::size_t>(feature)] =
                    decoder_hybrid[source_token + static_cast<std::size_t>(feature)];
            }
        }
        append_global_hybrid(tokens);

        // ---- 6b：切出要交付的那 40 帧 ----
        MotionChunk chunk;
        chunk.start_frame = frame_count_;
        chunk.frame_count = frames_to_copy;
        chunk.denoise_ms = denoise_ms;
        chunk.decode_ms = decode_ms;

        const std::size_t frame_offset = static_cast<std::size_t>(generated_frame_offset);
        const std::size_t frames = static_cast<std::size_t>(frames_to_copy);
        const std::size_t joints = static_cast<std::size_t>(dims.num_joints);

        chunk.motion.assign(decoded.normalized_motion.begin() +
                                static_cast<std::ptrdiff_t>(frame_offset * dims.motion_dim),
                            decoded.normalized_motion.begin() +
                                static_cast<std::ptrdiff_t>((frame_offset + frames) * dims.motion_dim));
        chunk.posed_joints.assign(decoded.posed_joints.begin() +
                                      static_cast<std::ptrdiff_t>(frame_offset * joints * 3),
                                  decoded.posed_joints.begin() +
                                      static_cast<std::ptrdiff_t>((frame_offset + frames) * joints * 3));
        chunk.local_rotations.assign(decoded.local_rotations.begin() +
                                         static_cast<std::ptrdiff_t>(frame_offset * joints * 9),
                                     decoded.local_rotations.begin() +
                                         static_cast<std::ptrdiff_t>((frame_offset + frames) * joints * 9));
        chunk.root_positions.assign(decoded.root_positions.begin() +
                                        static_cast<std::ptrdiff_t>(frame_offset * 3),
                                    decoded.root_positions.begin() +
                                        static_cast<std::ptrdiff_t>((frame_offset + frames) * 3));
        chunk.foot_contacts.assign(decoded.foot_contacts.begin() +
                                       static_cast<std::ptrdiff_t>(frame_offset * 4),
                                   decoded.foot_contacts.begin() +
                                       static_cast<std::ptrdiff_t>((frame_offset + frames) * 4));

        frame_count_ += frames_to_copy;
        written += frames_to_copy;
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

}  // namespace ardy
