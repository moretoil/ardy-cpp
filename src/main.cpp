// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
//  ardy-cli：命令行文本生成动作
// =============================================================================
//
// 三种用法：
//
//   单句生成
//     ardy-cli "A person walks forward." --seconds 6 --out walk.bvh
//
//   脚本模式（每行一句，每句生成一个 2 秒窗口，动作连续衔接）
//     ardy-cli --script prompts.txt --out sequence.bvh
//
//   交互模式（从标准输入逐句读，适合演示“边播边换词”）
//     ardy-cli --interactive --out session.bvh
//
// 同一个 --seed 一定得到同一段动作；中途换提示词时角色不会重置，
// 而是从当前姿势继续 —— 这就是“交互式”的全部意思。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ardy_engine.hpp"
#include "bvh.hpp"
#include "tokenizer.hpp"

namespace {

/// 解析后的命令行参数。
struct Arguments {
    std::string prompt;                     ///< 位置参数拼成的提示词
    std::string script;                     ///< --script 文件路径
    std::string output = "output.bvh";      ///< --out
    std::string model_dir = "models/int8";  ///< --model-dir（相对 exe 或当前目录）
    std::string device = "cpu";             ///< --device：cpu / dml / auto
    int adapter_id = -1;                    ///< --device-id：DirectML 设备号
    std::string dump;                       ///< --dump：导出关节原始数据供比对
    std::string tokenize;                   ///< --tokenize：只打印分词结果
    int seed = 0;                           ///< --seed
    double seconds = 4.0;                   ///< --seconds
    // 官方网页演示用 3.5（模型自带的默认值 2.0 偏弱，跟提示词不够紧）。
    double cfg_weight = 3.5;
    int threads = 0;                        ///< --threads
    bool interactive = false;               ///< --interactive
    bool verbose = false;                   ///< --verbose：逐窗口耗时
    bool quiet = false;                     ///< --quiet：只输出错误
    std::string debug_dir;                  ///< 预留
};

/// 打印帮助。
void usage() {
    std::cout <<
        "usage: ardy-cli [prompt] [options]\n"
        "\n"
        "  --seconds N        seconds to generate (default 4.0, multiples of 2 recommended)\n"
        "  --seed N|STRING    deterministic seed (default 0)\n"
        "  --cfg-weight W     text guidance weight (default 3.5, matching the web demo)\n"
        "  --out PATH         BVH output path (default output.bvh)\n"
        "  --script PATH      text file, one prompt per generation window\n"
        "  --interactive      read prompts from stdin, one window at a time\n"
        "  --model-dir DIR    model assets (default models/int8)\n"
        "  --device DEV       cpu | dml | auto (default cpu)\n"
        "  --device-id N      DXGI adapter index for DirectML (default: best GPU)\n"
        "  --threads N        ONNX Runtime intra-op threads\n"
        "  --dump PATH        write raw joint positions for numeric verification\n"
        "  --tokenize TEXT    print the token ids for TEXT and exit\n"
        "  --verbose          per-window timings\n";
}

/// 手写参数解析。
///
/// 之所以不用 getopt：Windows/MinGW 下的 getopt 行为与 Linux 不同（不支持
/// 长选项），而这里的选项表很小，手写反而更可控：报错信息统一、支持
/// “位置参数拼成提示词”（方便直接写 ardy-cli A person walks forward.）。
bool parse(int argc, char** argv, Arguments& args, std::string& error) {
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        // 读取下一个参数作为选项值。
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        try {
            if (arg == "-h" || arg == "--help") {
                usage();
                std::exit(0);
            } else if (arg == "--seconds") {
                args.seconds = std::stod(next("--seconds"));
            } else if (arg == "--seed") {
                const std::string value = next("--seed");
                args.seed = std::atoi(value.c_str());
            } else if (arg == "--cfg-weight") {
                args.cfg_weight = std::stod(next("--cfg-weight"));
            } else if (arg == "--out") {
                args.output = next("--out");
            } else if (arg == "--script") {
                args.script = next("--script");
            } else if (arg == "--model-dir") {
                args.model_dir = next("--model-dir");
            } else if (arg == "--device") {
                args.device = next("--device");
            } else if (arg == "--device-id") {
                args.adapter_id = std::stoi(next("--device-id"));
            } else if (arg == "--threads") {
                args.threads = std::stoi(next("--threads"));
            } else if (arg == "--dump") {
                args.dump = next("--dump");
            } else if (arg == "--tokenize") {
                args.tokenize = next("--tokenize");
            } else if (arg == "--interactive") {
                args.interactive = true;
            } else if (arg == "--verbose") {
                args.verbose = true;
            } else if (arg == "--quiet") {
                args.quiet = true;
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("unknown option " + arg);
            } else {
                positional.push_back(arg);
            }
        } catch (const std::exception& exception) {
            error = exception.what();
            return false;
        }
    }
    // 例：ardy-cli A person walks forward → 拼回一句完整提示词。
    for (const auto& word : positional) {
        if (!args.prompt.empty()) args.prompt += " ";
        args.prompt += word;
    }
    return true;
}

/// 把若干个分段顺序拼进三个大数组（BVH 需要整段连续数据）。
void append_chunks(const std::vector<ardy::MotionChunk>& chunks, std::vector<float>& joints,
                   std::vector<float>& rotations, std::vector<float>& root) {
    for (const ardy::MotionChunk& chunk : chunks) {
        joints.insert(joints.end(), chunk.posed_joints.begin(), chunk.posed_joints.end());
        rotations.insert(rotations.end(), chunk.local_rotations.begin(), chunk.local_rotations.end());
        root.insert(root.end(), chunk.root_positions.begin(), chunk.root_positions.end());
    }
}

/// 写一份二进制关节数据，供 tools/compare_dump.py 与 Python 参考实现逐帧比对。
///
/// 格式（小端）：
///     char[8]  "ARDYJNT1"
///     int32    frames
///     int32    joints
///     float32  frames × joints × 3
///
/// 用二进制而不是文本，是为了避免十进制舍入掩盖真正的数值差异 ——
/// 我们的目标是验证到 1e-6 量级。
bool write_dump(const std::string& path, const std::vector<float>& joints, int frames, int joint_count) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;
    const char magic[8] = {'A', 'R', 'D', 'Y', 'J', 'N', 'T', '1'};
    std::fwrite(magic, 1, sizeof(magic), file);
    const std::int32_t header[2] = {frames, joint_count};
    std::fwrite(header, sizeof(std::int32_t), 2, file);
    std::fwrite(joints.data(), sizeof(float), joints.size(), file);
    std::fclose(file);
    return true;
}

/// --verbose 下打印单个窗口的耗时与根节点位移。
void print_window(const ardy::MotionChunk& chunk, const ardy::Config& config, bool verbose) {
    if (!verbose) return;
    const int joints = config.dims.num_joints;
    const float* first = &chunk.posed_joints[0];
    const float* last = &chunk.posed_joints[static_cast<std::size_t>(chunk.frame_count - 1) * joints * 3];
    std::printf("  frames %4d..%4d  %6.0f ms denoise  %6.0f ms decode  root z %.2f -> %.2f\n",
                chunk.start_frame, chunk.start_frame + chunk.frame_count - 1, chunk.denoise_ms,
                chunk.decode_ms, first[2], last[2]);
}

}  // namespace

int main(int argc, char** argv) {
    Arguments args;
    std::string error;
    if (!parse(argc, argv, args, error)) {
        std::cerr << "ardy-cli: " << error << "\n\n";
        usage();
        return 2;
    }
    if (!args.interactive && args.script.empty() && args.prompt.empty() && args.tokenize.empty()) {
        usage();
        return 2;
    }

    // 允许从任意目录启动：如果相对路径找不到模型，就试着用“exe 上一级目录
    // 下的同名路径”。这样双击/快捷方式运行也能正确找到 models/int8。
    {
        namespace fs = std::filesystem;
        if (!fs::exists(args.model_dir)) {
            std::error_code error_code;
            const fs::path executable = fs::absolute(argv[0], error_code);
            if (!error_code) {
                const fs::path candidate = executable.parent_path().parent_path() / args.model_dir;
                if (fs::exists(candidate)) args.model_dir = candidate.string();
            }
        }
    }

    // 分词一致性模式：只打印 token id，不加载任何模型。
    // tools/verify_tokenizer.py 用它和 HuggingFace tokenizers 做对比。
    if (!args.tokenize.empty()) {
        ardy::Tokenizer tokenizer = ardy::Tokenizer::from_file(args.model_dir + "/vocab.txt", 128);
        const ardy::Tokenizer::Encoding encoding = tokenizer.encode(args.tokenize);
        for (std::size_t i = 0; i < encoding.ids.size(); ++i) {
            std::printf("%s%lld", i == 0 ? "" : " ", static_cast<long long>(encoding.ids[i]));
        }
        std::printf("\n");
        return 0;
    }

    try {
        ardy::EngineOptions options;
        options.device = args.device;
        options.adapter_id = args.adapter_id;
        options.intra_op_threads = args.threads;
        options.cfg_weight = args.cfg_weight;
        options.verbose = args.verbose;

        // 加载三个 ONNX 图（这一步是冷启动的主要开销）。
        const auto load_started = std::chrono::steady_clock::now();
        ardy::ArdyRuntime runtime(args.model_dir, options);
        const double load_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - load_started)
                                   .count();

        const ardy::Config& config = runtime.config();
        const int fps = config.dims.fps;
        if (!args.quiet) {
            std::printf("ARDY Mini  |  %s\n", config.model_id.c_str());
            std::printf("device     |  %s  (%s)\n", runtime.providers()[0].c_str(),
                        runtime.device_label().c_str());
            // 顺带统计一下模型文件总大小，方便确认加载的是哪套资产（int8/fp16/fp32）。
            std::printf("loaded in %.0f ms  |  %d fps  |  %.1f MB of graphs\n", load_ms, fps,
                        [&] {
                            double total = 0.0;
                            for (const auto* name : {"text_encoder.onnx", "denoiser.onnx", "decoder.onnx"}) {
                                FILE* file = std::fopen((args.model_dir + "/" + name).c_str(), "rb");
                                if (file) {
                                    std::fseek(file, 0, SEEK_END);
                                    total += static_cast<double>(std::ftell(file));
                                    std::fclose(file);
                                }
                            }
                            return total / 1e6;
                        }());
        }

        ardy::ArdySession session(runtime, static_cast<std::uint32_t>(args.seed));
        std::vector<float> joints;
        std::vector<float> rotations;
        std::vector<float> root;
        const int window_frames = config.dims.generation_frames;

        // 生成一段并把结果追加进大数组。
        auto run_window = [&](const std::string& prompt, int frames) {
            const auto started = std::chrono::steady_clock::now();
            std::vector<ardy::MotionChunk> chunks = session.step(prompt, frames);
            const double elapsed = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
            append_chunks(chunks, joints, rotations, root);
            if (!args.quiet) {
                std::printf("  \"%s\"  ->  %d frames in %.0f ms\n", prompt.c_str(), frames, elapsed);
            }
            for (const auto& chunk : chunks) print_window(chunk, config, args.verbose);
        };

        if (args.interactive) {
            // 交互模式：每次输入生成 --seconds 秒，动作接着上一句继续。
            const int frames = static_cast<int>(std::lround(args.seconds * fps));
            std::string line;
            while (true) {
                std::cout << "prompt> " << std::flush;
                if (!std::getline(std::cin, line)) break;
                const auto start = line.find_first_not_of(" \t");
                const auto end = line.find_last_not_of(" \t");
                if (start == std::string::npos) continue;   // 空行
                line = line.substr(start, end - start + 1);
                if (line == "quit" || line == "exit") break;
                if (line.empty()) continue;
                run_window(line, frames);
            }
        } else if (!args.script.empty()) {
            // 脚本模式：一行一个窗口（每行 = 2 秒），窗口之间自然衔接。
            std::ifstream script(args.script);
            if (!script) throw std::runtime_error("cannot open script " + args.script);
            std::string line;
            while (std::getline(script, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                run_window(line, window_frames);
            }
        } else {
            // 单句模式。
            const int frames = static_cast<int>(std::lround(args.seconds * fps));
            run_window(args.prompt, frames);
        }

        if (joints.empty()) {
            std::cerr << "ardy-cli: nothing generated\n";
            return 1;
        }

        const int frames = static_cast<int>(joints.size() /
                                            (static_cast<std::size_t>(config.dims.num_joints) * 3));

        // 导出 BVH。
        ardy::BvhWriter writer(config);
        if (!writer.write(args.output, rotations, root, fps)) {
            throw std::runtime_error("cannot write " + args.output);
        }

        // 可选的二进制导出（数值校验用）。
        if (!args.dump.empty()) {
            if (!write_dump(args.dump, joints, frames, config.dims.num_joints)) {
                throw std::runtime_error("cannot write " + args.dump);
            }
        }

        // 人眼可读的统计摘要：前进距离 + 髋部高度范围。
        if (!args.quiet) {
            const std::size_t joints_per_frame = static_cast<std::size_t>(config.dims.num_joints) * 3;
            const float* first = &joints[0];
            const float* last = &joints[static_cast<std::size_t>(frames - 1) * joints_per_frame];
            double height_min = 1e9, height_max = -1e9;
            for (int frame = 0; frame < frames; ++frame) {
                const float y = joints[static_cast<std::size_t>(frame) * joints_per_frame + 1];
                height_min = std::min(height_min, static_cast<double>(y));
                height_max = std::max(height_max, static_cast<double>(y));
            }
            std::printf("generated %d frames (%.1f s)\n", frames, static_cast<double>(frames) / fps);
            std::printf("  forward travel  %.2f m\n", last[2] - first[2]);
            std::printf("  hip height      %.2f .. %.2f m\n", height_min, height_max);
            std::printf("  saved           %s\n", args.output.c_str());
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "ardy-cli: " << exception.what() << "\n";
        return 1;
    }
}
