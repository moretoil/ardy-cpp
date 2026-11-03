// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  为 DirectML 挑选正确的 DXGI 显卡
// =============================================================================
//
// 为什么需要这个文件？
//
//   ONNX Runtime 的 DirectML 执行提供者用 device_id 指定显卡，而这个 id 是
//   “DXGI 枚举顺序”，不是常见的“系统里第几块显卡”。笔记本上这个顺序非常反
//   直觉，实测一台机器是这样的：
//
//       0 → Intel Iris Xe（核显）
//       1 → NVIDIA RTX 3050（独显）   ← 想要的
//       2/3 → Intel Iris Xe（重复枚举）
//       4 → Microsoft Basic Render Driver（纯软件光栅器）
//
//   更糟的是，远程桌面/投屏类软件会安装“虚拟显示适配器”，它们能被枚举出来，
//   但根本没有可用的 3D 计算能力。一旦把 DML 绑到这类适配器上，创建会话时
//   看起来正常，第一次推理就会抛：
//
//       DXGI_ERROR_DEVICE_REMOVED (887A0005)
//       GPU 设备实例已经被暂停
//
//   所以不能无脑用 device_id=0，必须自己枚举一遍、挑出真正能干活的独显。
//
// 选取策略：
//   1. 跳过软件适配器（DXGI_ADAPTER_FLAG_SOFTWARE）
//   2. 厂商优先级：NVIDIA/AMD > 其它 > Intel 核显
//   3. 同厂商内比较专用显存（DedicatedVideoMemory），大的胜出

#if defined(_WIN32)

#include <dxgi.h>
#include <windows.h>

namespace ardy {

/// 一块显卡的摘要信息。
struct DxgiAdapter {
    int index = -1;                 ///< DXGI 枚举下标（就是 DML 的 device_id）
    unsigned int vendor = 0;        ///< 厂商 PCI ID（0x10DE=NVIDIA, 0x1002=AMD, 0x8086=Intel）
    unsigned long long vram = 0;    ///< 专用显存字节数
    bool software = false;          ///< 是否为软件光栅器（WARP）
    char name[128] = {};            ///< 显卡名字（UTF-8）
};

/// IID_IDXGIFactory1 = {770AAE78-F26F-4DBA-A829-253C83D1B387}
///
/// 手写 GUID 而不用 __uuidof：MinGW 下该运算符对部分 COM 接口不可靠。
static const IID kIidDxgiFactory1 = {0x770AAE78, 0xF26F, 0x4DBA,
                                     {0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87}};

/// 按 DXGI 顺序枚举所有显卡。
///
/// 任何一步失败都返回空列表（调用方会退回 device 0），不抛异常 ——
/// 在没有显卡的机器上我们也希望程序能继续用 CPU 跑。
inline std::vector<DxgiAdapter> enumerate_dxgi_adapters() {
    std::vector<DxgiAdapter> adapters;
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(kIidDxgiFactory1, reinterpret_cast<void**>(&factory))) || factory == nullptr) {
        return adapters;
    }
    for (UINT index = 0;; ++index) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(index, &adapter) != S_OK || adapter == nullptr) break;
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description))) {
            DxgiAdapter entry;
            entry.index = static_cast<int>(index);
            entry.vendor = description.VendorId;
            entry.vram = description.DedicatedVideoMemory;
            entry.software = (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
            // 描述是宽字符，转 UTF-8 方便直接打印到控制台。
            WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, entry.name, sizeof(entry.name), nullptr,
                                nullptr);
            adapters.push_back(entry);
        }
        adapter->Release();   // COM 对象必须手动释放
    }
    factory->Release();
    return adapters;
}

/// 厂商打分：独显厂商优先，Intel（多为核显）最后。
inline int dxgi_vendor_rank(unsigned int vendor) {
    switch (vendor) {
        case 0x10DE:  // NVIDIA
        case 0x1002:  // AMD
            return 3;
        case 0x8086:  // Intel
            return 1;
        default:
            return 2;   // 未知厂商（含虚拟适配器）
    }
}

/// 返回最适合跑推理的适配器下标；没有可用显卡时返回 -1。
inline int best_dxgi_adapter() {
    int best = -1;
    int best_rank = -1;
    unsigned long long best_vram = 0;
    for (const DxgiAdapter& adapter : enumerate_dxgi_adapters()) {
        if (adapter.software) continue;   // 软件渲染没有意义
        const int rank = dxgi_vendor_rank(adapter.vendor);
        if (rank > best_rank || (rank == best_rank && adapter.vram > best_vram)) {
            best = adapter.index;
            best_rank = rank;
            best_vram = adapter.vram;
        }
    }
    return best;
}

/// 把下标转成人能看懂的名字（启动横幅里显示）。
inline std::string dxgi_adapter_name(int index) {
    for (const DxgiAdapter& adapter : enumerate_dxgi_adapters()) {
        if (adapter.index == index) {
            char buffer[192];
            std::snprintf(buffer, sizeof(buffer), "%s (%.1f GB)", adapter.name,
                          static_cast<double>(adapter.vram) / (1024.0 * 1024.0 * 1024.0));
            return buffer;
        }
    }
    return "default adapter";
}

}  // namespace ardy

#else

// 非 Windows 平台：没有 DXGI，调用方会退回 CPU（或 CUDA）。
namespace ardy {
inline int best_dxgi_adapter() { return -1; }
inline std::string dxgi_adapter_name(int) { return "default adapter"; }
}  // namespace ardy

#endif
