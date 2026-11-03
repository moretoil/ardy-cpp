// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  极简 JSON 解析器（零第三方依赖）
// =============================================================================
//
// 为什么自己写？
//   引擎运行时只需要读取一个 model.json（约 25 KB），里面是尺寸、扩散调度表、
//   归一化统计量、骨骼定义等元数据。为了引入 nlohmann/json（~900 KB 单头文件）
//   或 RapidJSON 而增加编译负担并不划算，所以这里手写一个只覆盖 JSON 规范子集
//   的递归下降解析器：
//     - 支持：对象、数组、字符串（含 \u 转义与代理对）、数字、布尔、null
//     - 不支持：注释（JSON 标准本身也不允许）、BOM
//
// 用法：
//     json::Value root = json::parse(file_contents);
//     int fps = root["dimensions"]["fps"].as_int();
//     auto  arr = root["diffusion"]["timesteps"].to_ints();
//
// 设计取舍：
//   * 用 std::map 存对象成员 —— 解析结果按键有序，方便调试；成员数量很少，
//     查找性能无所谓。
//   * 数字统一按 double 存储 —— JSON 不区分整数/浮点，取值时再按需转换。
//   * 越界/类型错误一律抛 ParseError，避免解析出的脏数据静默传递到推理里。

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ardy {
namespace json {

/// 解析失败时抛出的异常，消息统一带 "json: " 前缀便于定位。
class ParseError : public std::runtime_error {
 public:
    explicit ParseError(const std::string& message) : std::runtime_error("json: " + message) {}
};

struct Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

/// JSON 值的通用容器。
///
/// 为了代码简单，所有可能的载荷都作为成员平铺在结构体里（而不是用
/// std::variant 或继承），实际有效的那一个由 kind 决定。JSON 文档本身很小，
/// 这点空间浪费可以忽略。
struct Value {
    /// 判别标签，决定下面哪个成员有意义。
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;          ///< kind == Bool 时有效
    double number = 0.0;           ///< kind == Number 时有效
    std::string text;              ///< kind == String 时有效
    std::vector<Value> items;      ///< kind == Array 时有效
    std::map<std::string, Value> members;  ///< kind == Object 时有效

    bool is_null() const { return kind == Kind::Null; }
    bool is_array() const { return kind == Kind::Array; }
    bool is_object() const { return kind == Kind::Object; }
    bool is_number() const { return kind == Kind::Number; }
    bool is_string() const { return kind == Kind::String; }

    /// 数组返回元素个数；对象返回成员个数。
    std::size_t size() const { return kind == Kind::Array ? items.size() : members.size(); }

    /// 按下标访问数组元素（越界抛异常）。
    const Value& operator[](std::size_t index) const {
        if (kind != Kind::Array || index >= items.size()) {
            throw ParseError("array index out of range");
        }
        return items[index];
    }

    /// 按键访问对象成员（类型不符或键不存在都抛异常）。
    const Value& operator[](std::string_view key) const {
        if (kind != Kind::Object) {
            throw ParseError("member lookup on non-object");
        }
        auto it = members.find(std::string(key));
        if (it == members.end()) {
            throw ParseError("missing key '" + std::string(key) + "'");
        }
        return it->second;
    }

    /// 判断对象里是否存在某个键（不抛异常，用于可选字段）。
    bool has(std::string_view key) const {
        return kind == Kind::Object && members.find(std::string(key)) != members.end();
    }

    // ---- 类型转换辅助（都带类型检查）------------------------------------

    double as_double() const {
        if (kind != Kind::Number) throw ParseError("expected a number");
        return number;
    }

    /// 四舍五入取整。JSON 里的 20.0 与 20 等价，这里用 lround 兼容两种写法。
    int as_int() const {
        if (kind != Kind::Number) throw ParseError("expected a number");
        return static_cast<int>(std::lround(number));
    }

    bool as_bool() const {
        if (kind != Kind::Bool) throw ParseError("expected a bool");
        return boolean;
    }

    const std::string& as_string() const {
        if (kind != Kind::String) throw ParseError("expected a string");
        return text;
    }

    /// 把数字数组整体转成 std::vector<double>。
    std::vector<double> to_doubles() const {
        if (kind != Kind::Array) throw ParseError("expected an array of numbers");
        std::vector<double> out;
        out.reserve(items.size());
        for (const Value& item : items) out.push_back(item.as_double());
        return out;
    }

    /// 把数字数组整体转成 std::vector<int>（用于骨骼父节点索引等）。
    std::vector<int> to_ints() const {
        std::vector<double> values = to_doubles();
        std::vector<int> out;
        out.reserve(values.size());
        for (double value : values) out.push_back(static_cast<int>(std::lround(value)));
        return out;
    }

    /// 把字符串数组整体转成 std::vector<std::string>（用于关节名字列表）。
    std::vector<std::string> to_strings() const {
        if (kind != Kind::Array) throw ParseError("expected an array of strings");
        std::vector<std::string> out;
        out.reserve(items.size());
        for (const Value& item : items) out.push_back(item.as_string());
        return out;
    }
};

/// 递归下降解析器。
///
/// 文法与 JSON 标准一致，解析入口是 parse()。解析过程中 cursor_ 始终指向
/// "下一个待读字符"，所有 expect/peek 都基于它。
class Parser {
 public:
    explicit Parser(std::string_view text) : text_(text) {}

    /// 解析整段文本；若末尾还有多余字符则报错（防止传错文件类型）。
    Value parse() {
        skip_space();
        Value value = parse_value();
        skip_space();
        if (cursor_ != text_.size()) throw ParseError("trailing characters");
        return value;
    }

 private:
    /// 跳过空白（JSON 只允许这四种空白字符）。
    void skip_space() {
        while (cursor_ < text_.size()) {
            char c = text_[cursor_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++cursor_;
            } else {
                break;
            }
        }
    }

    /// 查看当前字符但不前进；已到末尾则抛异常。
    char peek() const {
        if (cursor_ >= text_.size()) throw ParseError("unexpected end of input");
        return text_[cursor_];
    }

    /// 断言当前字符是 expected 并前进一格。
    void expect(char expected) {
        if (cursor_ >= text_.size() || text_[cursor_] != expected) {
            throw ParseError(std::string("expected '") + expected + "'");
        }
        ++cursor_;
    }

    /// 尝试匹配一段字面量（true/false/null），成功则前进并返回 true。
    bool consume_literal(std::string_view literal) {
        if (text_.substr(cursor_, literal.size()) == literal) {
            cursor_ += literal.size();
            return true;
        }
        return false;
    }

    /// 按首字符分派到具体的解析函数。
    Value parse_value() {
        switch (peek()) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': {
                Value value;
                value.kind = Value::Kind::String;
                value.text = parse_string();
                return value;
            }
            case 't': {
                if (!consume_literal("true")) throw ParseError("bad literal");
                Value value;
                value.kind = Value::Kind::Bool;
                value.boolean = true;
                return value;
            }
            case 'f': {
                if (!consume_literal("false")) throw ParseError("bad literal");
                Value value;
                value.kind = Value::Kind::Bool;
                value.boolean = false;
                return value;
            }
            case 'n': {
                if (!consume_literal("null")) throw ParseError("bad literal");
                return Value{};
            }
            default: return parse_number();
        }
    }

    /// 解析 {...}。允许空对象 {}。
    Value parse_object() {
        expect('{');
        Value value;
        value.kind = Value::Kind::Object;
        skip_space();
        if (peek() == '}') {
            ++cursor_;
            return value;
        }
        while (true) {
            skip_space();
            std::string key = parse_string();
            skip_space();
            expect(':');
            skip_space();
            value.members.emplace(std::move(key), parse_value());
            skip_space();
            char c = peek();
            if (c == ',') {
                ++cursor_;
                continue;
            }
            expect('}');
            break;
        }
        return value;
    }

    /// 解析 [...]。允许空数组 []。
    Value parse_array() {
        expect('[');
        Value value;
        value.kind = Value::Kind::Array;
        skip_space();
        if (peek() == ']') {
            ++cursor_;
            return value;
        }
        while (true) {
            skip_space();
            value.items.push_back(parse_value());
            skip_space();
            char c = peek();
            if (c == ',') {
                ++cursor_;
                continue;
            }
            expect(']');
            break;
        }
        return value;
    }

    /// 把 Unicode 码点按 UTF-8 编码追加到输出串。
    ///
    /// 四种长度分支分别对应 1/2/3/4 字节的 UTF-8 序列。
    static void append_utf8(std::string& out, std::uint32_t code) {
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    /// 解析带引号的字符串，处理全部标准转义序列。
    ///
    /// \uXXXX 解析出的若是高位代理（0xD800-0xDBFF），会继续读取紧随其后的
    /// \uXXXX 低位代理并合成真正的码点（BMP 之外的字符，例如 emoji）。
    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            char c = peek();
            ++cursor_;
            if (c == '"') break;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            char escape = peek();
            ++cursor_;
            switch (escape) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    std::uint32_t code = 0;
                    for (int i = 0; i < 4; ++i) {
                        char digit = peek();
                        ++cursor_;
                        code <<= 4;
                        if (digit >= '0' && digit <= '9') code |= static_cast<std::uint32_t>(digit - '0');
                        else if (digit >= 'a' && digit <= 'f') code |= static_cast<std::uint32_t>(digit - 'a' + 10);
                        else if (digit >= 'A' && digit <= 'F') code |= static_cast<std::uint32_t>(digit - 'A' + 10);
                        else throw ParseError("bad \\u escape");
                    }
                    // 代理对合并：0x10000 + (high - 0xD800) * 0x400 + (low - 0xDC00)
                    if (code >= 0xD800 && code <= 0xDBFF && text_.substr(cursor_, 2) == "\\u") {
                        cursor_ += 2;
                        std::uint32_t low = 0;
                        for (int i = 0; i < 4; ++i) {
                            char digit = peek();
                            ++cursor_;
                            low <<= 4;
                            if (digit >= '0' && digit <= '9') low |= static_cast<std::uint32_t>(digit - '0');
                            else if (digit >= 'a' && digit <= 'f') low |= static_cast<std::uint32_t>(digit - 'a' + 10);
                            else if (digit >= 'A' && digit <= 'F') low |= static_cast<std::uint32_t>(digit - 'A' + 10);
                            else throw ParseError("bad \\u escape");
                        }
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, code);
                    break;
                }
                default: throw ParseError("bad escape sequence");
            }
        }
        return out;
    }

    /// 解析数字：可选负号、整数部分、可选小数部分、可选指数部分。
    ///
    /// 用 strtod 做最终转换（已由前面的扫描保证是合法数字文本），
    /// 这样能正确处理 "1e-5" 这类科学计数法。
    Value parse_number() {
        std::size_t start = cursor_;
        if (peek() == '-') ++cursor_;
        while (cursor_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[cursor_]))) ++cursor_;
        if (cursor_ < text_.size() && text_[cursor_] == '.') {
            ++cursor_;
            while (cursor_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[cursor_]))) ++cursor_;
        }
        if (cursor_ < text_.size() && (text_[cursor_] == 'e' || text_[cursor_] == 'E')) {
            ++cursor_;
            if (cursor_ < text_.size() && (text_[cursor_] == '+' || text_[cursor_] == '-')) ++cursor_;
            while (cursor_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[cursor_]))) ++cursor_;
        }
        if (cursor_ == start) throw ParseError("expected a value");
        std::string token(text_.substr(start, cursor_ - start));
        Value value;
        value.kind = Value::Kind::Number;
        value.number = std::strtod(token.c_str(), nullptr);
        return value;
    }

    std::string_view text_;      ///< 待解析的完整文本（不持有所有权）
    std::size_t cursor_ = 0;     ///< 当前读取位置
};

/// 便捷入口：解析一段文本。
inline Value parse(std::string_view text) { return Parser(text).parse(); }

}  // namespace json
}  // namespace ardy
