// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// =============================================================================
//  BERT WordPiece 分词器（纯 C++，零第三方依赖）
// =============================================================================
//
// 为什么必须自己实现？
//   MiniLM 文本编码器是在 HuggingFace tokenizers 的处理流程下训练出来的。
//   分词结果哪怕只差一个 token，文本条件向量就会变，动作也跟着变。所以这里
//   完整复刻了那条流水线（顺序不能变）：
//
//     1. BertNormalizer     清理控制字符、处理中日韩字符、转小写、去重音
//     2. BertPreTokenizer   按空白切分、按标点切分、CJK 逐字切分
//     3. WordPiece          贪心最长匹配，非词首片段加 "##" 前缀
//     4. 模板处理           首尾加上 [CLS] ... [SEP]，token_type_ids 全 0
//
// 关于去重音：
//   真正的 BERT 用 Unicode NFD 分解再去掉组合符号，那需要一张完整的 Unicode
//   表。这里用一张手写的映射表覆盖 Latin-1 补充区和 Latin Extended-A，
//   足以处理 "café"、"naïve" 这类英文动作描述里偶尔出现的词；其他文字原样
//   保留（CJK 走逐字切分）。
//
// 与官方的对齐情况：tools/verify_tokenizer.py 用 25 条提示词对比
// HuggingFace tokenizers，结果 25/25 完全一致（含标点、连续空格、长句截断）。

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ardy {

class Tokenizer {
 public:
    /// 从 vocab.txt 加载词表。
    ///
    /// vocab.txt 每行一个 token，行号即 token id（BERT 的标准格式）。
    /// max_length 是模型训练时的上下文上限（MiniLM 为 128），超出部分会被
    /// 截断，但保留最后的 [SEP]。
    static Tokenizer from_file(const std::string& vocab_path, int max_length = 128) {
        Tokenizer tokenizer;
        tokenizer.max_length_ = max_length;
        std::ifstream stream(vocab_path, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot open vocabulary: " + vocab_path);
        std::string line;
        int index = 0;
        while (std::getline(stream, line)) {
            // Windows 换行会产生尾随 \r，必须去掉，否则词表键会带脏字符。
            if (!line.empty() && line.back() == '\r') line.pop_back();
            tokenizer.vocab_.emplace(line, index);
            ++index;
        }
        if (tokenizer.vocab_.empty()) throw std::runtime_error("empty vocabulary: " + vocab_path);
        tokenizer.unk_id_ = tokenizer.require("[UNK]");
        tokenizer.cls_id_ = tokenizer.require("[CLS]");
        tokenizer.sep_id_ = tokenizer.require("[SEP]");
        return tokenizer;
    }

    /// 分词结果，三个数组长度相同，直接对应文本编码器的三个输入。
    struct Encoding {
        std::vector<std::int64_t> ids;            ///< token id 序列
        std::vector<std::int64_t> attention_mask; ///< 全 1（本引擎不做 batch padding）
        std::vector<std::int64_t> token_type_ids; ///< 单句场景全 0
    };

    /// 把一句英文提示词编码成 token id。
    ///
    /// 流程：规范化 → 预切词 → 每个词做 WordPiece → 首尾插特殊 token → 截断。
    /// 注意循环里提前 break 的原因：要给结尾的 [SEP] 留一个位置。
    Encoding encode(const std::string& text) const {
        std::vector<std::string> words = pre_tokenize(normalize(text));
        std::vector<std::int64_t> ids;
        ids.reserve(words.size() + 2);
        ids.push_back(cls_id_);
        for (const std::string& word : words) {
            const std::vector<std::int64_t>& pieces = word_pieces(word);
            for (std::int64_t id : pieces) {
                ids.push_back(id);
                if (static_cast<int>(ids.size()) >= max_length_ - 1) break;
            }
            if (static_cast<int>(ids.size()) >= max_length_ - 1) break;
        }
        ids.push_back(sep_id_);
        truncate_keeping_final(ids);

        Encoding encoding;
        encoding.ids = ids;
        encoding.attention_mask.assign(ids.size(), 1);
        encoding.token_type_ids.assign(ids.size(), 0);
        return encoding;
    }

    std::size_t vocab_size() const { return vocab_.size(); }
    int max_length() const { return max_length_; }

 private:
    /// 取某个特殊 token 的 id，词表里没有就直接报错（说明 vocab.txt 不对）。
    int require(const std::string& token) const {
        auto it = vocab_.find(token);
        if (it == vocab_.end()) throw std::runtime_error("vocabulary is missing " + token);
        return it->second;
    }

    // 提示词超长时的截断策略：保留前 max_length-1 个，再把最后一个 [SEP] 补上。
    // 这与 HuggingFace 的“单序列右截断 + 保留结束标记”行为一致；
    // 如果连 [SEP] 一起截掉，模型收到的就是一句没有终止符的残缺句子。
    void truncate_keeping_final(std::vector<std::int64_t>& ids) const {
        if (static_cast<int>(ids.size()) <= max_length_) return;
        std::vector<std::int64_t> truncated;
        truncated.reserve(max_length_);
        for (int i = 0; i < max_length_ - 1; ++i) truncated.push_back(ids[i]);
        truncated.push_back(ids.back());
        ids.swap(truncated);
    }

    // ---- BERT 规范化（normalisation）-------------------------------------

    /// 控制字符（会被直接丢弃）。制表/换行/回车例外，它们当作空白处理。
    static bool is_control(std::uint32_t cp) {
        if (cp == 0x09 || cp == 0x0A || cp == 0x0D) return false;
        if (cp < 0x20 || cp == 0x7F) return true;
        return (cp >= 0x80 && cp <= 0x9F);
    }

    /// 各种空白字符，统一折叠成一个半角空格。
    static bool is_space(std::uint32_t cp) {
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
               cp == 0x0B || cp == 0x0C || cp == 0x00A0 || cp == 0x2028 || cp == 0x2029;
    }

    /// CJK 统一表意文字范围（与 BERT 原实现一致）。命中时要前后补空格，
    /// 保证它们会被单独切分成一个个 token。
    static bool is_chinese_char(std::uint32_t cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
               (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B920 && cp <= 0x2CEAF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }

    /// 标点：ASCII 标点 + 常用 CJK/全角标点。标点会被切成独立 token。
    static bool is_punctuation(std::uint32_t cp) {
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
            (cp >= 123 && cp <= 126)) {
            return true;
        }
        return (cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) ||
               (cp >= 0xFF00 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
               (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
    }

    /// 小写化 + 去重音（近似 NFD 的效果）。
    ///
    /// BERT 的 strip_accents 实际上是“NFD 分解后丢掉组合符号”，例如
    /// 'é' → 'e' + U+0301 → 'e'。这里用查表直接得到目标字母，省掉整张
    /// Unicode 分解表。ASCII 走快速分支。
    static std::uint32_t fold_latin(std::uint32_t cp) {
        // Latin-1 Supplement 与 Latin Extended-A 的重音字母映射（大写小写都覆盖）。
        static const std::unordered_map<std::uint32_t, std::uint32_t> table = {
            {0x00C0, 'a'}, {0x00C1, 'a'}, {0x00C2, 'a'}, {0x00C3, 'a'}, {0x00C4, 'a'}, {0x00C5, 'a'},
            {0x00C7, 'c'}, {0x00C8, 'e'}, {0x00C9, 'e'}, {0x00CA, 'e'}, {0x00CB, 'e'},
            {0x00CC, 'i'}, {0x00CD, 'i'}, {0x00CE, 'i'}, {0x00CF, 'i'}, {0x00D1, 'n'},
            {0x00D2, 'o'}, {0x00D3, 'o'}, {0x00D4, 'o'}, {0x00D5, 'o'}, {0x00D6, 'o'},
            {0x00D9, 'u'}, {0x00DA, 'u'}, {0x00DB, 'u'}, {0x00DC, 'u'}, {0x00DD, 'y'},
            {0x00E0, 'a'}, {0x00E1, 'a'}, {0x00E2, 'a'}, {0x00E3, 'a'}, {0x00E4, 'a'}, {0x00E5, 'a'},
            {0x00E7, 'c'}, {0x00E8, 'e'}, {0x00E9, 'e'}, {0x00EA, 'e'}, {0x00EB, 'e'},
            {0x00EC, 'i'}, {0x00ED, 'i'}, {0x00EE, 'i'}, {0x00EF, 'i'}, {0x00F1, 'n'},
            {0x00F2, 'o'}, {0x00F3, 'o'}, {0x00F4, 'o'}, {0x00F5, 'o'}, {0x00F6, 'o'},
            {0x00F9, 'u'}, {0x00FA, 'u'}, {0x00FB, 'u'}, {0x00FC, 'u'}, {0x00FD, 'y'}, {0x00FF, 'y'},
            {0x0100, 'a'}, {0x0101, 'a'}, {0x0102, 'a'}, {0x0103, 'a'}, {0x0104, 'a'}, {0x0105, 'a'},
            {0x0106, 'c'}, {0x0107, 'c'}, {0x0108, 'c'}, {0x0109, 'c'}, {0x010A, 'c'}, {0x010B, 'c'},
            {0x010C, 'c'}, {0x010D, 'c'}, {0x010E, 'd'}, {0x010F, 'd'}, {0x0112, 'e'}, {0x0113, 'e'},
            {0x0114, 'e'}, {0x0115, 'e'}, {0x0116, 'e'}, {0x0117, 'e'}, {0x0118, 'e'}, {0x0119, 'e'},
            {0x011A, 'e'}, {0x011B, 'e'}, {0x011C, 'g'}, {0x011D, 'g'}, {0x011E, 'g'}, {0x011F, 'g'},
            {0x0120, 'g'}, {0x0121, 'g'}, {0x0122, 'g'}, {0x0123, 'g'}, {0x0124, 'h'}, {0x0125, 'h'},
            {0x0126, 'h'}, {0x0127, 'h'}, {0x0128, 'i'}, {0x0129, 'i'}, {0x012A, 'i'}, {0x012B, 'i'},
            {0x012C, 'i'}, {0x012D, 'i'}, {0x012E, 'i'}, {0x012F, 'i'}, {0x0130, 'i'}, {0x0131, 'i'},
            {0x0132, 'i'}, {0x0133, 'i'}, {0x0134, 'j'}, {0x0135, 'j'}, {0x0136, 'k'}, {0x0137, 'k'},
            {0x0139, 'l'}, {0x013A, 'l'}, {0x013B, 'l'}, {0x013C, 'l'}, {0x013D, 'l'}, {0x013E, 'l'},
            {0x013F, 'l'}, {0x0140, 'l'}, {0x0141, 'l'}, {0x0142, 'l'}, {0x0143, 'n'}, {0x0144, 'n'},
            {0x0145, 'n'}, {0x0146, 'n'}, {0x0147, 'n'}, {0x0148, 'n'}, {0x014C, 'o'}, {0x014D, 'o'},
            {0x014E, 'o'}, {0x014F, 'o'}, {0x0150, 'o'}, {0x0151, 'o'}, {0x0154, 'r'}, {0x0155, 'r'},
            {0x0156, 'r'}, {0x0157, 'r'}, {0x0158, 'r'}, {0x0159, 'r'}, {0x015A, 's'}, {0x015B, 's'},
            {0x015C, 's'}, {0x015D, 's'}, {0x015E, 's'}, {0x015F, 's'}, {0x0160, 's'}, {0x0161, 's'},
            {0x0162, 't'}, {0x0163, 't'}, {0x0164, 't'}, {0x0165, 't'}, {0x0168, 'u'}, {0x0169, 'u'},
            {0x016A, 'u'}, {0x016B, 'u'}, {0x016C, 'u'}, {0x016D, 'u'}, {0x016E, 'u'}, {0x016F, 'u'},
            {0x0170, 'u'}, {0x0171, 'u'}, {0x0172, 'u'}, {0x0173, 'u'}, {0x0174, 'w'}, {0x0175, 'w'},
            {0x0176, 'y'}, {0x0177, 'y'}, {0x0178, 'y'}, {0x0179, 'z'}, {0x017A, 'z'}, {0x017B, 'z'},
            {0x017C, 'z'}, {0x017D, 'z'}, {0x017E, 'z'},
        };
        if (cp < 0x80) {
            return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
        }
        auto it = table.find(cp);
        if (it != table.end()) return it->second;
        return cp;
    }

    /// 把 Unicode 码点按 UTF-8 编码追加到输出串（1/2/3/4 字节分支）。
    static void append_utf8(std::string& out, std::uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    /// UTF-8 解码成码点数组。
    ///
    /// 非法起始字节直接记为 U+FFFD（替换字符），不抛异常 —— 分词器对脏输入
    /// 应该宽容，反正最后会落到 [UNK]。
    static std::vector<std::uint32_t> decode_utf8(const std::string& text) {
        std::vector<std::uint32_t> code_points;
        std::size_t i = 0;
        while (i < text.size()) {
            unsigned char lead = static_cast<unsigned char>(text[i]);
            std::uint32_t cp = 0;
            int extra = 0;
            if (lead < 0x80) {
                cp = lead;
            } else if ((lead & 0xE0) == 0xC0) {
                cp = lead & 0x1F;
                extra = 1;
            } else if ((lead & 0xF0) == 0xE0) {
                cp = lead & 0x0F;
                extra = 2;
            } else if ((lead & 0xF8) == 0xF0) {
                cp = lead & 0x07;
                extra = 3;
            } else {
                cp = 0xFFFD;
            }
            ++i;
            for (int k = 0; k < extra && i < text.size(); ++k, ++i) {
                cp = (cp << 6) | (static_cast<unsigned char>(text[i]) & 0x3F);
            }
            code_points.push_back(cp);
        }
        return code_points;
    }

    /// BERT 规范化的主循环：清控制字符 → 小写/去重音 → 折叠空白 → CJK 两侧补空格。
    std::string normalize(const std::string& text) const {
        std::string out;
        out.reserve(text.size());
        for (std::uint32_t cp : decode_utf8(text)) {
            if (is_control(cp)) continue;
            std::uint32_t folded = fold_latin(cp);
            if (is_space(folded)) folded = ' ';
            if (is_chinese_char(folded)) {
                out.push_back(' ');
                append_utf8(out, folded);
                out.push_back(' ');
                continue;
            }
            append_utf8(out, folded);
        }
        return out;
    }

    /// 预切词：按空白断开，标点与 CJK 各自成为独立片段。
    ///
    /// 例："walks forward, then stops." →
    ///     ["walks", "forward", ",", "then", "stops", "."]
    static std::vector<std::string> pre_tokenize(const std::string& normalized) {
        std::vector<std::string> words;
        std::string current;
        auto flush = [&]() {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        };
        for (std::uint32_t cp : decode_utf8(normalized)) {
            if (cp == ' ') {
                flush();
                continue;
            }
            if (is_chinese_char(cp) || is_punctuation(cp)) {
                flush();
                std::string piece;
                append_utf8(piece, cp);
                words.push_back(std::move(piece));
                continue;
            }
            append_utf8(current, cp);
        }
        flush();
        return words;
    }

    /// WordPiece 贪心最长匹配：把一个“词”拆成若干 subword。
    ///
    /// 算法：
    ///   - 从位置 start 开始，尝试从最长的候选开始查词表；
    ///   - 命中就记录 id，并把 start 移到该片段末尾继续；
    ///   - 全部命中即成功；只要有一个子串查不到，整个词退化成 [UNK]。
    ///
    /// 细节：非词首片段要加 "##" 前缀（例如 "forward" 可能切成
    /// "for" + "##ward"），这是 BERT 词表的约定。回溯时不能在 UTF-8
    /// 续字节中间截断，否则会把一个汉字劈成两半。
    ///
    /// 返回引用指向 thread_local 缓冲，避免每个词都拷贝一个 vector
    /// （分词在交互式使用里会被频繁调用）。
    const std::vector<std::int64_t>& word_pieces(const std::string& word) const {
        thread_local std::vector<std::int64_t> pieces;
        pieces.assign(1, unk_id_);
        if (word.size() > 100) return pieces;   // 超长词直接 [UNK]（与官方限制一致）

        const std::size_t length = word.size();
        std::vector<std::int64_t> resolved;
        std::size_t start = 0;
        while (start < length) {
            std::size_t end = length;
            std::int64_t found = -1;
            while (start < end) {
                std::string piece = word.substr(start, end - start);
                if (start > 0) piece = "##" + piece;
                auto it = vocab_.find(piece);
                if (it != vocab_.end()) {
                    found = it->second;
                    break;
                }
                --end;
                // Never split inside a UTF-8 continuation byte.
                while (end > start && (static_cast<unsigned char>(word[end]) & 0xC0) == 0x80) --end;
            }
            if (found < 0) return pieces;  // whole word becomes [UNK]
            resolved.push_back(found);
            start = end;
        }
        pieces.swap(resolved);
        return pieces;
    }

    std::unordered_map<std::string, int> vocab_;
    int max_length_ = 128;
    int unk_id_ = 0;
    int cls_id_ = 0;
    int sep_id_ = 0;
};

}  // namespace ardy
