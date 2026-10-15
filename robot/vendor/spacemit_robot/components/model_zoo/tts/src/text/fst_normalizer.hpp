/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef FST_NORMALIZER_HPP
#define FST_NORMALIZER_HPP

/**
 * FstNormalizer - kaldifst 的薄封装层
 *
 * 负责: 按顺序加载多个 FST 规则文件, 串行 Apply,
 *       文件缺失/加载失败时整体退化为 identity (原文返回)。
 *
 * 隔离目的: 避免 text_normalizer.cpp 直接 include kaldifst 头,
 *           同时在 TTS_USE_FST=OFF 时也能正常编译 (提供空实现)。
 */

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tts {
namespace text {

class FstNormalizer {
public:
    FstNormalizer();
    ~FstNormalizer();

    /// @brief 按给定顺序加载 FST 文件, 任一缺失即整体失败 (返回 false, 进入 identity 模式)
    /// @param paths 规则文件绝对路径列表, 按应用顺序给 (例如 date → phone → number)
    bool Load(const std::vector<std::string>& paths);

    /// @brief 是否有加载成功的规则
    bool Ready() const;

    /// @brief 对输入文本依次应用所有规则。未加载/异常时原文返回。
    std::string Apply(const std::string& text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
};

}  // namespace text
}  // namespace tts

#endif  // FST_NORMALIZER_HPP
