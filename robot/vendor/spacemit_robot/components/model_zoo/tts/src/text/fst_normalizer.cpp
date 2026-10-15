/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "text/fst_normalizer.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef TTS_USE_FST
#include "kaldifst/csrc/text-normalizer.h"
#endif

namespace tts {
namespace text {

// =============================================================================
// Impl: 隐藏 kaldifst 头文件, 兼容 TTS_USE_FST=OFF 的空实现
// =============================================================================

struct FstNormalizer::Impl {
#ifdef TTS_USE_FST
    std::vector<std::unique_ptr<kaldifst::TextNormalizer>> normalizers;
#endif
};

FstNormalizer::FstNormalizer() : impl_(std::make_unique<Impl>()) {}
FstNormalizer::~FstNormalizer() = default;

// =============================================================================
// Load
// =============================================================================

bool FstNormalizer::Load(const std::vector<std::string>& paths) {
#ifdef TTS_USE_FST
    std::lock_guard<std::mutex> lock(mu_);
    impl_->normalizers.clear();

    for (const auto& path : paths) {
        try {
            auto n = std::make_unique<kaldifst::TextNormalizer>(path);
            impl_->normalizers.push_back(std::move(n));
        } catch (const std::exception& e) {
            std::cerr << "[FstNormalizer] Failed to load FST '" << path
                << "': " << e.what() << std::endl;
            impl_->normalizers.clear();
            return false;
        } catch (...) {
            std::cerr << "[FstNormalizer] Failed to load FST '" << path
                << "': unknown error" << std::endl;
            impl_->normalizers.clear();
            return false;
        }
    }
    return !impl_->normalizers.empty();
#else
    (void)paths;
    return false;
#endif
}

// =============================================================================
// Ready
// =============================================================================

bool FstNormalizer::Ready() const {
#ifdef TTS_USE_FST
    std::lock_guard<std::mutex> lock(mu_);
    return !impl_->normalizers.empty();
#else
    return false;
#endif
}

// =============================================================================
// Apply
// =============================================================================

std::string FstNormalizer::Apply(const std::string& text) const {
#ifdef TTS_USE_FST
    if (text.empty()) return text;

    std::lock_guard<std::mutex> lock(mu_);
    if (impl_->normalizers.empty()) return text;

    std::string out = text;
    try {
        for (const auto& n : impl_->normalizers) {
            out = n->Normalize(out);
        }
    } catch (const std::exception& e) {
        std::cerr << "[FstNormalizer] Apply failed: " << e.what()
            << ", returning original text" << std::endl;
        return text;
    } catch (...) {
        std::cerr << "[FstNormalizer] Apply failed with unknown error, returning original text"
            << std::endl;
        return text;
    }
    return out;
#else
    return text;
#endif
}

}  // namespace text
}  // namespace tts
