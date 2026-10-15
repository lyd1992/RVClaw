/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Greedy CTC decoder with byte-level BPE support.
 */

#ifndef ZIPFORMER_CTC_DECODER_H
#define ZIPFORMER_CTC_DECODER_H

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "backends/zipformer/zipformer_symbol_table.h"

namespace zipformer {

class CtcDecoder {
public:
    explicit CtcDecoder(const SymbolTable& sym, int32_t blank_id = 0)
        : sym_(sym), blank_id_(blank_id) {}

    std::string Decode(const float* log_probs, int32_t T,
                        int32_t vocab_size) const {
        std::vector<int32_t> tokens;
        int32_t prev_id = -1;

        for (int32_t t = 0; t < T; ++t) {
            const float* frame = log_probs + t * vocab_size;
            int32_t best_id = 0;
            float best_val = frame[0];
            for (int32_t v = 1; v < vocab_size; ++v) {
                if (frame[v] > best_val) {
                    best_val = frame[v];
                    best_id = v;
                }
            }
            if (best_id != prev_id && best_id != blank_id_) {
                tokens.push_back(best_id);
            }
            prev_id = best_id;
        }

        std::string result;
        std::vector<uint8_t> byte_buf;

        for (int32_t id : tokens) {
            std::string sym = sym_.GetSymbol(id);

            if (sym.size() == 6 && sym[0] == '<' && sym[1] == '0' &&
                    sym[2] == 'x' && sym[5] == '>') {
                uint8_t byte = static_cast<uint8_t>(
                        strtol(sym.substr(3, 2).c_str(), nullptr, 16));
                byte_buf.push_back(byte);
                continue;
            }

            if (!byte_buf.empty()) {
                result.append(reinterpret_cast<const char*>(byte_buf.data()),
                                byte_buf.size());
                byte_buf.clear();
            }

            if (sym == "<blk>" || sym == "<sos/eos>" || sym == "<unk>") continue;

            // U+2581 (▁) is BPE word boundary
            if (sym.size() >= 3 && sym[0] == '\xe2' && sym[1] == '\x96' &&
                    sym[2] == '\x81') {
                result += sym.substr(3);
            } else {
                result += sym;
            }
        }

        if (!byte_buf.empty()) {
            result.append(reinterpret_cast<const char*>(byte_buf.data()),
                            byte_buf.size());
        }

        return result;
    }

private:
    const SymbolTable& sym_;
    int32_t blank_id_;
};

}  // namespace zipformer

#endif  // ZIPFORMER_CTC_DECODER_H
