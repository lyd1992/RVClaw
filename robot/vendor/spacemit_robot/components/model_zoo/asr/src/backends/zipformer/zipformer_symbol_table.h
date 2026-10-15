/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Token symbol table for CTC decoding (tokens.txt).
 */

#ifndef ZIPFORMER_SYMBOL_TABLE_H
#define ZIPFORMER_SYMBOL_TABLE_H

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>

namespace zipformer {

class SymbolTable {
public:
    bool Load(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            return false;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            auto pos = line.rfind(' ');
            if (pos == std::string::npos) continue;
            std::string sym = line.substr(0, pos);
            int32_t id = std::stoi(line.substr(pos + 1));
            id2sym_[id] = sym;
        }
        return true;
    }

    std::string GetSymbol(int32_t id) const {
        auto it = id2sym_.find(id);
        if (it != id2sym_.end()) return it->second;
        return "<unk>";
    }

    int32_t Size() const { return static_cast<int32_t>(id2sym_.size()); }

private:
    std::unordered_map<int32_t, std::string> id2sym_;
};

}  // namespace zipformer

#endif  // ZIPFORMER_SYMBOL_TABLE_H
