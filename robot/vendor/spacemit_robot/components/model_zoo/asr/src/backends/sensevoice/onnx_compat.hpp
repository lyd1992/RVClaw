/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file onnx_compat.hpp
 * @brief ONNX Runtime compatibility layer for RISC-V and other platforms
 *
 * Include this header BEFORE any ONNX Runtime headers to fix
 * platform-specific compatibility issues.
 */

#ifndef ONNX_COMPAT_HPP
#define ONNX_COMPAT_HPP

#include <cstdint>

// RISC-V doesn't have native __fp16 support
// Define a workaround before including ONNX Runtime headers
#if defined(__riscv) || defined(__riscv__)
    #ifndef __fp16
        // Use uint16_t as storage type for fp16 on RISC-V
        // This allows compilation but disables native fp16 operations
        typedef uint16_t __fp16;
    #endif
#endif

// Now it's safe to include ONNX Runtime headers
#include <onnxruntime_cxx_api.h>

#endif  // ONNX_COMPAT_HPP
