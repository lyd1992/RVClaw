/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMMON_H
#define COMMON_H

/**
 * @file common.h
 * @brief Unified header for common CV utilities
 * 
 * This header includes all common utility modules for easy access.
 * Individual module headers can also be included directly for better compilation performance.
 */

// Common data types
#include "datatype.h"

// Image processing utilities
#include "image_processing.h"

// NMS and IoU utilities
#include "nms.h"

// Drawing utilities
#include "drawing.h"

// YOLO-specific utilities
#include "yolo_utils.h"

// Embedding utilities
#include "embedding_utils.h"

#endif  // COMMON_H
