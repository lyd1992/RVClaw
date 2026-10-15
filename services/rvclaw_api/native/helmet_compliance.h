// Copyright 2026 RVClaw contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef RVCLAW_HELMET_COMPLIANCE_H
#define RVCLAW_HELMET_COMPLIANCE_H

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rvclaw {

struct Box {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

inline float width(const Box& box) { return std::max(0.0f, box.x2 - box.x1); }
inline float height(const Box& box) { return std::max(0.0f, box.y2 - box.y1); }
inline float area(const Box& box) { return width(box) * height(box); }

inline float intersection_over_union(const Box& lhs, const Box& rhs) {
    const Box overlap{
        std::max(lhs.x1, rhs.x1),
        std::max(lhs.y1, rhs.y1),
        std::min(lhs.x2, rhs.x2),
        std::min(lhs.y2, rhs.y2),
    };
    const float intersection = area(overlap);
    const float union_area = area(lhs) + area(rhs) - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

enum class HelmetState : std::uint8_t {
    kChecking = 0,
    kHelmet = 1,
    kNoHelmet = 2,
};

inline const char* helmet_state_text(HelmetState state) {
    if (state == HelmetState::kHelmet) return "HELMET";
    if (state == HelmetState::kNoHelmet) return "NO HELMET";
    return "CHECKING";
}

struct PersonTarget {
    Box bbox;
    int track_id = -1;
    float score = 0.0f;
};

struct PpeDetection {
    Box bbox;
    HelmetState state = HelmetState::kHelmet;
    float score = 0.0f;
};

struct PpeResult {
    Box bbox;
    HelmetState state = HelmetState::kHelmet;
    float score = 0.0f;
    int track_id = -1;
};

inline Box head_zone(const Box& person, float frame_width, float frame_height) {
    const float person_width = width(person);
    const float person_height = height(person);
    const float x_padding = person_width * 0.08f;
    return {
        std::max(0.0f, person.x1 - x_padding),
        std::max(0.0f, person.y1 - person_height * 0.05f),
        std::min(frame_width, person.x2 + x_padding),
        std::min(frame_height, person.y1 + person_height * 0.42f),
    };
}

// A person box that is clipped above the top edge cannot prove that the head
// is visible. Keep the track in CHECKING until a complete upper ROI appears.
inline bool head_zone_visible(
    const Box& person, float frame_width, float frame_height) {
    const float person_height = height(person);
    if (person_height <= 0.0f || frame_width <= 0.0f || frame_height <= 0.0f) {
        return false;
    }
    const float head_zone_top = person.y1 - person_height * 0.05f;
    constexpr float kEdgeEpsilon = 1.0f;
    return person.x1 > kEdgeEpsilon &&
           person.y1 > kEdgeEpsilon &&
           person.x2 < frame_width - kEdgeEpsilon &&
           person.y2 < frame_height - kEdgeEpsilon &&
           head_zone_top >= 0.0f;
}

class HelmetStateTracker {
public:
    explicit HelmetStateTracker(int no_helmet_checks = 3, int stale_frames = 60)
        : no_helmet_checks_(no_helmet_checks), stale_frames_(stale_frames) {}

    void update_people(const std::vector<PersonTarget>& people) {
        ++frame_index_;
        for (const auto& person : people) {
            if (person.track_id < 0) continue;
            auto& state = states_[person.track_id];
            state.last_seen_frame = frame_index_;
        }
        for (auto it = states_.begin(); it != states_.end();) {
            if (frame_index_ - it->second.last_seen_frame > stale_frames_) {
                it = states_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool needs_confirmation(const std::vector<PersonTarget>& people) const {
        for (const auto& person : people) {
            const auto it = states_.find(person.track_id);
            if (it != states_.end() && it->second.state == HelmetState::kChecking) {
                return true;
            }
        }
        return false;
    }

    // -1 means the person was not included in this PPE pass (for example, the
    // 5th tracked person). -2 means the head ROI was clipped and the result is
    // explicitly uncertain. Zero means the visible head ROI had no hard hat.
    void observe_hardhats(
        const std::vector<PersonTarget>& people,
        const std::vector<float>& hardhat_scores) {
        for (std::size_t index = 0; index < people.size(); ++index) {
            if (people[index].track_id < 0 || index >= hardhat_scores.size()) {
                continue;
            }
            auto& state = states_[people[index].track_id];
            state.last_seen_frame = frame_index_;
            if (hardhat_scores[index] == -2.0f) {
                state.state = HelmetState::kChecking;
                state.consecutive_misses = 0;
                state.hardhat_score = 0.0f;
                continue;
            }
            if (hardhat_scores[index] < 0.0f) continue;
            ++state.checks;
            if (hardhat_scores[index] > 0.0f) {
                state.state = HelmetState::kHelmet;
                state.consecutive_misses = 0;
                state.hardhat_score = hardhat_scores[index];
                continue;
            }
            ++state.consecutive_misses;
            if (state.consecutive_misses >= no_helmet_checks_) {
                state.state = HelmetState::kNoHelmet;
                state.hardhat_score = 0.0f;
            }
        }
    }

    std::vector<PpeResult> results(
        const std::vector<PersonTarget>& people) const {
        std::vector<PpeResult> output;
        output.reserve(people.size());
        for (const auto& person : people) {
            const auto it = states_.find(person.track_id);
            const auto state = it == states_.end() ? TrackState{} : it->second;
            output.push_back({
                person.bbox,
                state.state,
                state.state == HelmetState::kHelmet ? state.hardhat_score : 0.0f,
                person.track_id,
            });
        }
        return output;
    }

private:
    struct TrackState {
        HelmetState state = HelmetState::kChecking;
        int consecutive_misses = 0;
        int checks = 0;
        std::uint64_t last_seen_frame = 0;
        float hardhat_score = 0.0f;
    };

    int no_helmet_checks_;
    std::uint64_t stale_frames_;
    std::uint64_t frame_index_ = 0;
    std::unordered_map<int, TrackState> states_;
};

// Match SafetyVision's upstream rule: surface every PPE detection immediately.
// Use the best-overlapping SafetyVision Person box when IoU >= 0.05; otherwise
// keep the PPE detection's own box. There is deliberately no head-zone check,
// minimum-size filter, conflict suppression, tracking, or temporal voting.
inline std::vector<PpeResult> resolve_safetyvision(
    const std::vector<PersonTarget>& people,
    const std::vector<PpeDetection>& detections) {
    constexpr float kPersonIouMin = 0.05f;
    std::vector<PpeResult> results;
    results.reserve(detections.size());

    for (const auto& detection : detections) {
        const PersonTarget* best_person = nullptr;
        float best_iou = 0.0f;
        for (const auto& person : people) {
            const float iou = intersection_over_union(detection.bbox, person.bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_person = &person;
            }
        }

        results.push_back({
            best_person != nullptr && best_iou >= kPersonIouMin
                ? best_person->bbox
                : detection.bbox,
            detection.state,
            detection.score,
        });
    }
    return results;
}

}  // namespace rvclaw

#endif  // RVCLAW_HELMET_COMPLIANCE_H
