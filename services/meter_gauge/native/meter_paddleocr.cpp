#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include "ppocr_detector.h"

namespace {

// PP-OCR emits raw model text.  Escape only what would break the JSON
// envelope, and drop line breaks the recognizer occasionally produces.
void print_escaped(const std::string& text) {
    for (char c : text) {
        if (c == '\\' || c == '"') std::cout << '\\';
        if (c == '\n' || c == '\r') continue;
        std::cout << c;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path;
    std::string config_path;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--image") image_path = argv[++i];
        else if (arg == "--config") config_path = argv[++i];
    }
    if (image_path.empty() || config_path.empty()) {
        std::cerr << "usage: meter_paddleocr --image PATH --config YAML\n";
        return 2;
    }
    try {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("cannot read image");
        YAML::Node config = YAML::LoadFile(config_path);
        // The factory normally expands resource paths; this standalone tool
        // constructs the detector directly, so expand the home-relative model
        // paths here as well.
        const char* home = std::getenv("HOME");
        const std::string home_dir = home ? home : "";
        auto expand_home = [&](const std::string& value) {
            if (value.rfind("~/", 0) == 0 && !home_dir.empty()) return home_dir + value.substr(1);
            return value;
        };
        config["model_path"] = expand_home(config["model_path"].as<std::string>());
        config["default_params"]["rec_model_path"] = expand_home(config["default_params"]["rec_model_path"].as<std::string>());
        auto model = vision_deploy::PPOCRDetector::create(config, false);
        auto* ocr = dynamic_cast<vision_deploy::PPOCRDetector*>(model.get());
        if (!ocr) throw std::runtime_error("failed to create PPOCRDetector");
        auto items = ocr->detect_text(image);
        // image_width/image_height are emitted so callers can normalise the
        // polygon coordinates without re-decoding the image themselves.
        std::cout << "{\"ok\":true,\"image_width\":" << image.cols
                  << ",\"image_height\":" << image.rows << ",\"items\":[";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) std::cout << ',';
            const auto& item = items[i];
            std::cout << "{\"text\":\"";
            print_escaped(item.text);
            std::cout << "\",\"score\":" << item.score;
            // Quadrilateral in image pixels, ordered TL, TR, BR, BL.  Uploaded
            // photos have no calibrated display region, so the caller needs the
            // geometry to locate the reading among many other text boxes.
            std::cout << ",\"poly\":[";
            for (size_t k = 0; k < item.polygon.size(); ++k) {
                if (k) std::cout << ',';
                std::cout << '[' << item.polygon[k].x << ',' << item.polygon[k].y << ']';
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
