/**
 * @file meter_reader_tflite.cpp
 * @brief Implementation of MeterReaderTFLite component for ESPHome.
 * 
 * This file contains the implementation of the meter reading component
 * that uses TensorFlow Lite Micro for digit recognition from camera images.
 */

#include "meter_reader_tflite.h"
#include "esp_log.h"
#include "debug_utils.h"
#include "model_config.h"
#include <numeric>

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

void MeterReaderTFLite::setup() {
    ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
    
    // Validate essential configuration
    if (camera_width_ == 0 || camera_height_ == 0) {
        ESP_LOGE(TAG, "Camera dimensions not set!");
        mark_failed();
        return;
    }
    
    if (!camera_) {
        ESP_LOGE(TAG, "No camera configured!");
        mark_failed();
        return;
    }
	
	// Apply crop zones from global variable if available
	crop_zone_handler_.apply_global_zones();

    // Setup camera callback with frame buffer management
    camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
        // Only accept frames if requested and not currently processing
        if (frame_requested_.load() && !processing_frame_.load()) {
            pending_frame_ = image;
            frame_available_.store(true);
            frame_requested_.store(false);
            last_frame_received_ = millis();
            ESP_LOGD(TAG, "Frame captured successfully");
        } else {
            frames_skipped_++;
        }
    });

    // Delay model loading to allow system stabilization
    ESP_LOGI(TAG, "Model loading will begin in 30 seconds...");
    this->set_timeout(30000, [this]() {
        ESP_LOGI(TAG, "Starting model loading...");
        
        if (!this->load_model()) {
            ESP_LOGE(TAG, "Failed to load model");
            this->mark_failed();
            return;
        }

        this->model_loaded_ = true;
        
        ESP_LOGI(TAG, "Setting up Image Processor...");
        image_processor_ = std::make_unique<ImageProcessor>(
            ImageProcessorConfig{camera_width_, camera_height_, pixel_format_},
            &model_handler_
        );
            
        ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
        this->print_debug_info();
    });
}

void MeterReaderTFLite::update() {
    // Skip update if system not ready
    if (!model_loaded_ || !camera_) {
        ESP_LOGW(TAG, "Update skipped - system not ready");
        return;
    }

    // Reset processing state for new update cycle
    processing_frame_.store(false);
    
    // Request new frame if none available or pending
    if (!frame_available_.load() && !frame_requested_.load()) {
        frame_requested_.store(true);
        last_request_time_ = millis();
        ESP_LOGD(TAG, "Requesting new frame");
    } else if (frame_available_.load()) {
        // Process existing frame immediately
        ESP_LOGD(TAG, "Processing available frame");
        process_available_frame();
    }
}

void MeterReaderTFLite::loop() {
    // Process available frame if update() has triggered processing
    if (frame_available_.load() && !processing_frame_.load()) {
        process_available_frame();
    }

    // Handle frame request timeout (10 seconds)
    if (frame_requested_.load() && (millis() - last_request_time_ > 10000)) {
        ESP_LOGW(TAG, "Frame request timeout - no frame received in 10 seconds");
        frame_requested_.store(false);
        frames_skipped_++;
    }
}

void MeterReaderTFLite::process_available_frame() {
    processing_frame_.store(true);
    
    std::shared_ptr<camera::CameraImage> frame;
    {
        // Atomic frame retrieval
        frame = pending_frame_;
        pending_frame_.reset();
        frame_available_.store(false);
    }
    
    if (frame && frame->get_data_buffer() && frame->get_data_length() > 0) {
        ESP_LOGD(TAG, "Processing frame (%zu bytes)", frame->get_data_length());
        process_full_image(frame);
        frames_processed_++;
    } else {
        ESP_LOGE(TAG, "Invalid frame available for processing");
    }
    
    processing_frame_.store(false);
}

void MeterReaderTFLite::process_full_image(std::shared_ptr<camera::CameraImage> frame) {
    DURATION_START();
    
    // Validate input frame
    if (!frame || !frame->get_data_buffer() || frame->get_data_length() == 0) {
        ESP_LOGE(TAG, "Invalid frame received for processing");
        DURATION_END("process_full_image (invalid frame)");
        return;
    }

    ESP_LOGD(TAG, "Processing frame (%zu bytes)", frame->get_data_length());

    // try {
        // Get configured crop zones or use full image
        auto zones = crop_zone_handler_.get_zones();
        if (zones.empty()) {
            zones.push_back({0, 0, camera_width_, camera_height_});
            ESP_LOGD(TAG, "No crop zones configured - using full frame");
        }

        // Process image through pipeline
        auto processed_zones = image_processor_->split_image_in_zone(frame, zones);
        
        std::vector<float> readings;
        std::vector<float> confidences;
        bool processing_success = true;

        // Process each zone through model
        for (auto& result : processed_zones) {
            float value, confidence;
            if (process_model_result(result, &value, &confidence)) {
                readings.push_back(value);
                confidences.push_back(confidence);
                
                ESP_LOGD(TAG, "Zone result - Value: %.1f, Confidence: %.2f", 
                        value, confidence);
            } else {
                ESP_LOGE(TAG, "Model result processing failed for zone");
                processing_success = false;
                break;
            }
        }

        // Publish results if successful
        if (processing_success && !readings.empty()) {
            float final_reading = combine_readings(readings);
            float avg_confidence = std::accumulate(confidences.begin(), 
                                                 confidences.end(), 0.0f) / confidences.size();
            
            ESP_LOGI(TAG, "Final reading: %.1f (avg confidence: %.2f)", 
                    final_reading, avg_confidence);

            if (value_sensor_) {
                value_sensor_->publish_state(final_reading);
            }
            
            if (confidence_sensor_ != nullptr) {
                confidence_sensor_->publish_state(avg_confidence);
            }
            
            if (debug_mode_) {
                this->print_debug_info();
            }
        } else {
            ESP_LOGE(TAG, "Frame processing failed");
        }
        
    // } catch (const std::exception& e) {
        // ESP_LOGE(TAG, "Exception during image processing: %s", e.what());
    // }

    DURATION_END("process_full_image");
}

bool MeterReaderTFLite::process_model_result(const ImageProcessor::ProcessResult& result, 
                                           float* value, float* confidence) {
    // Invoke model with processed image data
    if (!model_handler_.invoke_model(result.data->get(), result.size)) {
        ESP_LOGE(TAG, "Model invocation failed");
        return false;
    }

    // Get model output
    const float* output = model_handler_.get_output();
    if (!output) {
        ESP_LOGE(TAG, "Failed to get model output");
        return false;
    }

    // Find highest probability class
    float max_prob = 0.0f;
    int predicted_class = 0;
    const int num_classes = model_handler_.get_output_size();
    
    for (int i = 0; i < num_classes; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    // Handle digit '0' (class 10 represents 0)
    if (predicted_class == 10) {
        predicted_class = 0;
    }

    *value = static_cast<float>(predicted_class);
    *confidence = max_prob;

    ESP_LOGD(TAG, "Model output - Class: %d, Confidence: %.2f", predicted_class, max_prob);
    return true;
}

void MeterReaderTFLite::set_model(const uint8_t *model, size_t length) {
    model_ = model;
    model_length_ = length;
    ESP_LOGD(TAG, "Model set: %zu bytes", length);
}

void MeterReaderTFLite::set_camera_image_format(int width, int height, const std::string &pixel_format) {
    camera_width_ = width;
    camera_height_ = height;
    pixel_format_ = pixel_format;
    ESP_LOGD(TAG, "Camera format set: %dx%d, %s", width, height, pixel_format.c_str());
}

float MeterReaderTFLite::combine_readings(const std::vector<float> &readings) {
    float combined_value = 0.0f;
    float multiplier = 1.0f;
    
    // Combine digits from least significant to most significant
    for (auto it = readings.rbegin(); it != readings.rend(); ++it) {
        combined_value += (*it) * multiplier;
        multiplier *= 10.0f;
    }
    
    return combined_value;
}

MeterReaderTFLite::~MeterReaderTFLite() {
    // Clean up pending frame
    pending_frame_.reset();
}

size_t MeterReaderTFLite::available() const {
    return 0; // Frames processed directly in callback
}

uint8_t *MeterReaderTFLite::peek_data_buffer() {
    return nullptr; // Image data handled internally
}

void MeterReaderTFLite::consume_data(size_t consumed) {
    // Not used - image processed in one go
}

void MeterReaderTFLite::return_image() {
    // Image released after processing completes
}

void MeterReaderTFLite::set_image(std::shared_ptr<camera::CameraImage> image) {
    // Part of CameraImageReader interface - not used directly
}

void MeterReaderTFLite::set_model_config(const std::string &model_type) {
    model_type_ = model_type;
}

void MeterReaderTFLite::print_debug_info() {
    ESP_LOGI(TAG, "--- MeterReaderTFLite Debug Info ---");
    ESP_LOGI(TAG, "  Model Loaded: %s", model_loaded_ ? "Yes" : "No");
    ESP_LOGI(TAG, "  Camera Dimensions: %dx%d", camera_width_, camera_height_);
    ESP_LOGI(TAG, "  Pixel Format: %s", pixel_format_.c_str());
    ESP_LOGI(TAG, "  Confidence Threshold: %.2f", confidence_threshold_);
    ESP_LOGI(TAG, "  Tensor Arena Size (Requested): %zu bytes", tensor_arena_size_requested_);
    ESP_LOGI(TAG, "  Tensor Arena Size (Actual): %zu bytes", tensor_arena_allocation_.actual_size);
    ESP_LOGI(TAG, "  Model Size: %zu bytes", model_length_);
    ESP_LOGI(TAG, "  Frames Processed: %lu", frames_processed_);
    ESP_LOGI(TAG, "  Frames Skipped: %lu", frames_skipped_);
    ESP_LOGI(TAG, "  Debug Mode: %s", debug_mode_ ? "Enabled" : "Disabled");
    
    memory_manager_.report_memory_status(
        tensor_arena_size_requested_,
        tensor_arena_allocation_.actual_size,
        model_handler_.get_arena_peak_bytes(),
        model_length_
    );
    ESP_LOGI(TAG, "----------------------------------");
}

// void MeterReaderTFLite::print_debug_info() {
    // print_meter_reader_debug_info(this);
// }

bool MeterReaderTFLite::load_model() {
    DURATION_START();
    ESP_LOGI(TAG, "Loading TFLite model...");
    
    // Get model configuration
    ModelConfig config;
    auto it = MODEL_CONFIGS.find(model_type_);
    if (it != MODEL_CONFIGS.end()) {
        config = it->second;
        ESP_LOGI(TAG, "Using model config: %s", config.description.c_str());
    } else {
        config = DEFAULT_MODEL_CONFIG;
        ESP_LOGW(TAG, "Model type '%s' not found, using default config: %s", 
                model_type_.c_str(), config.description.c_str());
    }

    // Allocate tensor arena
    tensor_arena_allocation_ = MemoryManager::allocate_tensor_arena(tensor_arena_size_requested_);
    if (!tensor_arena_allocation_) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return false;
    }

    // Load the model
    if (!model_handler_.load_model(model_, model_length_, 
                                 tensor_arena_allocation_.data.get(), 
                                 tensor_arena_allocation_.actual_size,
                                 config)) {
        ESP_LOGE(TAG, "Failed to load model into interpreter");
        return false;
    }

    ESP_LOGI(TAG, "Model loaded successfully");
    ESP_LOGI(TAG, "Input dimensions: %dx%dx%d", 
            model_handler_.get_input_width(),
            model_handler_.get_input_height(),
            model_handler_.get_input_channels());
    
    DURATION_END("load_model");
    return true;
}

void MeterReaderTFLite::set_crop_zones(const std::string &zones_json) {
    ESP_LOGI(TAG, "Setting crop zones from JSON");
    crop_zone_handler_.parse_zones(zones_json);
    
    // Set default zone if none parsed
    if (crop_zone_handler_.get_zones().empty()) {
        ESP_LOGI(TAG, "No zones found in JSON, setting default zone");
        crop_zone_handler_.set_default_zone(camera_width_, camera_height_);
    }
    
    ESP_LOGI(TAG, "Configured %d crop zones", crop_zone_handler_.get_zones().size());
}

bool MeterReaderTFLite::allocate_tensor_arena() {
    ESP_LOGI(TAG, "Allocating tensor arena: %zu bytes", tensor_arena_size_requested_);
    
    tensor_arena_allocation_ = MemoryManager::allocate_tensor_arena(tensor_arena_size_requested_);
    if (!tensor_arena_allocation_) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return false;
    }
    
    ESP_LOGI(TAG, "Tensor arena allocated successfully: %zu bytes", 
            tensor_arena_allocation_.actual_size);
    return true;
}

size_t MeterReaderTFLite::get_arena_peak_bytes() const {
    return model_handler_.get_arena_peak_bytes();
}


// void MeterReaderTFLite::set_crop_zones_global(GlobalVarComponent<std::string> *crop_zones_global) {
    // crop_zones_global_ = crop_zones_global;
    // ESP_LOGI(TAG, "Crop zones global variable set");
// }


#ifdef DEBUG_METER_READER_TFLITE
class DebugCameraImage : public camera::CameraImage {
public:
    DebugCameraImage(const uint8_t* data, size_t size, int width, int height)
        : data_(data, data + size), width_(width), height_(height) {}

    uint8_t* get_data_buffer() override { return data_.data(); }
    size_t get_data_length() override { return data_.size(); }
    bool was_requested_by(camera::CameraRequester requester) const override { 
        return false;  // Debug image isn’t tied to requester
    }

    int get_width() const { return width_; }
    int get_height() const { return height_; }

private:
    std::vector<uint8_t> data_;
    int width_;
    int height_;
};

void MeterReaderTFLite::set_debug_image(const uint8_t* data, size_t size) {
    debug_image_ = std::make_shared<DebugCameraImage>(
        data, size, camera_width_, camera_height_);
    ESP_LOGI(TAG, "Debug image set: %zu bytes (%dx%d)", 
             size, camera_width_, camera_height_);
}

void MeterReaderTFLite::test_with_debug_image() {
    if (debug_image_) {
        ESP_LOGI(TAG, "Processing with debug image...");
        process_full_image(debug_image_);
    } else {
        ESP_LOGE(TAG, "No debug image set to process.");
    }
}

void MeterReaderTFLite::set_debug_mode(bool debug_mode) {
    debug_mode_ = debug_mode;
    ESP_LOGI(TAG, "Debug mode %s", debug_mode ? "enabled" : "disabled");
}
#endif

}  // namespace meter_reader_tflite
}  // namespace esphome