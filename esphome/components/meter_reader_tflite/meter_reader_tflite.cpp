#include "meter_reader_tflite.h"
#include "esp_log.h"
#include "debug_utils.h"
#include "model_config.h"

/**
 * @file meter_reader_tflite.cpp
 * @brief ESPHome component for meter reading using TensorFlow Lite Micro.
 *
 * This component captures images from an ESP32 camera, processes them,
 * and uses a TensorFlow Lite Micro model to read meter values.
 */

/**
 * @brief Namespace for the ESPHome meter_reader_tflite component.
 */
namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

/**
 * @brief Sets up the MeterReaderTFLite component.
 *
 * This function validates camera configuration, sets up the image callback,
 * and schedules the model loading after a delay.
 */




/**
 * @brief The main setup function for the MeterReaderTFLite component.
 * Initializes camera, sets up image callbacks, and loads the TFLite model.
 */
void MeterReaderTFLite::setup() {
    ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
    
    // Validate essential configuration first
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

    // Create frame queue with capacity for 2 frames (double buffering)
    frame_queue_ = xQueueCreate(2, sizeof(std::shared_ptr<camera::CameraImage>));
    if (frame_queue_ == nullptr) {
        ESP_LOGE(TAG, "Failed to create frame queue!");
        mark_failed();
        return;
    }

    // Add image callback to the camera component to receive frames.
    camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
        // Only queue frames if we're actively processing
        if (!process_next_frame_) {
            frames_skipped_++;
            return;
        }

        // Try to send the frame to the queue (non-blocking)
        BaseType_t result = xQueueSend(frame_queue_, &image, 0);
        if (result != pdPASS) {
            frames_skipped_++;
            ESP_LOGW(TAG, "Frame queue full, dropping frame");
        } else {
            ESP_LOGD(TAG, "Frame queued for processing");
        }
    });

    // Delay model loading to allow other components to initialize and stabilize.
    ESP_LOGI(TAG, "Model loading will begin in 5 seconds...");
    
    // Set a timeout to load the TensorFlow Lite model after a specified delay.
    this->set_timeout(5000, [this]() {
        ESP_LOGI(TAG, "Starting model loading...");
        
        // Attempt to load the TFLite model.
        if (!this->load_model()) {
            ESP_LOGE(TAG, "Failed to load model");
            this->print_debug_info();
            this->mark_failed();
            return;
        }

        // Mark model as loaded upon success.
        this->model_loaded_ = true;
        
        ESP_LOGI(TAG, "Setting up Image Processor...");

        // Initialize the ImageProcessor with camera and model configurations.
        image_processor_ = std::make_unique<ImageProcessor>(
            ImageProcessorConfig{camera_width_, camera_height_, pixel_format_},
            &model_handler_
        );
            
        ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
        this->print_debug_info();
            
        // Start periodic updates now that setup is complete.
        this->set_update_interval(this->get_update_interval());
    });
}


/**
 * @brief Called periodically to request a new frame for processing.
 *
 * This function is part of the PollingComponent interface and is responsible
 * for triggering image capture and processing when the system is ready.
 */
void MeterReaderTFLite::update() {
    // Skip update if the model is not loaded or camera is not configured.
    if (!model_loaded_ || !camera_) {
        ESP_LOGW(TAG, "Update skipped - system not ready");
        return;
    }

    // Request the next frame to be processed.
    process_next_frame_ = true;
    last_request_time_ = millis(); // Record the time of the last request.
    
    ESP_LOGD(TAG, "Frame processing requested");
}

/**
 * @brief The main loop function for the MeterReaderTFLite component.
 *
 * This function is called repeatedly and can be used for continuous monitoring
 * or handling long-running tasks. Currently, it checks for stale frame requests.
 */
void MeterReaderTFLite::loop() {
    // Process frames from the queue if model is loaded and we're ready
    if (model_loaded_ && process_next_frame_) {
        std::shared_ptr<camera::CameraImage> frame;
        
        // Try to receive a frame from the queue (non-blocking)
        if (xQueueReceive(frame_queue_, &frame, 0) == pdPASS) {
            // Process the frame
            process_full_image(frame);
            
            // Reset the flag to wait for the next update cycle
            process_next_frame_ = false;
        }
    }

    // Handle stale frame requests
    if (process_next_frame_ && (millis() - last_request_time_ > 10000)) {
        ESP_LOGW(TAG, "No frame received for 10 seconds after request");
        process_next_frame_ = false;
    }
}

/**
 * @brief Processes a full camera image, including cropping, scaling, and TFLite inference.
 *
 * @param frame A shared pointer to the camera image data.
 */
void MeterReaderTFLite::process_full_image(std::shared_ptr<camera::CameraImage> frame) {
    DURATION_START(); // Start duration measurement for this function.
    
    // Validate the input frame to ensure it's not null and contains data.
    if (!frame || !frame->get_data_buffer() || frame->get_data_length() == 0) {
        ESP_LOGE(TAG, "Invalid frame received for processing");
        this->print_debug_info(); // Print debug information on error.
        DURATION_END("process_full_image (invalid frame)");
        return;
    }

    ESP_LOGD(TAG, "Processing frame (%zu bytes)", frame->get_data_length());
    frames_processed_++;

    // Get configured crop zones. If no zones are defined, use the full image.
    auto zones = crop_zone_handler_.get_zones();
    if (zones.empty()) {
        zones.push_back({0, 0, camera_width_, camera_height_});
        ESP_LOGD(TAG, "No crop zones configured - using full frame");
    }

    // Prepare vectors to store readings and confidences from each processed zone.
    std::vector<float> readings;
    std::vector<float> confidences;
    bool processing_success = true;

    // Process the image through the ImageProcessor to split and prepare zones.
    auto processed_zones = image_processor_->split_image_in_zone(frame, zones);
    
    // Iterate through each processed zone and invoke the TFLite model.
    for (auto& result : processed_zones) {
        float value, confidence;
        // Process the model result to get the predicted value and confidence.
        if (process_model_result(result, &value, &confidence)) {
            readings.push_back(value);
            confidences.push_back(confidence);
            
            ESP_LOGD(TAG, "Zone result - Value: %.1f, Confidence: %.2f", 
                    value, confidence);
        } else {
            ESP_LOGE(TAG, "Model result processing failed for zone");
            this->print_debug_info();
            processing_success = false;
            break; // Stop processing if any zone fails.
        }
    }

    // Publish results if all zones were processed successfully and readings are available.
    if (processing_success && !readings.empty()) {
        float final_reading = combine_readings(readings); // Combine individual zone readings.
        float total_confidence = 0.0f;
        for (float conf : confidences) {
            total_confidence += conf;
        }
        float avg_confidence = total_confidence / confidences.size();
        
        ESP_LOGI(TAG, "Final reading: %.1f (avg confidence: %.2f)", 
                final_reading, avg_confidence);

        // Publish the final reading to the value sensor.
        if (value_sensor_) {
            value_sensor_->publish_state(final_reading);
        }
        
        // Publish confidence score if a confidence sensor is configured.
        if (confidence_sensor_ != nullptr) {
            confidence_sensor_->publish_state(avg_confidence);
        }
        
        // Print debug info if debug mode is enabled.
        if (debug_mode_) {
            this->print_debug_info();
        }
        
    } else {
        ESP_LOGE(TAG, "Frame processing failed");
        this->print_debug_info();
    }

    DURATION_END("process_full_image"); // End duration measurement.
}


/**
 * @brief Processes the output of the TensorFlow Lite model.
 *
 * @param result The processed image data and size.
 * @param value Pointer to store the predicted meter value.
 * @param confidence Pointer to store the confidence score of the prediction.
 * @return True if model invocation and result processing were successful, false otherwise.
 */
bool MeterReaderTFLite::process_model_result(const ImageProcessor::ProcessResult& result, float* value, float* confidence) {
    // Invoke the model with the processed image data.
    if (!model_handler_.invoke_model(result.data->get(), result.size)) {
        ESP_LOGE(TAG, "Model invocation failed");
        return false;
    }

    // Get the raw model output (inferred values).
    const float* output = model_handler_.get_output();
    if (!output) {
        ESP_LOGE(TAG, "Failed to get model output");
        return false;
    }

    // Find the highest probability class (simulating a softmax-like operation).
    float max_prob = 0.0f;
    int predicted_class = 0;
    const int num_classes = model_handler_.get_output_size();
    
    for (int i = 0; i < num_classes; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    // Special handling for digit '0' if needed (e.g., if class 10 represents 0).
    if (predicted_class == 10) {
        predicted_class = 0;
    }

    *value = static_cast<float>(predicted_class); // Assign the predicted digit as the value.
    *confidence = max_prob; // Assign the maximum probability as confidence.

    ESP_LOGD(TAG, "Model output - Class: %d, Confidence: %.2f", predicted_class, max_prob);
	
	// Publish confidence score if a confidence sensor is configured.
	if (confidence_sensor_ != nullptr) {
		confidence_sensor_->publish_state(max_prob);
	}

    return true;
}

/**
 * @brief Sets the TensorFlow Lite model data.
 *
 * @param model Pointer to the model data.
 * @param length Length of the model data in bytes.
 */
void MeterReaderTFLite::set_model(const uint8_t *model, size_t length) {
  model_ = model;
  model_length_ = length;
  ESP_LOGD(TAG, "Model set: %zu bytes", length);
}

/**
 * @brief Sets the camera image format and resolution.
 *
 * @param width Width of the camera image.
 * @param height Height of the camera image.
 * @param pixel_format Pixel format of the camera image (e.g., "RGB888", "JPEG").
 */
void MeterReaderTFLite::set_camera_image_format(int width, int height, const std::string &pixel_format) {
  camera_width_ = width;
  camera_height_ = height;
  pixel_format_ = pixel_format;
  ESP_LOGD(TAG, "Camera format set: %dx%d, %s", width, height, pixel_format.c_str());
}

/**
 * @brief Combines individual digit readings into a single float value.
 *
 * This function assumes the readings are ordered from most significant to least significant digit.
 * For example, readings {1.0, 2.0, 3.0} would result in 123.0.
 *
 * @param readings A vector of float values representing individual digit readings.
 * @return The combined float value.
 */
float MeterReaderTFLite::combine_readings(const std::vector<float> &readings) {
    float combined_value = 0.0f;
    float multiplier = 1.0f;
    // Iterate from the last reading (least significant digit) to the first.
    for (auto it = readings.rbegin(); it != readings.rend(); ++it) {
        combined_value += (*it) * multiplier;
        multiplier *= 10.0f;
    }
    return combined_value;
}

/**
 * @brief Destructor for the MeterReaderTFLite component.
 * Cleans up allocated resources, such as the FreeRTOS queue.
 */
MeterReaderTFLite::~MeterReaderTFLite() {
    if (frame_queue_ != nullptr) {
        // Clean up any remaining frames in the queue
        std::shared_ptr<camera::CameraImage> frame;
        while (xQueueReceive(frame_queue_, &frame, 0) == pdPASS) {
            frame.reset();
        }
        
        vQueueDelete(frame_queue_);
        frame_queue_ = nullptr;
    }
}

/**
 * @brief Returns the number of available image frames in the queue.
 * This is part of the CameraImageReader interface.
 * @return Always returns 0 as frames are processed directly in the callback.
 */
size_t MeterReaderTFLite::available() const {
    return 0; // Frames are processed directly in the callback, not queued here for external reading.
}

/**
 * @brief Peeks at the data buffer of the current image.
 * This is part of the CameraImageReader interface.
 * @return Always returns nullptr as image data is not exposed this way.
 */
uint8_t *MeterReaderTFLite::peek_data_buffer() {
    return nullptr; // Image data is handled internally.
}

/**
 * @brief Consumes data from the current image buffer.
 * This is part of the CameraImageReader interface.
 * @param consumed The number of bytes consumed. Not used in this implementation.
 */
void MeterReaderTFLite::consume_data(size_t consumed) {
    // Not used as image data is processed in one go.
}

/**
 * @brief Returns the current image, releasing its resources.
 * This is part of the CameraImageReader interface.
 */
void MeterReaderTFLite::return_image() {
    // Image is released after process_full_image completes.
}

/**
 * @brief Sets the current camera image for processing.
 * This is part of the CameraImageReader interface, but direct image processing
 * is handled via the camera callback.
 * @param image A shared pointer to the camera image.
 */
void MeterReaderTFLite::set_image(std::shared_ptr<camera::CameraImage> image) {
    // This method is part of the CameraImageReader interface but is not used
    // for direct image processing in this component. Images are received via
    // the add_image_callback mechanism.
}

/**
 * @brief Sets the model configuration based on a string identifier.
 * @param model_type A string indicating the type of model (e.g., "default").
 */
void MeterReaderTFLite::set_model_config(const std::string &model_type) {
    model_type_ = model_type;
    // Additional logic to load specific model configurations could go here.
}

/**
 * @brief Prints debug information about the component's state and memory usage.
 */
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
    
    // Report detailed memory status using the MemoryManager.
    memory_manager_.report_memory_status(
        tensor_arena_size_requested_,
        tensor_arena_allocation_.actual_size,
        model_handler_.get_arena_peak_bytes(),
        model_length_
    );
    ESP_LOGI(TAG, "----------------------------------");
}

#ifdef DEBUG_METER_READER_TFLITE
/**
 * @brief Sets the debug image data for testing purposes.
 * This image will be used instead of live camera feed when debug mode is active.
 * @param data Pointer to the debug image data.
 * @param size Size of the debug image data in bytes.
 */
void MeterReaderTFLite::set_debug_image(const uint8_t* data, size_t size) {
    // Create a CameraImage object from the static debug image data.
    // This allows the image processing pipeline to treat it like a live camera frame.
    debug_image_ = std::make_shared<camera::CameraImage>(data, size, camera_width_, camera_height_, pixel_format_);
    ESP_LOGI(TAG, "Debug image set: %zu bytes", size);
}

/**
 * @brief Triggers processing with the debug image.
 * This function is typically called once during setup when debug mode is enabled.
 */
void MeterReaderTFLite::test_with_debug_image() {
    if (debug_image_) {
        ESP_LOGI(TAG, "Processing with debug image...");
        process_full_image(debug_image_);
    } else {
        ESP_LOGE(TAG, "No debug image set to process.");
    }
}

/**
 * @brief Sets the debug mode flag.
 * @param debug_mode True to enable debug mode, false to disable.
 */
void MeterReaderTFLite::set_debug_mode(bool debug_mode) {
    debug_mode_ = debug_mode;
    ESP_LOGI(TAG, "Debug mode %s", debug_mode ? "enabled" : "disabled");
}
#endif


bool MeterReaderTFLite::load_model() {
    DURATION_START();
    ESP_LOGI(TAG, "Loading TFLite model...");
    
    // Get model configuration based on model type
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
    
    // If no zones were parsed, set a default zone
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

}  // namespace meter_reader_tflite
}  // namespace esphome


