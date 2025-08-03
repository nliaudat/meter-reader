#include "meter_reader_tflite.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  
  if (camera_width_ == 0 || camera_height_ == 0) {
    ESP_LOGE(TAG, "Camera dimensions not set!");
    mark_failed();
    return;
  }

  ESP_LOGI(TAG, "Model loading delayed for 10 seconds ...");
  // Delay model loading by 10 sec
  this->set_timeout(10000, [this]() { //10 sec
    if (!this->load_model()) {
      ESP_LOGE(TAG, "Failed to load model. Marking component as failed.");
      this->mark_failed();
      return;
    }
    this->model_loaded_ = true;
    
	ESP_LOGI(TAG, "Setting up Image Processor ...");
    // Initialize image processor AFTER model is loaded
    image_processor_ = std::make_unique<ImageProcessor>(
      ImageProcessorConfig{
        camera_width_,
        camera_height_,
        pixel_format_
      },
      &model_handler_
    );
    
    ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
    
    // Process debug image if in debug mode
    if (debug_mode_ && debug_image_) {
      ESP_LOGI(TAG, "Processing debug image after model load");
      set_image(debug_image_);
      process_image();
    }
  });
}

void MeterReaderTFLite::set_model(const uint8_t *model, size_t length) {
  model_ = model;
  model_length_ = length;
  ESP_LOGD(TAG, "Model set: %zu bytes", length);
}

void MeterReaderTFLite::set_camera_format(int width, int height, const std::string &pixel_format) {
  camera_width_ = width;
  camera_height_ = height;
  pixel_format_ = pixel_format;
  ESP_LOGD(TAG, "Camera format set: %dx%d, %s", width, height, pixel_format.c_str());
  
  // Reinitialize image processor with new format
  if (model_loaded_) {
	  image_processor_ = std::make_unique<ImageProcessor>(
		  ImageProcessorConfig{
			camera_width_,
			camera_height_,
			pixel_format_
		  },
		  &model_handler_
		);
  }
}

void MeterReaderTFLite::set_camera(esp32_camera::ESP32Camera *camera) {
  camera_ = camera;
  camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
    this->set_image(image);
    this->process_image();
  });
  ESP_LOGD(TAG, "Camera callback registered");
}

void MeterReaderTFLite::set_value_sensor(sensor::Sensor *sensor) {
  value_sensor_ = sensor;
  ESP_LOGD(TAG, "Value sensor set");
}

void MeterReaderTFLite::set_crop_zones(const std::string &zones_json) {
  crop_zone_handler_.parse_zones(zones_json);
  ESP_LOGD(TAG, "Crop zones set: %s", zones_json.c_str());
}

void MeterReaderTFLite::set_image(std::shared_ptr<camera::CameraImage> image) {
  current_image_ = image;
  image_offset_ = 0;
}

size_t MeterReaderTFLite::available() const {
  if (!current_image_) return 0;
  return current_image_->get_data_length() - image_offset_;
}

uint8_t *MeterReaderTFLite::peek_data_buffer() {
  if (!current_image_) return nullptr;
  return current_image_->get_data_buffer() + image_offset_;
}

void MeterReaderTFLite::consume_data(size_t consumed) {
  if (!current_image_) return;
  image_offset_ += consumed;
}

void MeterReaderTFLite::return_image() {
  current_image_.reset();
  image_offset_ = 0;
}

void MeterReaderTFLite::update() {
  if (!model_loaded_) {
    ESP_LOGW(TAG, "Model not loaded, skipping update");
    return;
  }

  if (!camera_) {
    ESP_LOGE(TAG, "Camera not configured");
    return;
  }

  ESP_LOGD(TAG, "Requesting new image from camera");
  camera_->request_image(camera::IDLE);
}

void MeterReaderTFLite::set_model_config(const std::string &model_identifier) {
	
    // First try exact match
    auto it = MODEL_CONFIGS.find(model_identifier);
    
    // If not found, try to extract short name from filename
    if (it == MODEL_CONFIGS.end()) {
        std::string short_name;
        if (model_identifier.find("class100-0180") != std::string::npos) {
            short_name = "class100-0180";
        } 
        else if (model_identifier.find("class100-0173") != std::string::npos) {
            short_name = "class100-0173";
        }
        else if (model_identifier.find("class10-0900") != std::string::npos) {
            short_name = "class10-0900";
        }
        else if (model_identifier.find("class10-0810") != std::string::npos) {
            short_name = "class10-0810";
        }
        else if (model_identifier.find("mnist") != std::string::npos) {
            short_name = "mnist";
        }
        
        if (!short_name.empty()) {
            it = MODEL_CONFIGS.find(short_name);
            ESP_LOGD(TAG, "Converted filename %s to model type %s", 
                    model_identifier.c_str(), short_name.c_str());
        }
    }

    if (it != MODEL_CONFIGS.end()) {
        model_type_ = it->first; // Store the actual identifier used
        model_handler_.set_config(it->second);
        ESP_LOGD(TAG, "Model config set for: %s", model_identifier.c_str());
    } else {
        model_type_ = "default";
        model_handler_.set_config(MODEL_CONFIGS.at("class100-0180"));
        ESP_LOGW(TAG, "Unknown model identifier: %s, using default", 
                model_identifier.c_str());
    }
}

bool MeterReaderTFLite::load_model() {
  if (!allocate_tensor_arena()) {
    return false;
  }

  if (!model_handler_.load_model(model_, model_length_,
                               tensor_arena_allocation_.data.get(),
                               tensor_arena_size_actual_,
                               model_handler_.get_config())) {
    return false;
  }

  report_memory_status();
  return true;
}

void MeterReaderTFLite::process_image() {
  if (debug_mode_ && !debug_image_) {
    ESP_LOGE(TAG, "Debug mode enabled but no debug image available");
    return;
  }
	
  if (!current_image_) {
    ESP_LOGE(TAG, "No image available for processing");
    return;
  }

  if (!image_processor_) {
    ESP_LOGE(TAG, "Image processor not initialized");
    return;
  }
  
  ESP_LOGD(TAG, "Processing image - Configured size: %dx%d, Format: %s", 
           camera_width_, camera_height_,
           pixel_format_.c_str());

  ESP_LOGD(TAG, "Image data length: %zu bytes", current_image_->get_data_length());

  // const auto& zones = crop_zone_handler_.get_zones();
  
  const auto& zones = debug_mode_ ? debug_crop_zones_ : crop_zone_handler_.get_zones();
  
  if (zones.empty()) {
    ESP_LOGW(TAG, "No crop zones defined");
    return;
  }

  // auto processed = image_processor_->process_image(current_image_, zones);
  auto processed = image_processor_->process_image(
    debug_mode_ ? debug_image_ : current_image_,
    zones
  ); 
  
  ESP_LOGD(TAG, "Processed %d image regions", processed.size());
  
  std::vector<float> readings;
  std::vector<float> confidences;

  for (auto& result : processed) {
    float value, confidence;
    if (model_handler_.invoke_model(result.data.get(), result.size, &value, &confidence)) {
      readings.push_back(value);
      confidences.push_back(confidence);
      ESP_LOGD(TAG, "Region result - Value: %.2f, Confidence: %.2f", value, confidence);
    }
  }

  if (!readings.empty()) {
    float final_reading = combine_readings(readings);
    if (value_sensor_) {
      value_sensor_->publish_state(final_reading);
    }
  }
  
  
  // ---  const auto& zones = debug_mode_ ? debug_crop_zones_ : crop_zone_handler_.get_zones();
  
  // auto processed = image_processor_->process_image(
    // debug_mode_ ? debug_image_ : current_image_,
    // zones
  // );

  // ESP_LOGD(TAG, "Processed %d image regions", processed.size());

  // for (auto& result : processed) {
    // float value, confidence;
    // if (model_handler_.invoke_model(result.data.get(), result.size, &value, &confidence)) {
      // ESP_LOGD(TAG, "Model inference result - Value: %.2f, Confidence: %.2f", value, confidence);
      
      // if (confidence >= confidence_threshold_) {
        // ESP_LOGD(TAG, "Confidence threshold met (%.2f >= %.2f)", confidence, confidence_threshold_);
        // if (value_sensor_) {
          // value_sensor_->publish_state(value);
          // ESP_LOGD(TAG, "Published value to sensor: %.2f", value);
        // }
      // } else {
        // ESP_LOGD(TAG, "Confidence below threshold (%.2f < %.2f)", confidence, confidence_threshold_);
      // }
    // } else {
      // ESP_LOGE(TAG, "Model invocation failed for processed image region");
    // }
  // }

  // return_image();
// }
  // ---

  return_image();
}

float MeterReaderTFLite::combine_readings(const std::vector<float> &readings) {
  int combined = 0;
  for (float reading : readings) {
    int digit = static_cast<int>(std::round(reading));
    if (digit == 10) digit = 0; // Handle special case
    combined = combined * 10 + digit;
  }
  return static_cast<float>(combined);
}

bool MeterReaderTFLite::allocate_tensor_arena() {
  tensor_arena_allocation_ = memory_manager_.allocate_tensor_arena(tensor_arena_size_requested_);
  tensor_arena_size_actual_ = tensor_arena_allocation_.actual_size;
  return static_cast<bool>(tensor_arena_allocation_);
}

// bool MeterReaderTFLite::load_model() {
  // if (!allocate_tensor_arena()) {
    // return false;
  // }

  // if (!model_handler_.load_model(model_, model_length_,
                               // tensor_arena_allocation_.data.get(),
                               // tensor_arena_size_actual_)) {
    // return false;
  // }

  // report_memory_status();
  // return true;
// }

void MeterReaderTFLite::report_memory_status() {
  memory_manager_.report_memory_status(
    tensor_arena_size_requested_,
    tensor_arena_size_actual_,
    get_arena_peak_bytes(),
    model_length_
  );
}

size_t MeterReaderTFLite::get_arena_peak_bytes() const {
  return model_handler_.get_arena_peak_bytes();
}

void MeterReaderTFLite::loop() {
  // Nothing here; image capture is async via callback
}




#ifdef DEBUG_METER_READER_TFLITE

void MeterReaderTFLite::set_debug_mode(bool debug_mode) {
    debug_mode_ = debug_mode;
    ESP_LOGI(TAG, "Debug mode %s", debug_mode ? "enabled" : "disabled");
    
    if (debug_mode) {
        // Delay model loading by 10 sec, same as in setup()
        this->set_timeout(10000, [this]() {
            if (!this->load_model()) {
                ESP_LOGE(TAG, "Failed to load model in debug mode");
                return;
            }
            this->model_loaded_ = true;
            
            // Initialize image processor with debug dimensions after model is loaded
            image_processor_ = std::make_unique<ImageProcessor>(
                ImageProcessorConfig{
                    640,  // debug image width
                    480,  // debug image height
                    pixel_format_
                },
                &model_handler_
            );
            ESP_LOGD(TAG, "Debug mode initialized with %zu crop zones", debug_crop_zones_.size());
            
            // Process debug image if available
            if (debug_image_) {
                ESP_LOGI(TAG, "Processing debug image after model load");
                set_image(debug_image_);
                process_image();
            }
        });
    }
}

void MeterReaderTFLite::set_debug_image(const uint8_t* image_data, size_t image_size) {
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[image_size]);
    memcpy(buffer.get(), image_data, image_size);
    
    debug_image_ = std::make_shared<DebugCameraImage>(
        buffer.get(),
        image_size,
        640,  // debug image width
        480,  // debug image height
        pixel_format_
    );
    
    ESP_LOGD(TAG, "Debug image set (%zu bytes)", image_size);
    
    // If in debug mode and model is already loaded, process immediately
    if (debug_mode_ && model_loaded_) {
        ESP_LOGI(TAG, "Processing newly set debug image");
        set_image(debug_image_);
        process_image();
    }
}

void MeterReaderTFLite::test_with_debug_image() {
    if (!debug_mode_) {
        ESP_LOGE(TAG, "Debug mode not enabled");
        return;
    }
    
    if (!debug_image_) {
        ESP_LOGE(TAG, "No debug image available");
        return;
    }
    
    if (!model_loaded_) {
        ESP_LOGI(TAG, "Model not loaded yet - will process after loading");
        return;
    }
    
    ESP_LOGI(TAG, "Processing debug image");
    set_image(debug_image_);
    process_image();
}
#endif

}  // namespace meter_reader_tflite
}  // namespace esphome