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
  
  if (!camera_) {
    ESP_LOGE(TAG, "No camera configured!");
    mark_failed();
    return;
  }
  
  // Initialize camera first
  // if (camera_) {
    // camera_->init(); // Ensure camera is initialized
    // ESP_LOGI(TAG, "Camera initialized");
  // } else {
    // ESP_LOGE(TAG, "No camera configured!");
    // mark_failed();
    // return;
  // }

  ESP_LOGI(TAG, "Camera is ready, delaying model loading...");
  // ESP_LOGI(TAG, "Model loading delayed for 10 seconds ...");
  // Delay model loading by 10 sec
  this->set_timeout(10000, [this]() { //10 sec
    if (!this->load_model()) {
      ESP_LOGE(TAG, "Failed to load model. Marking component as failed.");
      this->mark_failed();
      return;
    }
    this->model_loaded_ = true;
	
/* #ifdef DEBUG_METER_READER_TFLITE
    if (debug_mode_) {
        
        // Override for debug
        camera_width_ = 640;
        camera_height_ = 480;
        
        ESP_LOGD(TAG, "Debug mode - Using fixed image dimensions 640x480");
    }
#endif */
    
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

/* #ifdef DEBUG_METER_READER_TFLITE    
    // Process debug image if in debug mode
    if (debug_mode_ && debug_image_) {
      ESP_LOGI(TAG, "Processing debug image after model load");
      set_image(debug_image_);
      process_full_image();
    }
#endif */
	
  });
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

/* void MeterReaderTFLite::get_camera_image(esp32_camera::ESP32Camera *camera) {
  camera_ = camera;
  camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
	
    if (debug_mode_) {
        ESP_LOGD(TAG, "Debug mode active - ignoring live camera image");
        return;
    }	
	  
    ESP_LOGD(TAG, "New camera image received (%zu bytes)", image->get_data_length());
    
    // Validate image
    if (!image || image->get_data_length() == 0) {
      ESP_LOGE(TAG, "Received invalid image");
      return;
    }

    // Check if we're still processing previous image
    if (this->current_image_) {
      ESP_LOGW(TAG, "Still processing previous image - dropping frame");
      return;
    }

    // Process the new image
    this->set_image(image);
    this->process_full_image();
    
    // Always clean up when done
    this->return_image();
  });
  ESP_LOGD(TAG, "Camera callback registered");
} */


/* void MeterReaderTFLite::get_camera_image(esp32_camera::ESP32Camera *camera) {
    camera_ = camera;
    camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
        // Just store the image, don't process it here. It's done by update() at interval
        if (!debug_mode_) {  // Skip in debug mode
            this->set_image(image);
        }
    });
} */



// void MeterReaderTFLite::get_camera_image(esp32_camera::ESP32Camera *camera) {
  // camera_ = camera;
  // camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
    // this->set_image(image);
    ////this->process_full_image();
  // });
  // ESP_LOGD(TAG, "Camera callback registered");
// }

void MeterReaderTFLite::get_camera_image(esp32_camera::ESP32Camera *camera) {
  camera_ = camera;
  camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
    // Only store image if we're not currently processing and need a new one
    if (!current_image_) {
      current_image_ = image;
    }
    // Else: let the frame drop (camera continues streaming)
  });

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
    ESP_LOGD(TAG, "Image processing complete - resources released");
}

void MeterReaderTFLite::update() {
	
// update() called (every 60s) [meter_reader_tflite:  update_interval: 60s]

	
/* #ifdef DEBUG_METER_READER_TFLITE
  if (this->debug_mode_) {
    ESP_LOGD(TAG, "Skipping update() because debug_mode is active");
    return;
  }
#endif */

    if (debug_mode_) {
        ESP_LOGD(TAG, "Debug mode active - skipping camera update");
        return;
    }

    if (!model_loaded_) {
        ESP_LOGW(TAG, "Model not loaded, skipping update");
        return;
    }

    if (!camera_) {
        ESP_LOGE(TAG, "Camera not configured");
        return;
    }
	
	
	  if (!current_image_) {
		// No image waiting - request a fresh one
		ESP_LOGD(TAG, "Requesting new image from camera async");
		//Non-blocking: request_image(camera::IDLE) is asynchronous and doesn't block the camera flow
		// camera_->request_image(camera::IDLE); // request_image() is asynchronous and doesn't return an image directly - it triggers the callback you set up in get_camera_image()
		camera_->request_image(camera::IDLE);
		return;
	  }
	
	if (current_image_) {
        ESP_LOGD(TAG, "Processing captured image");
		//this->set_timeout(1000, [this]() { // 1-second delay
        process_full_image();
        return_image();  // Clear after processing
    }
	
  
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




void MeterReaderTFLite::process_full_image() {
	
  if (is_processing_image_) {
	  ESP_LOGD(TAG, "process_full_image already running, skipping invoke");
	  return;
  }
    is_processing_image_ = true;
	
  if (!model_loaded_) {
    ESP_LOGE(TAG, "Model not loaded");
    return;
  }

/* #ifdef DEBUG_METER_READER_TFLITE
  if (debug_mode_) {
    if (!debug_image_) {
      ESP_LOGE(TAG, "Debug mode enabled but no debug image available");
      return;
    }
    ESP_LOGD(TAG, "Processing debug image");
    current_image_ = debug_image_; // Use debug image as current image
  }
#endif */

  if (!current_image_) {
    ESP_LOGE(TAG, "No image available");
    return;
  }

	

    std::shared_ptr<camera::CameraImage> image_to_process;
    
    if (debug_mode_) {
        if (!debug_image_) {
            ESP_LOGE(TAG, "Debug mode enabled but no debug image available");
            return;
        }
        image_to_process = debug_image_;
        ESP_LOGD(TAG, "Processing debug image");
    } else {
        if (!current_image_) {
            ESP_LOGE(TAG, "No camera image available");
            return;
        }
        image_to_process = current_image_;
        ESP_LOGD(TAG, "Processing camera image");
    }
	

  if (!image_processor_) {
    ESP_LOGE(TAG, "Image processor not initialized");
    return;
  }
  
  ESP_LOGD(TAG, "Processing image - Configured size: %dx%d, Format: %s", 
           camera_width_, camera_height_,
           pixel_format_.c_str());

  ESP_LOGD(TAG, "Image data length: %zu bytes", current_image_->get_data_length());

/* #ifdef DEBUG_METER_READER_TFLITE

  const auto& zones = debug_mode_ && !debug_crop_zones_.empty() 
                  ? debug_crop_zones_ 
                  : crop_zone_handler_.get_zones();
#else */
	const auto& given_zones = crop_zone_handler_.get_zones(); 
/* #endif	 */
				  
  
  
  std::vector<CropZone> zones_to_use;
    
    if (given_zones.empty()) {
        ESP_LOGW(TAG, "No crop zones defined - processing full image");
        zones_to_use.push_back({0, 0, camera_width_, camera_height_});
    } else {
        zones_to_use = given_zones; // Copy existing zones
    }
  
  
  // if (zones.empty()) {
    // ESP_LOGW(TAG, "No crop zones defined - setting to full image");
	// zones = {{0, camera_width_, 0, camera_height_}};
    //return;
  // }




  auto processed_splited_zones = image_processor_->split_image_in_zone(
    debug_mode_ ? debug_image_ : current_image_,
    zones_to_use
  ); 
  
  ESP_LOGD(TAG, "Splited zones %d processed in image regions", processed_splited_zones.size());
  
  std::vector<float> readings;
  std::vector<float> confidences;

  for (auto& result : processed_splited_zones) {
    float value, confidence;
    if (model_handler_.invoke_model(result.data.get(), result.size, &value, &confidence)) {
      readings.push_back(value);
      confidences.push_back(confidence);
      ESP_LOGD(TAG, "Region result - Value: %.2f, Confidence: %.2f", value, confidence);
    }
  }

  if (!readings.empty()) {
    float final_reading = combine_readings(readings);
	ESP_LOGI(TAG, "DEBUG result: %.2f", final_reading);
    if (value_sensor_) {
      value_sensor_->publish_state(final_reading);
    }
  }
  

  // return_image();
  
  is_processing_image_ = false;
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



/* 
#ifdef DEBUG_METER_READER_TFLITE

void MeterReaderTFLite::set_debug_mode(bool debug_mode) {
    debug_mode_ = debug_mode;
    ESP_LOGI(TAG, "Debug mode %s", debug_mode ? "enabled" : "disabled");
    
    // if (debug_mode) {
        
        // this->set_timeout(10000, [this]() {
            // if (!this->load_model()) {
                // ESP_LOGE(TAG, "Failed to load model in debug mode");
                // return;
            // }
            // this->model_loaded_ = true;
            
            
            // image_processor_ = std::make_unique<ImageProcessor>(
                // ImageProcessorConfig{
                    // 640,  // debug image width
                    // 480,  // debug image height
                    // pixel_format_
                // },
                // &model_handler_
            // );
            // ESP_LOGD(TAG, "Debug mode initialized with %zu crop zones", debug_crop_zones_.size());
            
            
            // if (debug_image_) {
                // ESP_LOGI(TAG, "Processing debug image after model load");
                // set_image(debug_image_);
                // process_full_image();
            // }
        // });
    // }
}

void MeterReaderTFLite::set_debug_image(const uint8_t* image_data, size_t image_size) {
	
	if (!image_data || image_size == 0) {
        ESP_LOGE(TAG, "Invalid debug image data");
        return;
    }
	
	
	//disabled 04.08.2025
    // std::unique_ptr<uint8_t[]> buffer(new uint8_t[image_size]);
    // memcpy(buffer.get(), image_data, image_size);
    
	
    debug_image_ = std::make_shared<DebugCameraImage>(
        buffer.get(),
        image_size,
        640,  // debug image width
        480,  // debug image height
        pixel_format_
    );
    
    ESP_LOGD(TAG, "Debug image set (%zu bytes)", image_size);
	
	ESP_LOGD(TAG, "First 10 bytes of debug image: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
         image_data[0], image_data[1], image_data[2], image_data[3], image_data[4],
         image_data[5], image_data[6], image_data[7], image_data[8], image_data[9]);
   

}

void MeterReaderTFLite::test_with_debug_image() {
    if (!debug_mode_) {
        ESP_LOGE(TAG, "Debug mode not enabled");
        return;
    }
    
    if (!debug_image_) {
        ESP_LOGE(TAG, "DEBUG mode : No debug image available");
        return;
    }
    
    if (!model_loaded_) {
        ESP_LOGI(TAG, "Model not loaded yet - will process after loading");
        return;
    }
    
    ESP_LOGI(TAG, "Processing debug image");
    set_image(debug_image_);
    process_full_image();
}


// void MeterReaderTFLite::print_debug_info() {
    // ESP_LOGI(TAG, "Debug Mode Status:");
    // ESP_LOGI(TAG, "  Image: %s", debug_image_ ? "Loaded" : "Not loaded");
    // ESP_LOGI(TAG, "  Crop Zones: %zu", debug_crop_zones_.size());
    // ESP_LOGI(TAG, "  Model Loaded: %s", model_loaded_ ? "Yes" : "No");
// }


#endif
 */

void MeterReaderTFLite::print_debug_info() {
    ESP_LOGI(TAG, "══════════════ DEBUG INFORMATION ══════════════");
    
    // Core Component Status
    ESP_LOGI(TAG, "┌──────────────────────────────────────────────┐");
    ESP_LOGI(TAG, "│                CORE STATUS                   │");
    ESP_LOGI(TAG, "├──────────────────────────────────────────────┤");
    ESP_LOGI(TAG, "│ Debug Mode:          %-23s │", debug_mode_ ? "ACTIVE" : "INACTIVE");
    ESP_LOGI(TAG, "│ Model Loaded:        %-23s │", model_loaded_ ? "YES" : "NO");
    ESP_LOGI(TAG, "│ Model Type:          %-23s │", model_type_.c_str());
    ESP_LOGI(TAG, "│ Camera Configured:    %-23s │", camera_ ? "YES" : "NO");
    ESP_LOGI(TAG, "│ Sensor Configured:    %-23s │", value_sensor_ ? "YES" : "NO");
    ESP_LOGI(TAG, "└──────────────────────────────────────────────┘");
    
    
    // Model Handler Information
    ESP_LOGI(TAG, "┌──────────────────────────────────────────────┐");
    ESP_LOGI(TAG, "│              MODEL INFORMATION               │");
    ESP_LOGI(TAG, "├──────────────────────────────────────────────┤");
    if (model_loaded_) {
        ESP_LOGI(TAG, "│ Input Dimensions:   %-4dx%-4dx%-12d │",
                model_handler_.get_input_width(),
                model_handler_.get_input_height(),
                model_handler_.get_input_channels());
        ESP_LOGI(TAG, "│ Arena Usage:        %-6zu/%-6zu%-11s │",
                model_handler_.get_arena_peak_bytes(),
                tensor_arena_size_actual_,
                "bytes");
    } else {
        ESP_LOGI(TAG, "│ Model:              %-23s │", "NOT LOADED");
    }
    ESP_LOGI(TAG, "└──────────────────────────────────────────────┘");
    
    // Crop Zone Information
    ESP_LOGI(TAG, "┌──────────────────────────────────────────────┐");
    ESP_LOGI(TAG, "│              CROP ZONE STATUS                │");
    ESP_LOGI(TAG, "├──────────────────────────────────────────────┤");
/* #ifdef DEBUG_METER_READER_TFLITE
  const auto& active_zones = debug_mode_ && !debug_crop_zones_.empty() 
                  ? debug_crop_zones_ 
                  : crop_zone_handler_.get_zones();
#else */
	const auto& active_zones = crop_zone_handler_.get_zones(); 
/* #endif	 */
    ESP_LOGI(TAG, "│ Active Zones:       %-23zu │", active_zones.size());
    
    for (size_t i = 0; i < active_zones.size(); i++) {
        const auto& zone = active_zones[i];
        ESP_LOGI(TAG, "│ Zone %-2d:           [%4d,%-4d,%-4d,%-4d]     │",
                i+1, zone.x1, zone.y1, zone.x2, zone.y2);
    }
    ESP_LOGI(TAG, "└──────────────────────────────────────────────┘");
    
    // Image Processor Information
    ESP_LOGI(TAG, "┌──────────────────────────────────────────────┐");
    ESP_LOGI(TAG, "│            IMAGE PROCESSOR STATUS            │");
    ESP_LOGI(TAG, "├──────────────────────────────────────────────┤");
    if (image_processor_) {
        ESP_LOGI(TAG, "│ Processor:          %-23s │", "INITIALIZED");
        // ESP_LOGI(TAG, "│ Config Dimensions:  %-4dx%-19d │", 
                // camera_width_, camera_height_);
        ESP_LOGI(TAG, "│ Pixel Format:       %-23s │", pixel_format_.c_str());
    } else {
        ESP_LOGI(TAG, "│ Processor:          %-23s │", "NOT INITIALIZED");
    }
    ESP_LOGI(TAG, "└──────────────────────────────────────────────┘");
    
    // Memory Information
    ESP_LOGI(TAG, "┌──────────────────────────────────────────────┐");
    ESP_LOGI(TAG, "│              MEMORY INFORMATION              │");
    ESP_LOGI(TAG, "├──────────────────────────────────────────────┤");
    ESP_LOGI(TAG, "│ Model Size:         %-6zu%-17s │", 
            model_length_, "bytes");
    ESP_LOGI(TAG, "│ Arena Requested:    %-6zu%-17s │", 
            tensor_arena_size_requested_, "bytes");
    ESP_LOGI(TAG, "│ Arena Allocated:    %-6zu%-17s │", 
            tensor_arena_size_actual_, "bytes");
    if (model_loaded_) {
        ESP_LOGI(TAG, "│ Arena Peak Usage:   %-6zu%-17s │", 
                model_handler_.get_arena_peak_bytes(), "bytes");
    }
    ESP_LOGI(TAG, "└──────────────────────────────────────────────┘");
    
    ESP_LOGI(TAG, "═══════════════════════════════════════════════");
}




}  // namespace meter_reader_tflite
}  // namespace esphome