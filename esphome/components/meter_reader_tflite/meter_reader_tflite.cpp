#include "meter_reader_tflite.h"
#include "esp_log.h"
#include "debug_utils.h"



namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

/*
void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  
  if (camera_width_ == 0 || camera_height_ == 0) {
    ESP_LOGE(TAG, "Camera dimensions not set!");
	// this->print_debug_info();
    mark_failed();
    return;
  }
  
  if (!camera_) {
    ESP_LOGE(TAG, "No camera configured!");
	// this->print_debug_info();
    mark_failed();
    return;
  }

  // Setup camera callback immediately
  setup_camera_callback();

  // delay model loading for 10 sec
  ESP_LOGI(TAG, "Model loading delayed for 10 seconds ...");
  this->set_timeout(10000, [this]() {
    if (!this->load_model()) {
      ESP_LOGE(TAG, "Failed to load model");
	  this->print_debug_info();
      this->mark_failed();
      return;
    }
    this->model_loaded_ = true;
    
	ESP_LOGI(TAG, "Setting up Image Processor ...");
    image_processor_ = std::make_unique<ImageProcessor>(
      ImageProcessorConfig{camera_width_, camera_height_, pixel_format_},
      &model_handler_
    );
	
	ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
	
  // Add debug info after successful initialization
  this->print_debug_info();
	
  });
}

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
	
	    // Create frame queue (holds up to 2 frames)
    frame_queue_ = xQueueCreate(1, sizeof(std::shared_ptr<camera::CameraImage>*));
    if (!frame_queue_) {
        ESP_LOGE(TAG, "Failed to create frame queue!");
        mark_failed();
        return;
    }

    // Setup camera callback immediately
    setup_camera_callback();

    // Delay model loading for 10 seconds with proper timeout handling
    ESP_LOGI(TAG, "Model loading will begin in 10 seconds...");
    

    this->set_timeout(10000, [this]() {
        ESP_LOGI(TAG, "Starting model loading...");
        
        if (!this->load_model()) {
            ESP_LOGE(TAG, "Failed to load model");
            this->print_debug_info();
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
            
            // Start periodic updates now that setup is complete
            this->set_update_interval(this->get_update_interval());
            

    });


}

void MeterReaderTFLite::setup_camera_callback() {
    camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
        // Allocate copy on heap for queue
        auto* frame_copy = new std::shared_ptr<camera::CameraImage>(image);
        
        // Try to send to queue (non-blocking)
        if (xQueueSend(frame_queue_, &frame_copy, 0) != pdTRUE) {
            delete frame_copy;  // Queue full, drop frame
            ESP_LOGW(TAG, "Frame queue full - dropping frame");
        }
    });
}


void MeterReaderTFLite::update() {
	
	// Early exit if system isn't ready
    if (!model_loaded_ || !camera_) {
        ESP_LOGW(TAG, "Update skipped - system not ready (model:%d, camera:%d)", 
                model_loaded_, (camera_ != nullptr));
        return;
    }

    std::shared_ptr<camera::CameraImage>* frame_ptr = nullptr;
    if (xQueueReceive(frame_queue_, &frame_ptr, 0) == pdTRUE) {
        std::shared_ptr<camera::CameraImage> frame = *frame_ptr;
        delete frame_ptr;
        
        DURATION_START();
        process_full_image(frame);
        DURATION_END("process_full_image");
    }
}


/* void MeterReaderTFLite::update() {
    // Early exit if system isn't ready
    if (!model_loaded_ || !camera_) {
        ESP_LOGW(TAG, "Update skipped - system not ready (model:%d, camera:%d)", 
                model_loaded_, (camera_ != nullptr));
        return;
    }

    // Check if we should request a new frame
    bool should_request_frame = false;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!current_frame_ && !frame_requested_) {
            should_request_frame = true;
            frame_requested_ = true;
        }
    }

    // Request new frame if needed (outside mutex lock)
    if (should_request_frame) {
        ESP_LOGD(TAG, "Requesting new frame from camera to process in tflite");
        camera_->request_image(camera::IDLE);
        // Since request_image() returns void, we assume the request was made
        // and rely on the callback to handle frame_requested_ state
        return; // Exit and wait for callback
    }

    // Process existing frame if available
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (current_frame_) {
            // Process outside the lock to minimize mutex hold time
            auto frame = std::move(current_frame_);
            current_frame_.reset();
            
            // Release mutex before processing
            frame_mutex_.unlock();
			ESP_LOGD(TAG, "Processing image");
            process_full_image(frame);
            frame_mutex_.lock();
        }
    }
} */


/*
void MeterReaderTFLite::update() {
    // Early exit if system isn't ready
    if (!model_loaded_ || !camera_) {
        ESP_LOGW(TAG, "Update skipped - system not ready (model:%d, camera:%d)", 
                model_loaded_, (camera_ != nullptr));
        return;
    }

    // Check frame status under mutex
    bool should_request_frame = false;
    bool has_frame_to_process = false;
    std::shared_ptr<camera::CameraImage> frame_to_process;
    
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        
        if (!current_frame_ && !frame_requested_) {
            should_request_frame = true;
            frame_requested_ = true;
        }
        else if (current_frame_) {
            has_frame_to_process = true;
            frame_to_process = std::move(current_frame_);
            current_frame_.reset();
        }
    }

    // Request new frame if needed
    if (should_request_frame) {
        ESP_LOGD(TAG, "Requesting new frame from camera");
        camera_->request_image(camera::IDLE);
        
        // Set timeout for frame arrival
        this->set_timeout(3000, [this]() {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (frame_requested_) {
                ESP_LOGW(TAG, "Frame not received within timeout");
                frame_requested_ = false;
            }
        });
        return;
    }

    // Process frame if available
    if (has_frame_to_process) {
        ESP_LOGD(TAG, "Processing available frame");
        process_full_image(frame_to_process);
    }
    else {
        ESP_LOGD(TAG, "No frame available to process");
    }
}

*/

void MeterReaderTFLite::process_full_image(std::shared_ptr<camera::CameraImage> frame) {
    DURATION_START();
    
    // Validate input frame
    if (!frame || !frame->get_data_buffer() || frame->get_data_length() == 0) {
        ESP_LOGE(TAG, "Invalid frame received for processing");
		this->print_debug_info();
        DURATION_END("process_full_image (invalid frame)");
        return;
    }

    ESP_LOGD(TAG, "Processing frame (%zu bytes)", frame->get_data_length());

    // Get crop zones or use full frame if none configured
    auto zones = crop_zone_handler_.get_zones();
    if (zones.empty()) {
        zones.push_back({0, 0, camera_width_, camera_height_});
        ESP_LOGD(TAG, "No crop zones configured - using full frame");
    }

    // Process each zone
    std::vector<float> readings;
    std::vector<float> confidences;
    bool processing_success = true;

    // Process image through the pipeline
    auto processed_zones = image_processor_->split_image_in_zone(frame, zones);
    
    // Invoke model for each processed zone
    for (auto& result : processed_zones) {
        float value, confidence;
        if (process_model_result(result, &value, &confidence)) {
            readings.push_back(value);
            confidences.push_back(confidence);
            
            ESP_LOGD(TAG, "Zone result - Value: %.1f, Confidence: %.2f", 
                    value, confidence);
        } else {
            ESP_LOGE(TAG, "Model result processing failed for zone");
			this->print_debug_info();
            processing_success = false;
            break;
        }
    }

    // Publish results if successful
    if (processing_success && !readings.empty()) {
        float final_reading = combine_readings(readings);
        float total_confidence = 0.0f;
        for (float conf : confidences) {
            total_confidence += conf;
        }
        float avg_confidence = total_confidence / confidences.size();
        
        ESP_LOGI(TAG, "Final reading: %.1f (avg confidence: %.2f)", 
                final_reading, avg_confidence);

        if (value_sensor_) {
            value_sensor_->publish_state(final_reading);
        }
		
		if (debug_mode_) {
		  this->print_debug_info();
		}
		
		
    } else {
        ESP_LOGE(TAG, "Frame processing failed");
		this->print_debug_info();
    }

    // Memory cleanup
    frame.reset();
    DURATION_END("process_full_image");
}

bool MeterReaderTFLite::process_model_result(const ImageProcessor::ProcessResult& result, float* value, float* confidence) {
    // Invoke the model with the processed image data
    if (!model_handler_.invoke_model(result.data->get(), result.size)) {
        ESP_LOGE(TAG, "Model invocation failed");
		// this->print_debug_info();
        return false;
    }

    // Get the raw model output
    const float* output = model_handler_.get_output();
    if (!output) {
        ESP_LOGE(TAG, "Failed to get model output");
		// this->print_debug_info();
        return false;
    }

    // Find the highest probability class (softmax-like operation)
    float max_prob = 0.0f;
    int predicted_class = 0;
    const int num_classes = model_handler_.get_output_size();
    
    for (int i = 0; i < num_classes; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    // Special handling for digit '0' if needed (class 10 = 0)
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


void MeterReaderTFLite::set_value_sensor(sensor::Sensor *sensor) {
  value_sensor_ = sensor;
  ESP_LOGD(TAG, "Value sensor set");
}

void MeterReaderTFLite::set_crop_zones(const std::string &zones_json) {
  crop_zone_handler_.parse_zones(zones_json);
  ESP_LOGD(TAG, "Crop zones set: %s", zones_json.c_str());
}

void MeterReaderTFLite::set_image(std::shared_ptr<camera::CameraImage> image) {
  current_frame_ = image;
  image_offset_ = 0;
}

size_t MeterReaderTFLite::available() const {
  if (!current_frame_) return 0;
  return current_frame_->get_data_length() - image_offset_;
}

uint8_t *MeterReaderTFLite::peek_data_buffer() {
  if (!current_frame_) return nullptr;
  return current_frame_->get_data_buffer() + image_offset_;
}

void MeterReaderTFLite::consume_data(size_t consumed) {
  if (!current_frame_) return;
  image_offset_ += consumed;
}

void MeterReaderTFLite::return_image() {
    current_frame_.reset();
    image_offset_ = 0;
    ESP_LOGD(TAG, "Image processing complete - resources released");
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
	  ESP_LOGE(TAG, "Tensor arena allocation failed");
	  this->print_debug_info();
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
		ESP_LOGI(TAG, "│ Input Type:         %-23s │", 
                model_handler_.get_config().input_type.c_str());
        ESP_LOGI(TAG, "│ Normalize Input:    %-23s │", 
                model_handler_.get_config().normalize ? "YES" : "NO");
				
		// ESP_LOGI(TAG, "Model input requirements:");
		// ESP_LOGI(TAG, "  Dimensions: %d x %d x %d",
				// input->dims->data[1], 
				// input->dims->data[2],
				// input->dims->data[3]);
		// ESP_LOGI(TAG, "  Type: %s", 
				// input->type == kTfLiteUInt8 ? "uint8" : "float32");
		// ESP_LOGI(TAG, "  Bytes required: %d", input->bytes);
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


///////////// class destructor : 

MeterReaderTFLite::~MeterReaderTFLite() {
    ESP_LOGD(TAG, "Destroying MeterReaderTFLite instance");

    // 1. Release any held camera image
    current_frame_.reset();
    ESP_LOGD(TAG, "Released current camera image");

    // 2. Clean up debug image if exists
    if (debug_image_) {
        debug_image_.reset();
        ESP_LOGD(TAG, "Released debug image");
    }

    // 3. Release image processor resources
    if (image_processor_) {
        image_processor_.reset();
        ESP_LOGD(TAG, "Released image processor");
    }

    // 4. Clean up model handler and interpreter
    if (model_handler_.get_arena_peak_bytes() > 0) {
        ESP_LOGD(TAG, "Releasing model handler resources");
        // This will automatically clean up through unique_ptr
        model_handler_ = ModelHandler(); // Reset to fresh instance
    }

    // 5. Free tensor arena memory
    if (tensor_arena_allocation_) {
        ESP_LOGD(TAG, "Freeing tensor arena (%zu bytes)", 
                tensor_arena_size_actual_);
        // Memory will be freed by the unique_ptr's custom deleter
        tensor_arena_allocation_.data.reset();
    }

    // 6. Release any dynamically allocated buffers
    // if (crop_buffer_) {
        // free(crop_buffer_);
        // crop_buffer_ = nullptr;
        // ESP_LOGD(TAG, "Freed crop buffer");
    // }

    // 7. Clear camera reference
    camera_ = nullptr;
	
  // Clean up double buffers
  // for (auto& buffer : frame_buffers_) {
    // buffer.reset();
  // }

    ESP_LOGD(TAG, "Destruction complete");
}



}  // namespace meter_reader_tflite
}  // namespace esphome