#include "meter_reader_tflite.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  ESP_LOGD(TAG, "Initial configuration:");
  
  ESP_LOGD(TAG, "  Camera: %dx%d, Format: %s", 
           camera_width_, camera_height_, pixel_format_.c_str());
  ESP_LOGD(TAG, "  Confidence Threshold: %.2f", confidence_threshold_);
  ESP_LOGD(TAG, "  Tensor Arena Size: %zu bytes", tensor_arena_size_requested_);
  
  // Verify all required parameters are set
  if (camera_width_ == 0 || camera_height_ == 0) {
    ESP_LOGE(TAG, "Camera dimensions not set!");
    mark_failed();
    return;
  }

  // Image processor will be initialized when all parameters are available
  this->set_timeout(1000, [this]() {
    if (!this->load_model()) {
      ESP_LOGE(TAG, "Failed to load model. Marking component as failed.");
      this->mark_failed();
      return;
    }
    this->model_loaded_ = true;
    ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
  });
  
  ESP_LOGD(TAG, "Model input dimensions: %dx%dx%d",
           model_handler_.get_input_width(),
           model_handler_.get_input_height(),
           model_handler_.get_input_channels());
  
  // Initialize image processor with model dimensions
  image_processor_ = std::make_unique<ImageProcessor>(ImageProcessorConfig{
    camera_width_,
    camera_height_,
    model_handler_.get_input_width(),
    model_handler_.get_input_height(),
    pixel_format_
  });
}

void MeterReaderTFLite::set_input_size(int width, int height) {
  if (image_processor_ && camera_width_ > 0 && camera_height_ > 0) {
    image_processor_ = std::make_unique<ImageProcessor>(ImageProcessorConfig{
      camera_width_,
      camera_height_,
      width,
      height,
      pixel_format_
    });
  }
}

void MeterReaderTFLite::set_camera_format(int width, int height, const std::string &pixel_format) {
  camera_width_ = width;
  camera_height_ = height;
  pixel_format_ = pixel_format;
  if (image_processor_) {
    image_processor_ = std::make_unique<ImageProcessor>(ImageProcessorConfig{
      camera_width_,
      camera_height_,
      model_handler_.get_input_width(),
      model_handler_.get_input_height(),
      pixel_format_
    });
  }
}

void MeterReaderTFLite::set_camera(esp32_camera::ESP32Camera *camera) {
  camera_ = camera;
  camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
    this->set_image(image);
    this->process_image();
  });
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

  camera_->request_image(camera::IDLE);
}

void MeterReaderTFLite::set_confidence_threshold(float threshold) {
  confidence_threshold_ = threshold;
}

void MeterReaderTFLite::set_tensor_arena_size(size_t size_bytes) {
  tensor_arena_size_requested_ = size_bytes;
}

void MeterReaderTFLite::set_model(const uint8_t *model, size_t length) {
  model_ = model;
  model_length_ = length;
}

void MeterReaderTFLite::set_value_sensor(sensor::Sensor *sensor) {
  value_sensor_ = sensor;
}

void MeterReaderTFLite::set_crop_zones(const std::string &zones_json) {
  crop_zone_handler_.parse_zones(zones_json);
}

void MeterReaderTFLite::process_image() {
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

  auto processed = image_processor_->process_image(
    current_image_,
    crop_zone_handler_.get_zones()
  );

  ESP_LOGD(TAG, "Processed %d image regions", processed.size());

  for (auto& result : processed) {
    float value, confidence;
    if (model_handler_.invoke_model(result.data.get(), result.size, &value, &confidence)) {
      ESP_LOGD(TAG, "Model inference result - Value: %.2f, Confidence: %.2f", value, confidence);
      
      if (confidence >= confidence_threshold_) {
        ESP_LOGD(TAG, "Confidence threshold met (%.2f >= %.2f)", confidence, confidence_threshold_);
        if (value_sensor_) {
          value_sensor_->publish_state(value);
          ESP_LOGD(TAG, "Published value to sensor: %.2f", value);
        }
      } else {
        ESP_LOGD(TAG, "Confidence below threshold (%.2f < %.2f)", confidence, confidence_threshold_);
      }
    } else {
      ESP_LOGE(TAG, "Model invocation failed for processed image region");
    }
  }

  return_image();
}

bool MeterReaderTFLite::allocate_tensor_arena() {
  tensor_arena_allocation_ = memory_manager_.allocate_tensor_arena(tensor_arena_size_requested_);
  tensor_arena_size_actual_ = tensor_arena_allocation_.actual_size;
  return static_cast<bool>(tensor_arena_allocation_);
}

bool MeterReaderTFLite::load_model() {
  if (!allocate_tensor_arena()) {
    return false;
  }

  if (!model_handler_.load_model(model_, model_length_,
                               tensor_arena_allocation_.data.get(),
                               tensor_arena_size_actual_)) {
    return false;
  }

  report_memory_status();
  return true;
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


// #ifdef DEBUG_METER_READER_TFLITE
// void MeterReaderTFLite::set_debug_image(std::shared_ptr<camera::CameraImage> image) {
  // debug_image_ = image;
  // ESP_LOGD(TAG, "Debug image set (%zu bytes)", image->get_data_length());
// }

// bool MeterReaderTFLite::load_debug_image() {
  // if (!debug_image_) {
    // ESP_LOGE(TAG, "No debug image available");
    // return false;
  // }
  
  // ESP_LOGI(TAG, "Using debug image (%zu bytes)", debug_image_->get_data_length());
  // return true;
// }
// #endif


#ifdef DEBUG_METER_READER_TFLITE
void MeterReaderTFLite::set_debug_image(const uint8_t* image_data, size_t image_size) {
  // Copy data to RAM (remove if using PROGMEM)
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[image_size]);
  memcpy(buffer.get(), image_data, image_size);
  
  debug_image_ = std::make_shared<DebugCameraImage>(
    buffer.get(),
    image_size,
    camera_width_,
    camera_height_,
    pixel_format_
  );
  
  ESP_LOGD(TAG, "Debug image set (%zu bytes)", image_size);
}

void MeterReaderTFLite::test_with_debug_image() {
  if (!debug_image_) {
    ESP_LOGE(TAG, "No debug image available");
    return;
  }
  
  ESP_LOGI(TAG, "Processing debug image");
  set_image(debug_image_);
  process_image();
}
#endif

}  // namespace meter_reader_tflite
}  // namespace esphome