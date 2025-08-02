#include "meter_reader_tflite.h"
// #include "esp_heap_caps.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  this->set_timeout(1000, [this]() {
    ESP_LOGD(TAG, "load_model: start");
    if (!this->load_model()) {
      ESP_LOGE(TAG, "Failed to load model. Marking component as failed.");
      this->mark_failed();
      return;
    }
    this->model_loaded_ = true;
    ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
  });
}

void MeterReaderTFLite::set_camera(esp32_camera::ESP32Camera *camera) {
  this->camera_ = camera;
  this->camera_->add_image_callback([this](std::shared_ptr<camera::CameraImage> image) {
    this->set_image(image);
    this->process_image();
  });
}

void MeterReaderTFLite::set_image(std::shared_ptr<camera::CameraImage> image) {
  this->current_image_ = image;
  this->image_offset_ = 0;
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
  ESP_LOGD(TAG, "Update called");

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

void MeterReaderTFLite::set_camera_format(int width, int height, const std::string &pixel_format) {
  this->camera_width_ = width;
  this->camera_height_ = height;
  this->pixel_format_ = pixel_format;
  this->image_processor_ = ImageProcessor({
    this->camera_width_,
    this->camera_height_,
    this->model_input_width_,
    this->model_input_height_,
    this->pixel_format_
  });
}

void MeterReaderTFLite::process_image() {
  if (!this->current_image_) {
    ESP_LOGE(TAG, "No image available");
    return;
  }

  auto processed = this->image_processor_.process_image(
    this->current_image_,
    this->crop_zone_handler_.get_zones()
  );

  for (auto& result : processed) {
    float value, confidence;
    if (model_handler_.invoke_model(result.data.get(), result.size, &value, &confidence)) {
      if (confidence >= confidence_threshold_ && value_sensor_) {
        value_sensor_->publish_state(value);
      }
    }
  }

  return_image();
}


// void MeterReaderTFLite::process_single_image(std::shared_ptr<camera::CameraImage> image) {
  // const uint8_t *data = image->get_data_buffer();
  // size_t length = image->get_data_length();

  // float value, confidence;
  // if (model_handler_.invoke_model(data, length, &value, &confidence)) {
    // ESP_LOGD(TAG, "Inference result: value=%.2f, confidence=%.2f", value, confidence);
    
    // if (confidence >= confidence_threshold_ && value_sensor_) {
      // value_sensor_->publish_state(value);
    // }
  // }
// }


bool MeterReaderTFLite::allocate_tensor_arena() {
  this->tensor_arena_allocation_ = this->memory_manager_.allocate_tensor_arena(tensor_arena_size_requested_);
  this->tensor_arena_size_actual_ = this->tensor_arena_allocation_.actual_size;
  return static_cast<bool>(this->tensor_arena_allocation_);
}

bool MeterReaderTFLite::load_model() {
  ESP_LOGD(TAG, "load_model: allocating tensor arena");
  if (!allocate_tensor_arena()) {
    return false;
  }

#ifdef ESP_NN
  ESP_LOGI(TAG, "ESP-NN optimizations are enabled");
#else
  ESP_LOGW(TAG, "ESP-NN not enabled - using default kernels");
#endif

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
    this->get_arena_peak_bytes(),
    model_length_
  );
}

size_t MeterReaderTFLite::get_arena_peak_bytes() const {
  return model_handler_.get_arena_peak_bytes();
}

void MeterReaderTFLite::loop() {
  // Nothing here; image capture is async via callback
}

}  // namespace meter_reader_tflite
}  // namespace esphome