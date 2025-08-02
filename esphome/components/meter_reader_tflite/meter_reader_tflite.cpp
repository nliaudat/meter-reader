#include "meter_reader_tflite.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  
  // Verify all required parameters are set
  if (camera_width_ == 0 || camera_height_ == 0) {
    ESP_LOGE(TAG, "Camera dimensions not set!");
    mark_failed();
    return;
  }

  if (model_input_width_ == 0 || model_input_height_ == 0) {
    ESP_LOGE(TAG, "Model input dimensions not set!");
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
}

void MeterReaderTFLite::set_input_size(int width, int height) {
  model_input_width_ = width;
  model_input_height_ = height;
  if (image_processor_ && camera_width_ > 0 && camera_height_ > 0) {
    image_processor_ = std::make_unique<ImageProcessor>(ImageProcessorConfig{
      camera_width_,
      camera_height_,
      model_input_width_,
      model_input_height_,
      pixel_format_
    });
  }
}

void MeterReaderTFLite::set_camera_format(int width, int height, const std::string &pixel_format) {
  camera_width_ = width;
  camera_height_ = height;
  pixel_format_ = pixel_format;
  if (model_input_width_ > 0 && model_input_height_ > 0) {
    image_processor_ = std::make_unique<ImageProcessor>(ImageProcessorConfig{
      camera_width_,
      camera_height_,
      model_input_width_,
      model_input_height_,
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
  if (!current_image_ || !image_processor_) {
    ESP_LOGE(TAG, "No image available or processor not initialized");
    return;
  }

  auto processed = image_processor_->process_image(
    current_image_,
    crop_zone_handler_.get_zones()
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

}  // namespace meter_reader_tflite
}  // namespace esphome