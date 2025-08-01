#include "meter_reader_tflite.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <set>

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";
constexpr size_t MAX_OPERATORS = 90;

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
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

  // Request a new image
  camera_->request_image(camera::IDLE);
}

void MeterReaderTFLite::process_image() {
  if (!current_image_) {
    ESP_LOGE(TAG, "No image available for processing");
    return;
  }

  const uint8_t *data = current_image_->get_data_buffer();
  size_t length = current_image_->get_data_length();

  TfLiteTensor *input_tensor = interpreter_->input(0);
  if (!input_tensor) {
    ESP_LOGE(TAG, "Failed to get input tensor");
    this->return_image();
    return;
  }

  if (input_tensor->bytes != length) {
    ESP_LOGE(TAG, "Input tensor size mismatch (%d vs %d)", input_tensor->bytes, length);
    this->return_image();
    return;
  }

  memcpy(input_tensor->data.data, data, length);

  TfLiteStatus invoke_status = interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke failed");
    this->return_image();
    return;
  }

  TfLiteTensor *output_tensor = interpreter_->output(0);
  if (!output_tensor) {
    ESP_LOGE(TAG, "Failed to get output tensor");
    this->return_image();
    return;
  }

  float meter_value = output_tensor->data.f[0];
  float confidence = output_tensor->data.f[1];

  ESP_LOGD(TAG, "Inference result: value=%.2f, confidence=%.2f", meter_value, confidence);

  if (confidence >= confidence_threshold_ && value_sensor_) {
    value_sensor_->publish_state(meter_value);
  } else {
    ESP_LOGW(TAG, "Low confidence (%.2f < %.2f), skipping update", confidence, confidence_threshold_);
  }

  this->return_image();
}

// void MeterReaderTFLite::on_error(const std::string &error) {
  // ESP_LOGE(TAG, "Camera error: %s", error.c_str());
// }

bool MeterReaderTFLite::load_model() {
  ESP_LOGD(TAG, "load_model: start");
  if (model_ == nullptr || model_length_ == 0) {
    ESP_LOGE(TAG, "No model data available");
    return false;
  }

  ESP_LOGI(TAG, "Loading model (%zu bytes)", model_length_);

  if (heap_caps_get_total_size(MALLOC_CAP_SPIRAM) > 0) {
    ESP_LOGI(TAG, "PSRAM is available.");
  } else {
    ESP_LOGW(TAG, "PSRAM not available. Large tensor arenas may fail to allocate from internal RAM.");
  }

  tflite_model_ = tflite::GetModel(model_);
  if (tflite_model_ == nullptr) {
    ESP_LOGE(TAG, "Failed to get model from buffer. The model data may be corrupt or invalid.");
    return false;
  }

  if (tflite_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch");
    return false;
  }

  if (!allocate_tensor_arena()) {
    return false;
  }

  static tflite::MicroMutableOpResolver<MAX_OPERATORS> resolver;

  const auto *subgraphs = tflite_model_->subgraphs();
  if (subgraphs->size() != 1) {
    ESP_LOGE(TAG, "Only single subgraph models are supported");
    return false;
  }

  std::set<tflite::BuiltinOperator> required_ops;
  for (size_t i = 0; i < tflite_model_->operator_codes()->size(); ++i) {
    const auto *op_code = tflite_model_->operator_codes()->Get(i);
    auto builtin_code = op_code->builtin_code();
    required_ops.insert(builtin_code);
  }

  if (!meter_reader_tflite::OpResolverManager::RegisterOps(resolver, required_ops, TAG)) {
    ESP_LOGE(TAG, "Failed to register all required operators");
    return false;
  }

  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      tflite_model_,
      resolver,
      tensor_arena_.get(),
      tensor_arena_size_actual_);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to allocate tensors. Check logs for details from tflite_micro.");
    return false;
  }

  ESP_LOGI(TAG, "Model loaded successfully");
  report_memory_status();
  return true;
}

bool MeterReaderTFLite::allocate_tensor_arena() {
#ifdef ESP_NN
  ESP_LOGI(TAG, "ESP-NN optimizations are enabled");
#else
  ESP_LOGW(TAG, "ESP-NN not enabled - using default kernels");
#endif

  tensor_arena_size_actual_ = tensor_arena_size_requested_;

  uint8_t *arena_ptr = static_cast<uint8_t *>(heap_caps_malloc(tensor_arena_size_actual_, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (arena_ptr == nullptr) {
    ESP_LOGW(TAG, "Could not allocate tensor arena from PSRAM, trying internal RAM.");
    arena_ptr = static_cast<uint8_t *>(heap_caps_malloc(tensor_arena_size_actual_, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  }

  if (arena_ptr == nullptr) {
    ESP_LOGE(TAG, "Failed to allocate tensor arena from both PSRAM and internal RAM. Try reducing tensor_arena_size.");
    return false;
  }

  tensor_arena_.reset(arena_ptr);
  return true;
}

size_t MeterReaderTFLite::get_arena_peak_bytes() const {
  return interpreter_ ? interpreter_->arena_used_bytes() : 0;
}

void MeterReaderTFLite::report_memory_status() {
  size_t free_internal_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  size_t peak_bytes = this->get_arena_peak_bytes();
  ESP_LOGI(TAG, "Memory Status:");
  ESP_LOGI(TAG, "  Requested Arena: %zuB (%.1fKB)", tensor_arena_size_requested_, tensor_arena_size_requested_ / 1024.0f);
  ESP_LOGI(TAG, "  Allocated Arena: %zuB (%.1fKB)", tensor_arena_size_actual_, tensor_arena_size_actual_ / 1024.0f);
  ESP_LOGI(TAG, "  Arena Peak Usage: %zuB (%.1fKB)", peak_bytes, peak_bytes / 1024.0f);

  size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  if (total_psram > 0) {
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "  PSRAM: %zuB free of %zuB total (%.1fKB / %.1fKB)", free_psram, total_psram, free_psram / 1024.0f, total_psram / 1024.0f);
  }

  ESP_LOGI(TAG, "  Free Internal Heap: %zuB (%.1fKB)", free_internal_heap, free_internal_heap / 1024.0f);

  if (model_length_ > 0) {
    float ratio = static_cast<float>(tensor_arena_size_actual_) / model_length_;
    ESP_LOGI(TAG, "  Arena/Model Ratio: %.1fx", ratio);
  }
}

void MeterReaderTFLite::loop() {
  // Nothing here; image capture is async via callback
}

}  // namespace meter_reader_tflite
}  // namespace esphome
