#include "meter_reader_tflite.h"
#include "esp_heap_caps.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  
  if (!load_model()) {
    mark_failed();
    return;
  }

  ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
}

bool MeterReaderTFLite::load_model() {
  if (model_ == nullptr || model_length_ == 0) {
    ESP_LOGE(TAG, "No model data available");
    return false;
  }

  ESP_LOGI(TAG, "Loading model (%zu bytes)", model_length_);
  
  tflite_model_ = tflite::GetModel(model_);
  if (tflite_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch");
    return false;
  }

  if (!allocate_tensor_arena()) {
    return false;
  }

  // Create resolver with all required ops
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddQuantize();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddDequantize();

  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      tflite_model_, resolver, tensor_arena_.get(), tensor_arena_size_actual_);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to allocate tensors");
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
  
  // Try requested size first
  tensor_arena_size_actual_ = tensor_arena_size_requested_;
  tensor_arena_ = std::make_unique<uint8_t[]>(tensor_arena_size_actual_);
  
  // Fallback strategy if allocation fails
  if (!tensor_arena_) {
    ESP_LOGW(TAG, "Failed to allocate %zuB tensor arena, attempting fallback...", 
             tensor_arena_size_requested_);
             
    for (int i = 9; i >= 7; i--) {
      tensor_arena_size_actual_ = (tensor_arena_size_requested_ * i) / 10;
      tensor_arena_ = std::make_unique<uint8_t[]>(tensor_arena_size_actual_);
      if (tensor_arena_) {
        ESP_LOGW(TAG, "Allocated reduced %zuB tensor arena (%d%%)", 
                tensor_arena_size_actual_, i*10);
        break;
      }
    }
    
    if (!tensor_arena_) {
      tensor_arena_size_actual_ = 400 * 1024;
      tensor_arena_ = std::make_unique<uint8_t[]>(tensor_arena_size_actual_);
      if (!tensor_arena_) {
        ESP_LOGE(TAG, "Critical: Could not allocate minimum 400KB tensor arena");
        return false;
      }
      ESP_LOGE(TAG, "Using minimum %zuB tensor arena - model may not work", 
              tensor_arena_size_actual_);
    }
  }
  
  return true;
}

void MeterReaderTFLite::report_memory_status() {
  size_t free_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  ESP_LOGI(TAG, "Memory Status:");
  ESP_LOGI(TAG, "  Requested Arena: %zuB (%.1fKB)", 
          tensor_arena_size_requested_, tensor_arena_size_requested_/1024.0);
  ESP_LOGI(TAG, "  Allocated Arena: %zuB (%.1fKB)", 
          tensor_arena_size_actual_, tensor_arena_size_actual_/1024.0);
  ESP_LOGI(TAG, "  Free Heap: %zuB (%.1fKB)", free_heap, free_heap/1024.0);
  
  if (model_length_ > 0) {
    float ratio = (float)tensor_arena_size_actual_ / model_length_;
    ESP_LOGI(TAG, "  Arena/Model Ratio: %.1fx", ratio);
    if (ratio < 2.0f) {
      ESP_LOGW(TAG, "  Warning: Arena size is less than 2x model size");
    }
  }
}

size_t MeterReaderTFLite::get_arena_used_bytes() const {
    if (interpreter_ == nullptr) {
        return 0;
    }
    return interpreter_->arena_used_bytes();
}

size_t MeterReaderTFLite::get_arena_peak_bytes() const {
    return interpreter_ ? interpreter_->arena_used_bytes() : 0;
}

void MeterReaderTFLite::loop() {
  // Run inference periodically or based on triggers
}

}  // namespace meter_reader_tflite
}  // namespace esphome