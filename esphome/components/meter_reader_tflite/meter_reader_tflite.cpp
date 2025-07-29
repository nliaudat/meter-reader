#include "meter_reader_tflite.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

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

  // Create resolver with automatic operation detection
  static tflite::MicroMutableOpResolver<10> resolver; // Adjust size based on your model's ops
  
  // Get model subgraph and operators
  const auto* subgraphs = tflite_model_->subgraphs();
  if (subgraphs->size() != 1) {
    ESP_LOGE(TAG, "Only single subgraph models are supported");
    return false;
  }
  
  const auto* ops = (*subgraphs)[0]->operators();
  const auto* opcodes = tflite_model_->operator_codes();
  
  // Add required operations to the resolver
  for (size_t i = 0; i < ops->size(); i++) {
    const auto* op = (*ops)[i];
    const auto* opcode = (*opcodes)[op->opcode_index()];
    const auto builtin_code = opcode->builtin_code();
    
    switch (builtin_code) {
      case tflite::BuiltinOperator_CONV_2D:
        resolver.AddConv2D();
        break;
      case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        resolver.AddDepthwiseConv2D();
        break;
      case tflite::BuiltinOperator_FULLY_CONNECTED:
        resolver.AddFullyConnected();
        break;
      case tflite::BuiltinOperator_SOFTMAX:
        resolver.AddSoftmax();
        break;
      case tflite::BuiltinOperator_RESHAPE:
        resolver.AddReshape();
        break;
      case tflite::BuiltinOperator_QUANTIZE:
        resolver.AddQuantize();
        break;
      case tflite::BuiltinOperator_DEQUANTIZE:
        resolver.AddDequantize();
        break;
      default:
        ESP_LOGE(TAG, "Unsupported operator: %d", builtin_code);
        return false;
    }
  }

  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      tflite_model_,
      resolver,
      tensor_arena_.get(),
      tensor_arena_size_actual_);

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
  
  if (!tensor_arena_) {
    ESP_LOGE(TAG, "Failed to allocate tensor arena");
    return false;
  }
  
  return true;
}

void MeterReaderTFLite::report_memory_status() {
  size_t free_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  ESP_LOGI(TAG, "Memory Status:");
  ESP_LOGI(TAG, "  Requested Arena: %zuB (%.1fKB)", 
          tensor_arena_size_requested_, tensor_arena_size_requested_/1024.0f);
  ESP_LOGI(TAG, "  Allocated Arena: %zuB (%.1fKB)", 
          tensor_arena_size_actual_, tensor_arena_size_actual_/1024.0f);
  ESP_LOGI(TAG, "  Free Heap: %zuB (%.1fKB)", free_heap, free_heap/1024.0f);
  
  if (model_length_ > 0) {
    float ratio = static_cast<float>(tensor_arena_size_actual_) / model_length_;
    ESP_LOGI(TAG, "  Arena/Model Ratio: %.1fx", ratio);
  }
}

void MeterReaderTFLite::loop() {
  // Your inference logic here
}

}  // namespace meter_reader_tflite
}  // namespace esphome