#include "model_handler.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {
	
static const char *const TAG = "ModelHandler";

bool ModelHandler::load_model(const uint8_t *model_data, size_t model_size,
                            uint8_t* tensor_arena, size_t tensor_arena_size) {
  ESP_LOGD(TAG, "load_model: start");
  ESP_LOGI(TAG, "Loading model (%zu bytes)", model_size);

  tflite_model_ = tflite::GetModel(model_data);
  ESP_LOGD(TAG, "load_model: calling GetModel()");
  
  if (tflite_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch");
    return false;
  }

  static tflite::MicroMutableOpResolver<MAX_OPERATORS> resolver;
  std::set<tflite::BuiltinOperator> required_ops;
  
  ESP_LOGD(TAG, "load_model: parsing operators");
  for (size_t i = 0; i < tflite_model_->operator_codes()->size(); ++i) {
    const auto *op_code = tflite_model_->operator_codes()->Get(i);
    required_ops.insert(op_code->builtin_code());
  }

  if (!OpResolverManager::RegisterOps<MAX_OPERATORS>(resolver, required_ops, TAG)) {
    ESP_LOGE(TAG, "Failed to register operators");
    return false;
  }

  ESP_LOGD(TAG, "load_model: creating interpreter");
  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      tflite_model_,
      resolver,
      tensor_arena,
      tensor_arena_size);

  ESP_LOGD(TAG, "load_model: allocating tensors");
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to allocate tensors");
    return false;
  }

  ESP_LOGI(TAG, "Model loaded successfully");
  return true;
}

bool ModelHandler::invoke_model(const uint8_t *input_data, size_t input_size,
                              float* output_value, float* output_confidence) {

  if (!interpreter_) {
    ESP_LOGE(TAG, "Interpreter not initialized");
    return false;
  }

  TfLiteTensor *input = input_tensor();
  
  if (!input) {
    ESP_LOGE(TAG, "No input tensor available");
    return false;
  }

  ESP_LOGD(TAG, "Input tensor details:");
  ESP_LOGD(TAG, "  - Type: %d", input->type);
  ESP_LOGD(TAG, "  - Bytes: %d", input->bytes);
  ESP_LOGD(TAG, "  - Dimensions: %d", input->dims->size);
  for (int i = 0; i < input->dims->size; i++) {
    ESP_LOGD(TAG, "    - dim[%d]: %d", i, input->dims->data[i]);
  }
  
  if (!input || input->bytes != input_size) {
    ESP_LOGE(TAG, "Input tensor size mismatch (%d != %zu)", input->bytes, input_size);
    return false;
  }

  memcpy(input->data.data, input_data, input_size);
  
  ESP_LOGD(TAG, "Invoking model interpreter");
  if (interpreter_->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG, "Model invocation failed");
    return false;
  }

  TfLiteTensor *output = output_tensor();
  
  if (!output) {
    ESP_LOGE(TAG, "No output tensor available");
    return false;
  }

  ESP_LOGD(TAG, "Output tensor details:");
  ESP_LOGD(TAG, "  - Type: %d", output->type);
  ESP_LOGD(TAG, "  - Bytes: %d", output->bytes);
  ESP_LOGD(TAG, "  - Dimensions: %d", output->dims->size);
  for (int i = 0; i < output->dims->size; i++) {
    ESP_LOGD(TAG, "    - dim[%d]: %d", i, output->dims->data[i]);
  }

  if (output->bytes < 2 * sizeof(float)) {
    ESP_LOGE(TAG, "Output tensor too small (%d < %zu)", output->bytes, 2 * sizeof(float));
    return false;
  }

  if (output_value) {
    *output_value = output->data.f[0];
    ESP_LOGD(TAG, "Output value: %.2f", *output_value);
  }
  if (output_confidence) {
    *output_confidence = output->data.f[1];
    ESP_LOGD(TAG, "Output confidence: %.2f", *output_confidence);
  }
  
  return true;
}

size_t ModelHandler::get_arena_peak_bytes() const {
  return interpreter_ ? interpreter_->arena_used_bytes() : 0;
}

TfLiteTensor* ModelHandler::input_tensor() const {
  return interpreter_ ? interpreter_->input(0) : nullptr;
}

TfLiteTensor* ModelHandler::output_tensor() const {
  return interpreter_ ? interpreter_->output(0) : nullptr;
}

}  // namespace meter_reader_tflite
}  // namespace esphome