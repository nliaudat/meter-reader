#include "model_handler.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "ModelHandler";

bool ModelHandler::load_model(const uint8_t *model_data, size_t model_size,
                            uint8_t* tensor_arena, size_t tensor_arena_size) {
  tflite_model_ = tflite::GetModel(model_data);
  if (tflite_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch");
    return false;
  }

  static tflite::MicroMutableOpResolver<MAX_OPERATORS> resolver;
  std::set<tflite::BuiltinOperator> required_ops;
  
  for (size_t i = 0; i < tflite_model_->operator_codes()->size(); ++i) {
    const auto *op_code = tflite_model_->operator_codes()->Get(i);
    required_ops.insert(op_code->builtin_code());
  }

  if (!OpResolverManager::RegisterOps<MAX_OPERATORS>(resolver, required_ops, TAG)) {
    ESP_LOGE(TAG, "Failed to register operators");
    return false;
  }

  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      tflite_model_,
      resolver,
      tensor_arena,
      tensor_arena_size);

  return interpreter_->AllocateTensors() == kTfLiteOk;
}

bool ModelHandler::invoke_model(const uint8_t *input_data, size_t input_size,
                              float* output_value, float* output_confidence) {
  if (!interpreter_) return false;

  TfLiteTensor *input = input_tensor();
  if (!input || input->bytes != input_size) {
    ESP_LOGE(TAG, "Input tensor size mismatch");
    return false;
  }

  memcpy(input->data.data, input_data, input_size);
  
  if (interpreter_->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke failed");
    return false;
  }

  TfLiteTensor *output = output_tensor();
  if (!output || output->bytes < 2 * sizeof(float)) {
    ESP_LOGE(TAG, "Invalid output tensor");
    return false;
  }

  if (output_value) *output_value = output->data.f[0];
  if (output_confidence) *output_confidence = output->data.f[1];
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