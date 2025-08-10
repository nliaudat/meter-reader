#include "model_handler.h"
#include "esp_log.h"
#include "debug_utils.h"
#include <cmath>

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "ModelHandler";

bool ModelHandler::load_model(const uint8_t *model_data, size_t model_size,
                            uint8_t* tensor_arena, size_t tensor_arena_size,
                            const ModelConfig &config) {
  ESP_LOGD(TAG, "Loading model with config:");
  ESP_LOGD(TAG, "  Description: %s", config.description.c_str());
  ESP_LOGD(TAG, "  Output processing: %s", config.output_processing.c_str());
  
  config_ = config;
  
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


  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to allocate tensors");
    return false;
  }
  
  auto* input = input_tensor();
  if (input) {
    ESP_LOGI(TAG, "Input tensor dimensions:");
    for (int i = 0; i < input->dims->size; i++) {
      ESP_LOGI(TAG, "  Dim %d: %d", i, input->dims->data[i]);
    }
  }

  if (!validate_model_config()) {
    ESP_LOGE(TAG, "Model configuration validation failed");
    return false;
  }

  ESP_LOGI(TAG, "Model loaded successfully");
  return true;
}

bool ModelHandler::validate_model_config() {
  auto input_shape = input_tensor()->dims;
  if (input_shape->size != 4) {
    ESP_LOGE(TAG, "Unsupported input shape dimension: %d", input_shape->size);
    return false;
  }

  int model_channels = input_shape->data[3];
  if (model_channels != config_.input_channels) {
    ESP_LOGW(TAG, "Model config input_channels %d doesn't match actual model channels %d",
             config_.input_channels, model_channels);
    config_.input_channels = model_channels;
  }

  if (config_.input_size.first == 0 || config_.input_size.second == 0) {
    config_.input_size = {input_shape->data[1], input_shape->data[2]};
  } else if (input_shape->data[1] != config_.input_size.first || 
             input_shape->data[2] != config_.input_size.second) {
    ESP_LOGW(TAG, "Model config input_size (%d,%d) doesn't match actual model input size (%d,%d)",
             config_.input_size.first, config_.input_size.second,
             input_shape->data[1], input_shape->data[2]);
    config_.input_size = {input_shape->data[1], input_shape->data[2]};
  }

  return true;
}

float ModelHandler::process_output(const float* output_data) const {
  if (config_.output_processing == "direct_class") {
    // Find max probability (simple argmax)
    int max_idx = 0;
    float max_val = output_data[0];
    for (int i = 1; i < 10; i++) {
      if (output_data[i] > max_val) {
        max_val = output_data[i];
        max_idx = i;
      }
    }
    return static_cast<float>(max_idx);
  }
  else if (config_.output_processing == "softmax") {
    // Softmax without scaling
    float sum = 0.0f;
    float exp_values[10];
    for (int i = 0; i < 10; i++) {
      exp_values[i] = expf(output_data[i]);
      sum += exp_values[i];
    }
    
    int max_idx = 0;
    float max_val = 0.0f;
    for (int i = 0; i < 10; i++) {
      float prob = exp_values[i] / sum;
      if (prob > max_val) {
        max_val = prob;
        max_idx = i;
      }
    }
    return static_cast<float>(max_idx);
  }
  else { // "softmax_scale10"
    // Original implementation with 10x scaling
    float sum = 0.0f;
    float exp_values[10];
    for (int i = 0; i < 10; i++) {
      exp_values[i] = expf(output_data[i]);
      sum += exp_values[i];
    }
    
    int max_idx = 0;
    float max_val = 0.0f;
    for (int i = 0; i < 10; i++) {
      float prob = exp_values[i] / sum;
      if (prob > max_val) {
        max_val = prob;
        max_idx = i;
      }
    }
    return static_cast<float>(max_idx) / config_.scale_factor;
  }
}


bool ModelHandler::invoke_model(const uint8_t* input_data, size_t input_size) {
	
    DURATION_START();

    if (!interpreter_) {
        ESP_LOGE(TAG, "Interpreter not initialized");
        return false;
    }

    TfLiteTensor* input = input_tensor();
    if (!input) {
        ESP_LOGE(TAG, "No input tensor available");
        return false;
    }

/*     // Detailed input tensor logging
    ESP_LOGD(TAG, "Input tensor details:");
    ESP_LOGD(TAG, "  - Type: %d", input->type);
    ESP_LOGD(TAG, "  - Bytes: %d", input->bytes);
    ESP_LOGD(TAG, "  - Dimensions: %d", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        ESP_LOGD(TAG, "    - dim[%d]: %d", i, input->dims->data[i]);
    }

    ESP_LOGD(TAG, "Model expects input size: %d bytes (dims: %dx%dx%d)", 
            input->bytes, 
            input->dims->data[1], 
            input->dims->data[2],
            input->dims->data[3]);
    ESP_LOGD(TAG, "Provided input size: %zu bytes", input_size);

    // Validate input size
    if (input->bytes != input_size) {
        ESP_LOGE(TAG, "Input size mismatch! Model expects %d bytes, got %zu bytes",
                input->bytes, input_size);
        return false;
    } */

    // Copy input data
    // std::memcpy(input->data.uint8, input_data, input_size);
	
	
	// conversion for float32 models
    if (input->type == kTfLiteFloat32) {
        float* dst = input->data.f;
        const uint8_t* src = input_data;
        size_t elements = input_size / sizeof(uint8_t);
        
        // Normalize uint8 [0,255] to float32 [0,1]
        for (size_t i = 0; i < elements; i++) {
            dst[i] = static_cast<float>(src[i]) / 255.0f;
        }
    } 
    else {
        // For quantized models (uint8)
        std::memcpy(input->data.uint8, input_data, input_size);
    }
	
	ESP_LOGD(TAG, "First 5 normalized input values:");
	for (int i = 0; i < 5; i++) {
		ESP_LOGD(TAG, "  Input[%d]: %.4f", i, input->data.f[i]);
	}
	
	
    ESP_LOGD(TAG, "Input data copied successfully");

    // Perform inference
    TfLiteStatus invoke_status = interpreter_->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Model invocation failed with status: %d", invoke_status);
        return false;
    }
	
	// Log first 10 input values
    ESP_LOGD(TAG, "Debug : First 10 input values:");
    for (int i = 0; i < 10 && i < input_size/sizeof(float); i++) {
        ESP_LOGD(TAG, "  Input[%d]: %.4f", i, reinterpret_cast<const float*>(input_data)[i]);
    }
	
	
    ESP_LOGD(TAG, "Inference completed successfully");

    // Get output tensor
    TfLiteTensor* output = output_tensor();
    if (!output) {
        ESP_LOGE(TAG, "No output tensor available");
        return false;
    }

    // Detailed output tensor logging
    ESP_LOGD(TAG, "Output tensor details:");
    ESP_LOGD(TAG, "  - Type: %d", output->type);
    ESP_LOGD(TAG, "  - Bytes: %d", output->bytes);
    ESP_LOGD(TAG, "  - Dimensions: %d", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        ESP_LOGD(TAG, "    - dim[%d]: %d", i, output->dims->data[i]);
    }

    // Store output references
    model_output_ = output->data.f;
    output_size_ = output->dims->data[1]; // Assuming [1, N] shape
    
    ESP_LOGD(TAG, "Model output stored - %d classes available", output_size_);
    DURATION_END("ModelHandler::invoke_model");

    return true;
}

// const float* ModelHandler::get_output() const {
    // return model_output_;
// }

// int ModelHandler::get_output_size() const {
    // return output_size_;
// }

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