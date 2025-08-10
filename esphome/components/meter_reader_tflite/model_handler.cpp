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
  
  
   if (tflite_model_->subgraphs()->Get(0)->operators()->size() == 0) {
        ESP_LOGE(TAG, "Model has no operators!");
        return false;
    }
    
    // Check quantization
    // auto* input_tensor = interpreter_->input_tensor(0);
    // if (input_tensor->quantization.type() == kTfLiteNoQuantization) {
        // ESP_LOGW(TAG, "Model appears unquantized but config expects quantized!");
    // }
    
    // ESP_LOGI(TAG, "Input type: %s", 
            // input_tensor->type == kTfLiteUInt8 ? "uint8 (quantized)" : "float32");
  
  
  
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

    if (!interpreter_ || !input_tensor()) {
        ESP_LOGE(TAG, "Interpreter or input tensor not available");
        return false;
    }

    TfLiteTensor* input = input_tensor();
    
    // Validate input size
    if (input_size != input->bytes) {
        ESP_LOGE(TAG, "Input size mismatch! Model needs %d, got %zu", 
                input->bytes, input_size);
        return false;
    }

    // Handle different input types
    if (input->type == kTfLiteUInt8) {
        // Quantized model processing
        const float input_scale = input->params.scale;
        const int input_zero_point = input->params.zero_point;
        
        ESP_LOGD(TAG, "Quantized input - scale: %.6f, zero_point: %d",
                input_scale, input_zero_point);
        
        memcpy(input->data.uint8, input_data, input_size);
        
        // Debug log first 5 values
        ESP_LOGD(TAG, "First 5 quantized inputs:");
        for (int i = 0; i < 5 && i < input_size; i++) {
            ESP_LOGD(TAG, "  [%d]: %u (%.4f)", i, input->data.uint8[i],
                    (input->data.uint8[i] - input_zero_point) * input_scale);
        }
    } 
    else if (input->type == kTfLiteFloat32) {
        // Float model processing
        if (config_.normalize) {
            // Normalize uint8 [0,255] to float32 [0,1]
            float* dst = input->data.f;
            for (size_t i = 0; i < input_size; i++) {
                dst[i] = static_cast<float>(input_data[i]) / 255.0f;
            }
            
            ESP_LOGD(TAG, "First 5 normalized inputs:");
            for (int i = 0; i < 5 && i < input_size; i++) {
                ESP_LOGD(TAG, "  [%d]: %.4f", i, dst[i]);
            }
        } else {
            // Direct copy (if model expects 0-255 range)
            memcpy(input->data.data, input_data, input_size);
        }
    }

    // Perform inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return false;
    }

    // Handle output
    TfLiteTensor* output = output_tensor();
    if (!output) {
        ESP_LOGE(TAG, "No output tensor");
        return false;
    }

    if (output->type == kTfLiteUInt8) {
        // Prepare dequantized output buffer
        dequantized_output_.resize(output->dims->data[1]);
        const float scale = output->params.scale;
        const int zero_point = output->params.zero_point;
        
        for (int i = 0; i < output->dims->data[1]; i++) {
            dequantized_output_[i] = 
                (output->data.uint8[i] - zero_point) * scale;
        }
        model_output_ = dequantized_output_.data();
    } 
    else {
        model_output_ = output->data.f;
    }

    output_size_ = output->dims->data[1];
    DURATION_END("ModelHandler::invoke_model");
    return true;
}



/*
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

    // Detailed input tensor logging
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
    } 

    // Copy input data
    // std::memcpy(input->data.uint8, input_data, input_size);
	
	
	//const int required_size = input->bytes; // Get required size from model
	
	
	if (input->type == kTfLiteUInt8) {
        // Quantized model - direct copy with quantization params
        const float input_scale = input->params.scale;
        const int input_zero_point = input->params.zero_point;
        
        ESP_LOGD(TAG, "Quant params - scale: %.4f, zero_point: %d", 
                input_scale, input_zero_point);
        
        // Verify exact size match
        if (input_size != input->bytes) {
            ESP_LOGE(TAG, "Input size mismatch! Model needs %d, got %zu", 
                    input->bytes, input_size);
            return false;
        }
        
        // Direct memcpy for quantized data
        memcpy(input->data.uint8, input_data, input_size);
        
        // Debug first 10 values
        ESP_LOGD(TAG, "First 10 quantized input values:");
        for (int i = 0; i < 10 && i < input_size; i++) {
            ESP_LOGD(TAG, "  Input[%d]: %u (float: %.4f)", i, 
                    input->data.uint8[i],
                    (input->data.uint8[i] - input_zero_point) * input_scale);
        }
    }
	
	
	
    ESP_LOGD(TAG, "Input data copied successfully");

    // Perform inference
    TfLiteStatus invoke_status = interpreter_->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Model invocation failed with status: %d", invoke_status);
        return false;
    }
	
	// Log first 10 input values
    // ESP_LOGD(TAG, "Debug : First 10 input values:");
    // for (int i = 0; i < 10 && i < input_size/sizeof(float); i++) {
        // ESP_LOGD(TAG, "  Input[%d]: %.4f", i, reinterpret_cast<const float*>(input_data)[i]);
    // }
	
	
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
*/


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