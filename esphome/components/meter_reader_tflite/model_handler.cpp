#include "model_handler.h"
#include "esp_log.h"
#include "debug_utils.h"
#include <cmath>
#include <vector>
#include <algorithm>


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

#ifdef DEBUG_METER_READER_TFLITE
	static tflite::MicroMutableOpResolver<10> resolver;



	TfLiteStatus status = kTfLiteOk;
	if (resolver.AddQuantize() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddMul() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddAdd() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddConv2D() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddMaxPool2D() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddReshape() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddFullyConnected() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddDequantize() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddRelu() != kTfLiteOk) status = kTfLiteError;
	if (resolver.AddSoftmax() != kTfLiteOk) status = kTfLiteError;

	if (status != kTfLiteOk) {
		ESP_LOGE(TAG, "Failed to register one or more operations");
		return false;
	}

	ESP_LOGD(TAG, "All operations registered successfully");

#else
  // Your existing dynamic registration
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
#endif

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

  // Auto-detect model type if output processing not specified
  TfLiteTensor* output = output_tensor();
  if (output && config_.output_processing.empty()) {
    if (output->dims->size >= 2 && output->dims->data[1] == 100) {
      config_.output_processing = "logits_scale10";
      config_.scale_factor = 10.0f;
      ESP_LOGW(TAG, "Auto-detected class100 model, using logits_scale10 processing");
    } else if (output->dims->size >= 2 && output->dims->data[1] == 10) {
      config_.output_processing = "softmax";
      config_.scale_factor = 1.0f;
      ESP_LOGW(TAG, "Auto-detected class10 model, using softmax processing");
    }
  }

  ESP_LOGI(TAG, "Model loaded successfully");
  return true;
}

bool ModelHandler::validate_model_config() {
    auto* input = input_tensor();
    if (!input) return false;

    // Detailed model input information
    ESP_LOGI(TAG, "Model input tensor details:");
    ESP_LOGI(TAG, "  - Type: %s", input->type == kTfLiteUInt8 ? "uint8" : "float32");
    ESP_LOGI(TAG, "  - Bytes: %d", input->bytes);
    ESP_LOGI(TAG, "  - Dimensions: %d", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        ESP_LOGI(TAG, "    - dim[%d]: %d", i, input->dims->data[i]);
    }
    
    // Calculate expected size
    size_t expected_size = 1;
    for (int i = 1; i < input->dims->size; i++) {
        expected_size *= input->dims->data[i];
    }
    ESP_LOGI(TAG, "  - Expected data size: %zu bytes", expected_size);

    // Update config with ACTUAL model dimensions
    if (input->dims->size >= 4) {
        config_.input_size = {input->dims->data[1], input->dims->data[2]};
        config_.input_channels = input->dims->data[3];
    }
    
    return true;
}

ProcessedOutput ModelHandler::process_output(const float* output_data) const {
  const int num_classes = output_size_;
  ProcessedOutput result = {0.0f, 0.0f};
  
  if (num_classes <= 0) {
    ESP_LOGE(TAG, "Invalid number of output classes: %d", num_classes);
    return result;
  }

  // Debug: log output range
  float min_val = *std::min_element(output_data, output_data + num_classes);
  float max_val = *std::max_element(output_data, output_data + num_classes);
  ESP_LOGD(TAG, "Output range: min=%.2f, max=%.2f", min_val, max_val);

  // Find the max value and its index
  int max_idx = 0;
  float max_val_output = output_data[0];
  for (int i = 1; i < num_classes; i++) {
    if (output_data[i] > max_val_output) {
      max_val_output = output_data[i];
      max_idx = i;
    }
  }

  // Process based on output processing method
  if (config_.output_processing == "direct_class") {
    result.value = static_cast<float>(max_idx);
    result.confidence = max_val_output;
    ESP_LOGD(TAG, "Direct class - Value: %.1f, Confidence: %.6f", 
             result.value, result.confidence);
  }
  else if (config_.output_processing == "softmax") {
    // Calculate softmax probabilities
    float sum = 0.0f;
    std::vector<float> exp_values(num_classes);
    
    for (int i = 0; i < num_classes; i++) {
      exp_values[i] = expf(output_data[i]);
      sum += exp_values[i];
    }
    
    // Find the class with highest probability after softmax
    int softmax_max_idx = 0;
    float softmax_max_val = 0.0f;
    for (int i = 0; i < num_classes; i++) {
      float prob = exp_values[i] / sum;
      if (prob > softmax_max_val) {
        softmax_max_val = prob;
        softmax_max_idx = i;
      }
    }
    
    result.value = static_cast<float>(softmax_max_idx);
    result.confidence = softmax_max_val;
    ESP_LOGD(TAG, "Softmax - Value: %.1f, Confidence: %.6f", 
             result.value, result.confidence);
  }
  else if (config_.output_processing == "softmax_scale10") {
    // Calculate softmax probabilities
    float sum = 0.0f;
    std::vector<float> exp_values(num_classes);
    
    for (int i = 0; i < num_classes; i++) {
        exp_values[i] = expf(output_data[i]);
        sum += exp_values[i];
    }
    
    // Find the class with highest probability after softmax
    int softmax_max_idx = 0;
    float softmax_max_val = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        float prob = exp_values[i] / sum;
        if (prob > softmax_max_val) {
            softmax_max_val = prob;
            softmax_max_idx = i;
        }
    }
    
    result.value = static_cast<float>(softmax_max_idx) / config_.scale_factor;  // ÷ 10.0
    result.confidence = softmax_max_val;
    ESP_LOGD(TAG, "Softmax scale10 - Value: %.1f, Confidence: %.6f", 
             result.value, result.confidence);
  }
  else if (config_.output_processing == "logits_scale10") { 
    // Treat raw outputs as logits - just find maximum value
    result.value = static_cast<float>(max_idx) / config_.scale_factor;
    
    // For logits, confidence values can be very large, so we normalize to 0-1 range
    // using a simple sigmoid-like transformation for better interpretability
    float confidence_range = max_val - min_val;
    if (confidence_range > 0) {
      // Normalize to 0-1 range based on distance from min value
      result.confidence = (max_val_output - min_val) / confidence_range;
    } else {
      result.confidence = 1.0f; // All values are the same
    }
    
    ESP_LOGD(TAG, "Logits scale10 - Value: %.1f, Raw Max: %.2f, Confidence: %.6f", 
             result.value, max_val_output, result.confidence);
  }
  else {
    ESP_LOGE(TAG, "Unknown output processing method: %s", 
             config_.output_processing.c_str());
    // Default to direct classification
    result.value = static_cast<float>(max_idx);
    result.confidence = max_val_output;
  }

  return result;
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
        ESP_LOGE(TAG, "Input size mismatch! Expected %d, got %zu", input->bytes, input_size);
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
		// Float model processing - convert uint8 [0,255] to float32
		float* dst = input->data.f;
		if (config_.normalize) {
			// Normalize to [0,1]
			for (size_t i = 0; i < input_size; i++) {
				dst[i] = static_cast<float>(input_data[i]) / 255.0f;
			}
		} else {
			// Keep in [0,255] range (if model expects this)
			for (size_t i = 0; i < input_size; i++) {
				dst[i] = static_cast<float>(input_data[i]);
			}
		}
		
		ESP_LOGD(TAG, "First 5 float32 inputs:");
		for (int i = 0; i < 5 && i < input_size; i++) {
			ESP_LOGD(TAG, "  [%d]: %.4f", i, dst[i]);
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

    // Set output size FIRST 
    output_size_ = output->dims->data[1];

    if (output->type == kTfLiteUInt8) {
        // Prepare dequantized output buffer
        dequantized_output_.resize(output_size_);
        const float scale = output->params.scale;
        const int zero_point = output->params.zero_point;
        
        for (int i = 0; i < output_size_; i++) {
            dequantized_output_[i] = (output->data.uint8[i] - zero_point) * scale;
        }
        model_output_ = dequantized_output_.data();
    } 
    else {
        model_output_ = output->data.f;
    }

    // Log raw output values for debugging
    ESP_LOGD(TAG, "Raw output values (%d classes):", output_size_);
    for (int i = 0; i < output_size_ && i < 15; i++) {
        ESP_LOGD(TAG, "  Output[%d]: %.6f", i, model_output_[i]);
    }
    
    // Process the output to get both value and confidence
    processed_output_ = process_output(model_output_);
    ESP_LOGD(TAG, "Processed output - Value: %.1f, Confidence: %.6f", 
             processed_output_.value, processed_output_.confidence);
    
    DURATION_END("ModelHandler::invoke_model");
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