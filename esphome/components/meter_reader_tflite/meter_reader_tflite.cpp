#include "meter_reader_tflite.h"
// #include "esphome/components/sdcard/sdcard.h"
// #include "esphome/components/file/file.h"
#include "esp_heap_caps.h"

// #ifdef ESP_NN
// #include "esp-nn.h"
// #endif



namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  
  // if (!load_model()) {
    // mark_failed();
    // return;
  // }

  ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
}

bool MeterReaderTFLite::allocate_tensor_arena() {
	
  // ESP-NN detection at startup
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
             
    // Try 90%, 80%, 70% of requested size
    for (int i = 9; i >= 7; i--) {
      tensor_arena_size_actual_ = (tensor_arena_size_requested_ * i) / 10;
      tensor_arena_ = std::make_unique<uint8_t[]>(tensor_arena_size_actual_);
      if (tensor_arena_) {
        ESP_LOGW(TAG, "Allocated reduced %zuB tensor arena (%d%%)", 
                tensor_arena_size_actual_, i*10);
        break;
      }
    }
    
    // Final fallback to minimum working size
    if (!tensor_arena_) {
      tensor_arena_size_actual_ = 400 * 1024; // Absolute minimum
      tensor_arena_ = std::make_unique<uint8_t[]>(tensor_arena_size_actual_);
      if (!tensor_arena_) {
        ESP_LOGE(TAG, "Critical: Could not allocate minimum 400KB tensor arena");
        return false;
      }
      ESP_LOGE(TAG, "Using minimum %zuB tensor arena - model may not work", 
              tensor_arena_size_actual_);
    }
  }
  
  report_memory_status();
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
  
  if (model_size_ > 0) {
    float ratio = (float)tensor_arena_size_actual_ / model_size_;
    ESP_LOGI(TAG, "  Arena/Model Ratio: %.1fx", ratio);
    if (ratio < 2.0f) {
      ESP_LOGW(TAG, "  Warning: Arena size is less than 2x model size");
    }
  }
}

/*

bool MeterReaderTFLite::load_model() {
  // Load model from SD card
  auto file = File.open(model_path_.c_str(), "r");
  if (!file) {
    ESP_LOGE(TAG, "Failed to open model file: %s", model_path_.c_str());
    return false;
  }

  model_size_ = file.size();
  std::vector<uint8_t> model_data(model_size_);
  if (file.read(model_data.data(), model_size_) != model_size_) {
    ESP_LOGE(TAG, "Failed to read model file");
    file.close();
    return false;
  }
  file.close();

  // Dynamic adjustment recommendation
  size_t recommended_size = model_size_ * 2;
  if (tensor_arena_size_requested_ < recommended_size) {
    ESP_LOGW(TAG, "Recommended arena size is %zuB (2x model size)", recommended_size);
  }

  if (!allocate_tensor_arena()) {
    return false;
  }

  // Load TFLite model
  model_ = tflite::GetModel(model_data.data());
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch");
    return false;
  }

  // Setup interpreter
  static tflite::AllOpsResolver resolver;
  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      model_, resolver, tensor_arena_.get(), tensor_arena_size_actual_);

  // Allocate tensors
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to allocate tensors");
    return false;
  }

  ESP_LOGI(TAG, "Model loaded successfully");
  return true;
}

void MeterReaderTFLite::preprocess_image(uint8_t* image_data) {
  TfLiteTensor* input = interpreter_->input(0);
  
  // Example preprocessing - adjust based on your model requirements
  for (int i = 0; i < input_width_ * input_height_; i++) {
    // Normalize to [-1, 1] range
    input->data.f[i] = (image_data[i] / 127.5f) - 1.0f;
  }
} 

void MeterReaderTFLite::process_output(float* output_data) {
  // Example digit classification - adjust for your model
  int predicted_digit = 0;
  float max_confidence = 0.0f;
  
  for (int i = 0; i < 10; i++) {  // Assuming 10 digit classes
    if (output_data[i] > max_confidence) {
      max_confidence = output_data[i];
      predicted_digit = i;
    }
  }

  if (max_confidence >= confidence_threshold_) {
    ESP_LOGI(TAG, "Detected digit: %d (confidence: %.2f)", predicted_digit, max_confidence);
  } else {
    ESP_LOGW(TAG, "Low confidence detection: %d (%.2f)", predicted_digit, max_confidence);
  }
}

bool MeterReaderTFLite::run_inference() {
  // Get camera image (implement based on your camera setup)
  uint8_t* image_data = get_camera_image();
  if (!image_data) {
    ESP_LOGE(TAG, "Failed to get camera image");
    return false;
  }

  preprocess_image(image_data);

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG, "Inference failed");
    return false;
  }

  // Process results
  TfLiteTensor* output = interpreter_->output(0);
  process_output(output->data.f);

  return true;
}

*/

void MeterReaderTFLite::loop() {
  // Run inference periodically or based on triggers
  // static uint32_t last_run = 0;
  // uint32_t now = millis();
  
  // if (now - last_run > 5000) {  // Run every 5 seconds
    // if (run_inference()) {
      // ESP_LOGD(TAG, "Inference completed successfully");
    // }
    // last_run = now;
  // }
}



}  // namespace meter_reader_tflite
}  // namespace esphome