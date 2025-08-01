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

void MeterReaderTFLite::set_crop_zones(const std::string &zones_json) {
  this->parse_crop_zones(zones_json);
}

void MeterReaderTFLite::parse_crop_zones(const std::string &zones_json) {
  crop_zones_.clear();
  
  // Simple parser for the specific format: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
  std::string stripped = zones_json;
  stripped.erase(std::remove(stripped.begin(), stripped.end(), ' '), stripped.end());
  stripped.erase(std::remove(stripped.begin(), stripped.end(), '\n'), stripped.end());
  stripped.erase(std::remove(stripped.begin(), stripped.end(), '\r'), stripped.end());
  stripped.erase(std::remove(stripped.begin(), stripped.end(), '\t'), stripped.end());

  if (stripped.empty() || stripped == "[]") {
    ESP_LOGD(TAG, "No crop zones defined");
    return;
  }

  if (stripped.front() == '[' && stripped.back() == ']') {
    stripped = stripped.substr(1, stripped.length() - 2);
  }

  size_t pos = 0;
  while (pos < stripped.length()) {
    // Find next zone
    size_t start = stripped.find('[', pos);
    if (start == std::string::npos) break;
    size_t end = stripped.find(']', start);
    if (end == std::string::npos) break;

    std::string zone_str = stripped.substr(start + 1, end - start - 1);
    std::vector<int> coords;
    size_t coord_start = 0;
    size_t comma_pos;

    // Parse coordinates
    while ((comma_pos = zone_str.find(',', coord_start)) != std::string::npos) {
      std::string num_str = zone_str.substr(coord_start, comma_pos - coord_start);
      coords.push_back(std::stoi(num_str));
      coord_start = comma_pos + 1;
      if (comma_pos == std::string::npos) break;
    }
    // Get last number
    if (coord_start < zone_str.length()) {
      coords.push_back(std::stoi(zone_str.substr(coord_start)));
    }

    if (coords.size() == 4) {
      crop_zones_.push_back({coords[0], coords[1], coords[2], coords[3]});
      ESP_LOGD(TAG, "Added crop zone: [%d,%d,%d,%d]", 
               coords[0], coords[1], coords[2], coords[3]);
    } else {
      ESP_LOGE(TAG, "Invalid zone format: %s", zone_str.c_str());
    }

    pos = end + 1;
  }

  ESP_LOGI(TAG, "Parsed %d crop zones", crop_zones_.size());
}


 
void MeterReaderTFLite::set_camera_format(int width, int height, const std::string &pixel_format) {
  camera_width_ = width;
  camera_height_ = height;
  pixel_format_ = pixel_format;
  
  if (pixel_format == "RGB888") {
    bytes_per_pixel_ = 3;
  } else if (pixel_format == "GRAYSCALE") {
    bytes_per_pixel_ = 1;
  } else if (pixel_format == "RGB565") {
    bytes_per_pixel_ = 2;
  } else {
    ESP_LOGE(TAG, "Unsupported pixel format: %s", pixel_format.c_str());
    bytes_per_pixel_ = 3; // Default to RGB888
  }
}

std::shared_ptr<camera::CameraImage> MeterReaderTFLite::crop_and_resize_image(
    std::shared_ptr<camera::CameraImage> image, const CropZone &zone) {
  // Validate camera format is set
  if (camera_width_ == 0 || camera_height_ == 0) {
    ESP_LOGE(TAG, "Camera format not initialized!");
    return nullptr;
  }

  // Validate crop zone
  if (zone.x1 < 0 || zone.y1 < 0 || 
      zone.x2 > camera_width_ || zone.y2 > camera_height_ ||
      zone.x1 >= zone.x2 || zone.y1 >= zone.y2) {
    ESP_LOGE(TAG, "Invalid crop zone [%d,%d,%d,%d] for camera %dx%d",
             zone.x1, zone.y1, zone.x2, zone.y2, camera_width_, camera_height_);
    return nullptr;
  }

  const int crop_width = zone.x2 - zone.x1;
  const int crop_height = zone.y2 - zone.y1;
  
  // Allocate buffer for final image
  const size_t output_size = model_input_width_ * model_input_height_ * bytes_per_pixel_;
  uint8_t *output_data = new uint8_t[output_size];

  // Get source image data
  const uint8_t *src = image->get_data_buffer();

  // Handle different pixel formats
  if (pixel_format_ == "RGB888") {
    // RGB888 implementation
    const float x_ratio = static_cast<float>(crop_width) / model_input_width_;
    const float y_ratio = static_cast<float>(crop_height) / model_input_height_;

    for (int y = 0; y < model_input_height_; y++) {
      for (int x = 0; x < model_input_width_; x++) {
        const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
        const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
        const size_t src_idx = (src_y * camera_width_ + src_x) * 3;
        const size_t dst_idx = (y * model_input_width_ + x) * 3;
        
        output_data[dst_idx] = src[src_idx];     // R
        output_data[dst_idx+1] = src[src_idx+1]; // G
        output_data[dst_idx+2] = src[src_idx+2]; // B
      }
    }
  }
  else if (pixel_format_ == "GRAYSCALE") {
    // Grayscale implementation
    const float x_ratio = static_cast<float>(crop_width) / model_input_width_;
    const float y_ratio = static_cast<float>(crop_height) / model_input_height_;

    for (int y = 0; y < model_input_height_; y++) {
      for (int x = 0; x < model_input_width_; x++) {
        const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
        const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
        output_data[y * model_input_width_ + x] = src[src_y * camera_width_ + src_x];
      }
    }
  }

  // Create wrapper for the output image
  struct CroppedImage : public camera::CameraImage {
    uint8_t *data_;
    size_t length_;
    
    CroppedImage(uint8_t *data, size_t length) : data_(data), length_(length) {}
    ~CroppedImage() override { delete[] data_; }
    
    uint8_t *get_data_buffer() override { return data_; }
    size_t get_data_length() override { return length_; }
    // bool was_requested_by(uint8_t) const override { return true; }
	bool was_requested_by(camera::CameraRequester requester) const override { 
	  (void)requester;
	  return true;
	}
  };
  
  return std::make_shared<CroppedImage>(output_data, output_size);
}

void MeterReaderTFLite::process_image() {
  if (!current_image_) {
    ESP_LOGE(TAG, "No image available for processing");
    return;
  }

  if (crop_zones_.empty()) {
    // Process full image if no crop zones defined
    ESP_LOGD(TAG, "Processing full image");
    process_single_image(current_image_);
  } else {
    // Process each crop zone
    for (const auto &zone : crop_zones_) {
      ESP_LOGD(TAG, "Processing crop zone [%d,%d,%d,%d]", 
               zone.x1, zone.y1, zone.x2, zone.y2);
      auto cropped = crop_and_resize_image(current_image_, zone);
      if (cropped) {
        process_single_image(cropped);
      }
    }
  }

  this->return_image();
}

void MeterReaderTFLite::process_single_image(std::shared_ptr<camera::CameraImage> image) {
  const uint8_t *data = image->get_data_buffer();
  size_t length = image->get_data_length();

  TfLiteTensor *input_tensor = interpreter_->input(0);
  if (!input_tensor) {
    ESP_LOGE(TAG, "Failed to get input tensor");
    return;
  }

  if (input_tensor->bytes != length) {
    ESP_LOGE(TAG, "Input tensor size mismatch (%d vs %d)", input_tensor->bytes, length);
    return;
  }

  memcpy(input_tensor->data.data, data, length);

  TfLiteStatus invoke_status = interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke failed");
    return;
  }

  TfLiteTensor *output_tensor = interpreter_->output(0);
  if (!output_tensor) {
    ESP_LOGE(TAG, "Failed to get output tensor");
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
}

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

  ESP_LOGD(TAG, "load_model: calling GetModel()");
  tflite_model_ = tflite::GetModel(model_);
  if (tflite_model_ == nullptr) {
    ESP_LOGE(TAG, "Failed to get model from buffer. The model data may be corrupt or invalid.");
    return false;
  }

  if (tflite_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch");
    return false;
  }

  ESP_LOGD(TAG, "load_model: allocating tensor arena");
  if (!allocate_tensor_arena()) {
    return false;
  }

  static tflite::MicroMutableOpResolver<MAX_OPERATORS> resolver;

  ESP_LOGD(TAG, "load_model: parsing operators");
  const auto *subgraphs = tflite_model_->subgraphs();
  if (subgraphs->size() != 1) {
    ESP_LOGE(TAG, "Only single subgraph models are supported");
    return false;
  }

  // First collect all required ops from operator codes
  std::set<tflite::BuiltinOperator> required_ops;
  for (size_t i = 0; i < tflite_model_->operator_codes()->size(); ++i) {
    const auto *op_code = tflite_model_->operator_codes()->Get(i);
    auto builtin_code = op_code->builtin_code();
    required_ops.insert(builtin_code);
  }

	// Register all required ops at once
  if (!meter_reader_tflite::OpResolverManager::RegisterOps(resolver, required_ops, TAG)) {
    ESP_LOGE(TAG, "Failed to register all required operators");
    return false;
  }

  ESP_LOGD(TAG, "load_model: creating interpreter");
  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      tflite_model_,
      resolver,
      tensor_arena_.get(),
      tensor_arena_size_actual_);

  ESP_LOGD(TAG, "load_model: allocating tensors");
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
	
  // Allocate from PSRAM if available, otherwise fall back to internal RAM.
  // Using heap_caps_malloc is more robust for large allocations on ESP32.
    arena_ptr = static_cast<uint8_t *>(heap_caps_malloc(tensor_arena_size_actual_, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  }

  if (arena_ptr == nullptr) {
    ESP_LOGE(TAG, "Failed to allocate tensor arena from both PSRAM and internal RAM. Try reducing tensor_arena_size.");
    return false;
  }

  // Use reset() to assign the raw pointer to the unique_ptr with the custom deleter.
  tensor_arena_.reset(arena_ptr);
  return true;
}

size_t MeterReaderTFLite::get_arena_peak_bytes() const {
	// arena_used_bytes() returns the peak memory usage of the arena after tensor allocation.
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



// Helper function to dump tensor contents for debugging.


static void hexdump_tensor(const char *tag, const TfLiteTensor *tensor) {

  if (tensor == nullptr) {
    ESP_LOGW(tag, "Attempted to hexdump a null tensor.");
    return;
  }
  // The 'name' field is removed in newer TFLite versions.
  ESP_LOGD(tag, "Hexdump of tensor (%zu bytes, type %d):", tensor->bytes, tensor->type);
  ESP_LOG_BUFFER_HEXDUMP(tag, tensor->data.data, tensor->bytes, ESP_LOG_DEBUG);
}



void MeterReaderTFLite::loop() {
  // Nothing here; image capture is async via callback
}

}  // namespace meter_reader_tflite
}  // namespace esphome
