#include "image_processor.h"
#include "esp_log.h"
#include "debug_utils.h"
#include "managed_components/espressif__esp32-camera/conversions/include/img_converters.h"
#include <algorithm>

    // Model Expects: 7680 bytes (likely 32x32x3 = 3072 uint8 elements)
    // image : 1920 bytes (32x20x3 = 1920 uint8 elements)

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "ImageProcessor";

ImageProcessor::ImageProcessor(const ImageProcessorConfig &config, 
                             ModelHandler* model_handler)
  : config_(config), model_handler_(model_handler) {
  
  if (!config_.validate()) {
    ESP_LOGE(TAG, "Invalid image processor configuration");
  }

  if (config_.pixel_format == "RGB888") {
    bytes_per_pixel_ = 3;
    ESP_LOGD(TAG, "Using RGB888 format (3 bytes per pixel)");
  } else if (config_.pixel_format == "RGB565") {
    bytes_per_pixel_ = 2;
    ESP_LOGD(TAG, "Using RGB565 format (2 bytes per pixel)");
  } else if (config_.pixel_format == "YUV422") {
    bytes_per_pixel_ = 2;
    ESP_LOGD(TAG, "Using YUV422 format (2 bytes per pixel)");
  } else if (config_.pixel_format == "GRAYSCALE") {
    bytes_per_pixel_ = 1;
    ESP_LOGD(TAG, "Using grayscale format (1 byte per pixel)");
  } else if (config_.pixel_format == "JPEG") {
    bytes_per_pixel_ = 3;
    ESP_LOGD(TAG, "Using JPEG format (will decode to RGB888)");
  } else {
    ESP_LOGE(TAG, "Unsupported pixel format: %s", config_.pixel_format.c_str());
    bytes_per_pixel_ = 3;
    ESP_LOGW(TAG, "Defaulting to RGB888 format");
  }

  ESP_LOGD(TAG, "ImageProcessor initialized with:");
  ESP_LOGD(TAG, "  Camera resolution: %dx%d", config_.camera_width, config_.camera_height);
  ESP_LOGD(TAG, "  Model input: %dx%dx%d", 
           model_handler_->get_input_width(),
           model_handler_->get_input_height(),
           model_handler_->get_input_channels());
}

bool ImageProcessor::jpeg_to_rgb888(const uint8_t* src, size_t src_len, uint8_t* dst, uint32_t width, uint32_t height) {
    // Allocate temporary buffer in PSRAM if available
    uint8_t* temp_buf = (uint8_t*)heap_caps_malloc(width * height * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!temp_buf) {
        ESP_LOGE(TAG, "Failed to allocate PSRAM buffer");
        return false;
    }

    bool result = fmt2rgb888(src, src_len, PIXFORMAT_JPEG, temp_buf);
    if (result) {
        memcpy(dst, temp_buf, width * height * 3);
    }
    
    heap_caps_free(temp_buf);
    return result;
}

bool ImageProcessor::validate_jpeg(const uint8_t* data, size_t size) {
  if (size < 4) {
    ESP_LOGE(TAG, "JPEG too small (%zu bytes)", size);
    return false;
  }
  
  if (data[0] != 0xFF || data[1] != 0xD8) {
    ESP_LOGE(TAG, "Missing JPEG SOI marker (got 0x%02X%02X)", data[0], data[1]);
    return false;
  }
  
  bool has_eoi = false;
  for (size_t i = 2; i < size - 1; i++) {
    if (data[i] == 0xFF && data[i+1] == 0xD9) {
      has_eoi = true;
      break;
    }
  }
  
  if (!has_eoi) {
    ESP_LOGW(TAG, "No JPEG EOI marker found");
  }
  
  return true;
}

std::vector<ImageProcessor::ProcessResult> ImageProcessor::split_image_in_zone(
    std::shared_ptr<camera::CameraImage> image,
    const std::vector<CropZone> &zones) {
  
  std::vector<ProcessResult> results;
  ESP_LOGD(TAG, "Starting image processing");
  ESP_LOGD(TAG, "Input image size: %zu bytes", image->get_data_length());

  if (zones.empty()) {
    ESP_LOGD(TAG, "No zones specified - processing full image");
    CropZone full_zone{0, 0, config_.camera_width, config_.camera_height};
    ProcessResult result = process_zone(image, full_zone);
    if (result.data) {
      ESP_LOGD(TAG, "Full image processed successfully (%zu bytes)", result.size);
      results.push_back(std::move(result));
    } else {
      ESP_LOGE(TAG, "Failed to process full image");
    }
  } else {
    ESP_LOGD(TAG, "Processing %d crop zones", zones.size());
    for (size_t i = 0; i < zones.size(); i++) {
      ESP_LOGD(TAG, "Processing zone %d: [%d,%d,%d,%d]", 
              i+1, zones[i].x1, zones[i].y1, zones[i].x2, zones[i].y2);
      
      ProcessResult result = process_zone(image, zones[i]);
      if (result.data) {
        ESP_LOGD(TAG, "Zone %d processed successfully (%zu bytes)", i+1, result.size);
        results.push_back(std::move(result));
      } else {
        ESP_LOGE(TAG, "Failed to process zone %d", i+1);
      }
    }
  }
  
  ESP_LOGD(TAG, "Image processing complete. Generated %d outputs", results.size());
  return results;
}

ImageProcessor::ProcessResult ImageProcessor::process_zone(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone) {
  
  ProcessResult result{nullptr, 0};
  
  if (!validate_zone(zone)) {
    ESP_LOGE(TAG, "Invalid zone [%d,%d,%d,%d]", zone.x1, zone.y1, zone.x2, zone.y2);
    return result;
  }

  if (config_.pixel_format == "JPEG") {
    ESP_LOGD(TAG, "Decoding JPEG for zone [%d,%d,%d,%d]", zone.x1, zone.y1, zone.x2, zone.y2);
    const uint8_t *jpeg_data = image->get_data_buffer();
    size_t jpeg_size = image->get_data_length();
    
    if (!jpeg_data || jpeg_size == 0) {
      ESP_LOGE(TAG, "Invalid JPEG data");
      return result;
    }
    
    if (!validate_jpeg(jpeg_data, jpeg_size)) {
      ESP_LOGE(TAG, "Invalid JPEG markers");
      return result;
    }

    // Allocate RGB888 buffer
    size_t rgb_size = config_.camera_width * config_.camera_height * 3;
    auto rgb_buffer = std::unique_ptr<uint8_t[]>(new uint8_t[rgb_size]);
    
    if (!jpeg_to_rgb888(jpeg_data, jpeg_size, rgb_buffer.get(), 
                       config_.camera_width, config_.camera_height)) {
      return result;
    }
    
    return scale_cropped_region(rgb_buffer.get(), 
                              config_.camera_width, 
                              config_.camera_height, 
                              zone);
  }

  return scale_cropped_region(
      image->get_data_buffer(),
      config_.camera_width,
      config_.camera_height,
      zone);
}

ImageProcessor::ProcessResult ImageProcessor::scale_cropped_region(
    const uint8_t *src_data,
    int src_width,
    int src_height,
    const CropZone &zone) {
    
    DURATION_START();
    
    const int model_width = model_handler_->get_input_width(); // 32
    const int model_height = model_handler_->get_input_height(); // 20
    const int model_channels = model_handler_->get_input_channels(); // 3
	
	const size_t required_size = model_width * model_height * model_channels;
	ESP_LOGD(TAG, "Model requires %dx%dx%d (%zu bytes)", 
            model_width, model_height, model_channels, required_size);
    
    UniqueBufferPtr buffer = allocate_image_buffer(required_size);
    if (!buffer) return ProcessResult(nullptr, 0);

    uint8_t* raw_buffer = buffer.get();
    const bool is_quantized = model_handler_->is_model_quantized();

    // Calculate scaling factors
    const float width_scale = static_cast<float>(zone.x2 - zone.x1) / model_width;
    const float height_scale = static_cast<float>(zone.y2 - zone.y1) / model_height;

    for (int y = 0; y < model_height; y++) {
        for (int x = 0; x < model_width; x++) {
            const int src_x = zone.x1 + static_cast<int>(x * width_scale);
            const int src_y = zone.y1 + static_cast<int>(y * height_scale);
            const size_t src_pixel_index = (src_y * src_width + src_x) * bytes_per_pixel_;
            const size_t dst_pixel_index = (y * model_width + x) * model_channels;

            if (bytes_per_pixel_ == 1) { // Grayscale
                const uint8_t val = src_data[src_pixel_index];
                raw_buffer[dst_pixel_index] = val;
                raw_buffer[dst_pixel_index+1] = val;
                raw_buffer[dst_pixel_index+2] = val;
            } else { // RGB
                for (int c = 0; c < model_channels; c++) {
                    raw_buffer[dst_pixel_index + c] = 
                        src_data[src_pixel_index + (c % bytes_per_pixel_)];
                }
            }
        }
    }
    
    ESP_LOGD(TAG, "Scaled region: %dx%d->%dx%dx%d (size: %zu)",
            zone.x2-zone.x1, zone.y2-zone.y1,
            model_width, model_height, model_channels,
            required_size);
    
    DURATION_END("scale_cropped_region");
    return ProcessResult(std::move(buffer), required_size);
}

bool ImageProcessor::validate_zone(const CropZone &zone) const {
  if (zone.x1 < 0 || zone.y1 < 0 || 
      zone.x2 > config_.camera_width || zone.y2 > config_.camera_height ||
      zone.x1 >= zone.x2 || zone.y1 >= zone.y2) {
    ESP_LOGE(TAG, "Invalid crop zone [%d,%d,%d,%d] (camera: %dx%d)", 
             zone.x1, zone.y1, zone.x2, zone.y2,
             config_.camera_width, config_.camera_height);
    return false;
  }
  ESP_LOGD(TAG, "Valid crop zone [%d,%d,%d,%d]", zone.x1, zone.y1, zone.x2, zone.y2);
  return true;
}

}  // namespace meter_reader_tflite
}  // namespace esphome