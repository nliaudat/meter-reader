#include "image_processor.h"
#include "esp_log.h"
// #include "jpeg_decoder.h"
// #include "esphome/components/esp32_camera/esp32_camera.h" 
#include <cstring>

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "ImageProcessor";

ImageProcessor::ImageProcessor(const ImageProcessorConfig &config) : config_(config) {
  if (!config_.validate()) {
    ESP_LOGE(TAG, "Invalid image processor configuration");
  }

  if (config_.pixel_format == "RGB888") {
    bytes_per_pixel_ = 3;
  } else if (config_.pixel_format == "RGB565" || 
             config_.pixel_format == "YUV422" ||
             config_.pixel_format == "RGB444" ||
             config_.pixel_format == "RGB555") {
    bytes_per_pixel_ = 2;
  } else if (config_.pixel_format == "GRAYSCALE" ||
             config_.pixel_format == "RAW") {
    bytes_per_pixel_ = 1;
  } else if (config_.pixel_format == "YUV420") {
    bytes_per_pixel_ = 1; // YUV420 is more complex, but we'll treat it as grayscale for simplicity
  } else if (config_.pixel_format == "JPEG") {
    bytes_per_pixel_ = 3; // JPEG will be decoded to RGB888
  } else {
    ESP_LOGE(TAG, "Unsupported pixel format: %s", config_.pixel_format.c_str());
    bytes_per_pixel_ = 3; // Default to RGB888
  }
}

std::vector<ImageProcessor::ProcessResult> ImageProcessor::process_image(
    std::shared_ptr<camera::CameraImage> image,
    const std::vector<CropZone> &zones) {
  
  std::vector<ProcessResult> results;
  
  if (zones.empty()) {
    // Process full image if no zones specified
    CropZone full_zone{0, 0, config_.camera_width, config_.camera_height};
    ProcessResult result;
    if (config_.pixel_format == "JPEG") {
      result = decode_and_process_jpeg(image, full_zone);
    } else {
      result = crop_and_resize(image, full_zone);
    }
    if (result.data) {
      results.push_back(std::move(result));
    }
  } else {
    for (const auto &zone : zones) {
      if (validate_zone(zone)) {
        ProcessResult result;
        if (config_.pixel_format == "JPEG") {
          result = decode_and_process_jpeg(image, zone);
        } else {
          result = crop_and_resize(image, zone);
        }
        if (result.data) {
          results.push_back(std::move(result));
        } else {
          ESP_LOGE(TAG, "Failed to process zone [%d,%d,%d,%d]", 
                  zone.x1, zone.y1, zone.x2, zone.y2);
        }
      }
    }
  }
  
  return results;
}


ImageProcessor::ProcessResult ImageProcessor::decode_and_process_jpeg(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone) {
  
  ProcessResult result{nullptr, 0};
  
  if (!validate_zone(zone)) {
    return result;
  }

  // Get the JPEG data buffer
  const uint8_t *jpeg_data = image->get_data_buffer();
  size_t jpeg_size = image->get_data_length();
  
  if (!jpeg_data || jpeg_size == 0) {
    ESP_LOGE(TAG, "Invalid JPEG image data");
    return result;
  }

  // For ESPHome, we'll need to use a different approach since the camera component
  // doesn't expose direct JPEG decoding. We'll implement a simple downscaling
  // approach for JPEG images.
  
  // Allocate output buffer (we'll convert to RGB888)
  const size_t output_size = config_.model_input_width * config_.model_input_height * 3;
  result.data.reset(new uint8_t[output_size]);
  result.size = output_size;
  
  // Simple approach: treat JPEG as grayscale for basic processing
  // Note: This is a simplified approach - for full JPEG decoding you would need
  // to integrate a proper JPEG decoder library
  uint8_t *dst = result.data.get();
  
  const float x_ratio = static_cast<float>(zone.x2 - zone.x1) / config_.model_input_width;
  const float y_ratio = static_cast<float>(zone.y2 - zone.y1) / config_.model_input_height;

  // Simple downscaling (this won't properly decode JPEG, just demonstrates the structure)
  for (int y = 0; y < config_.model_input_height; y++) {
    for (int x = 0; x < config_.model_input_width; x++) {
      const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
      const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
      
      // For demonstration, we'll just use the first byte of each MCU (Minimum Coded Unit)
      // In a real implementation, you would properly decode the JPEG here
      size_t src_idx = (src_y * config_.camera_width + src_x) % jpeg_size;
      uint8_t val = jpeg_data[src_idx];
      
      // Convert to RGB888 (grayscale)
      size_t dst_idx = (y * config_.model_input_width + x) * 3;
      dst[dst_idx] = val;     // R
      dst[dst_idx+1] = val;   // G
      dst[dst_idx+2] = val;   // B
    }
  }

  ESP_LOGW(TAG, "Basic JPEG processing complete (not full decoding)");
  return result;
}


bool ImageProcessor::validate_zone(const CropZone &zone) const {
  if (zone.x1 < 0 || zone.y1 < 0 || 
      zone.x2 > config_.camera_width || zone.y2 > config_.camera_height ||
      zone.x1 >= zone.x2 || zone.y1 >= zone.y2) {
    ESP_LOGE(TAG, "Invalid crop zone [%d,%d,%d,%d]", 
             zone.x1, zone.y1, zone.x2, zone.y2);
    return false;
  }
  return true;
}

ImageProcessor::ProcessResult ImageProcessor::crop_and_resize(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone) {
  
  ProcessResult result{nullptr, 0};
  
  if (!validate_zone(zone)) {
    return result;
  }

  const int crop_width = zone.x2 - zone.x1;
  const int crop_height = zone.y2 - zone.y1;
  const size_t output_size = config_.model_input_width * config_.model_input_height * bytes_per_pixel_;
  
  result.data.reset(new uint8_t[output_size]);
  result.size = output_size;
  
  const uint8_t *src = image->get_data_buffer();
  uint8_t *dst = result.data.get();

  const float x_ratio = static_cast<float>(crop_width) / config_.model_input_width;
  const float y_ratio = static_cast<float>(crop_height) / config_.model_input_height;

  if (config_.pixel_format == "RGB888") {
    for (int y = 0; y < config_.model_input_height; y++) {
      for (int x = 0; x < config_.model_input_width; x++) {
        const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
        const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
        const size_t src_idx = (src_y * config_.camera_width + src_x) * 3;
        const size_t dst_idx = (y * config_.model_input_width + x) * 3;
        
        dst[dst_idx] = src[src_idx];
        dst[dst_idx+1] = src[src_idx+1];
        dst[dst_idx+2] = src[src_idx+2];
      }
    }
  } else if (config_.pixel_format == "RGB565" || 
             config_.pixel_format == "RGB444" ||
             config_.pixel_format == "RGB555") {
    for (int y = 0; y < config_.model_input_height; y++) {
      for (int x = 0; x < config_.model_input_width; x++) {
        const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
        const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
        const size_t src_idx = (src_y * config_.camera_width + src_x) * 2;
        const size_t dst_idx = (y * config_.model_input_width + x) * 2;
        
        dst[dst_idx] = src[src_idx];
        dst[dst_idx+1] = src[src_idx+1];
      }
    }
  } else if (config_.pixel_format == "YUV422") {
    for (int y = 0; y < config_.model_input_height; y++) {
      for (int x = 0; x < config_.model_input_width; x++) {
        const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
        const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
        const size_t src_idx = (src_y * config_.camera_width + src_x) * 2;
        const size_t dst_idx = (y * config_.model_input_width + x) * 2;
        
        dst[dst_idx] = src[src_idx];       // Y component
        dst[dst_idx+1] = src[src_idx+1];  // UV components
      }
    }
  } else { // GRAYSCALE, RAW, YUV420 (treated as grayscale)
    for (int y = 0; y < config_.model_input_height; y++) {
      for (int x = 0; x < config_.model_input_width; x++) {
        const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
        const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
        dst[y * config_.model_input_width + x] = src[src_y * config_.camera_width + src_x];
      }
    }
  }

  return result;
}

}  // namespace meter_reader_tflite
}  // namespace esphome