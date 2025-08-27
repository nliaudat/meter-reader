#include "image_processor.h"
#include "esp_log.h"
#include "debug_utils.h"
#include "managed_components/espressif__esp32-camera/conversions/include/img_converters.h"
#include <algorithm>

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

bool ImageProcessor::jpeg_to_rgb888(const uint8_t* src, size_t src_len, uint8_t* dst) {
    return fmt2rgb888(src, src_len, PIXFORMAT_JPEG, dst);
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
    auto rgb_buffer = allocate_image_buffer(rgb_size);
    if (!rgb_buffer) {
      ESP_LOGE(TAG, "Failed to allocate RGB buffer for JPEG decoding");
      return result;
    }
    
    if (!jpeg_to_rgb888(jpeg_data, jpeg_size, rgb_buffer->get())) {
      ESP_LOGE(TAG, "JPEG to RGB888 conversion failed");
      return result;
    }
    
    return scale_cropped_region(rgb_buffer->get(), 
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
    
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    const int model_channels = model_handler_->get_input_channels();
    const size_t required_size = model_width * model_height * model_channels;
    
    ESP_LOGD(TAG, "Model requires %dx%dx%d (%zu bytes)", model_width, model_height, model_channels, required_size);    

    // Allocate buffer
    UniqueBufferPtr buffer = allocate_image_buffer(required_size);
    if (!buffer) {
      ESP_LOGE(TAG, "Failed to allocate output buffer");
      return ProcessResult(nullptr, 0);
    }

    // Fixed-point scaling factors (16.16 format)
    const uint32_t width_scale = ((zone.x2 - zone.x1) << 16) / model_width;
    const uint32_t height_scale = ((zone.y2 - zone.y1) << 16) / model_height;

    uint8_t* dst = buffer->get();
    
    // Optimized scaling loop
    for (int y = 0; y < model_height; y++) {
        const int src_y = zone.y1 + ((y * height_scale) >> 16);
        const size_t src_row_offset = src_y * src_width * bytes_per_pixel_;
        
        for (int x = 0; x < model_width; x++) {
            const int src_x = zone.x1 + ((x * width_scale) >> 16);
            const size_t src_pixel_offset = src_row_offset + src_x * bytes_per_pixel_;
            const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

            // Handle different pixel formats
            if (bytes_per_pixel_ == 1) { // Grayscale
                const uint8_t val = src_data[src_pixel_offset];
                dst[dst_pixel_offset] = val;
                dst[dst_pixel_offset+1] = val;
                dst[dst_pixel_offset+2] = val;
            } else { // RGB
                for (int c = 0; c < model_channels; c++) {
                    dst[dst_pixel_offset + c] = src_data[src_pixel_offset + (c % bytes_per_pixel_)];
                }
            }
        }
    }

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