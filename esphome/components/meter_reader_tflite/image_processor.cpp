#include "image_processor.h"
#include "esp_log.h"
#ifdef USE_JPEG
#include "jpeg_decoder.h"
#endif
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
    bytes_per_pixel_ = 1;
  } else if (config_.pixel_format == "JPEG") {
    bytes_per_pixel_ = 3;
  } else {
    ESP_LOGE(TAG, "Unsupported pixel format: %s", config_.pixel_format.c_str());
    bytes_per_pixel_ = 3;
  }
}

std::vector<ImageProcessor::ProcessResult> ImageProcessor::process_image(
    std::shared_ptr<camera::CameraImage> image,
    const std::vector<CropZone> &zones) {
  
  std::vector<ProcessResult> results;
  ESP_LOGD(TAG, "Processing image with %s format, %dx%d input -> %dx%d output",
           config_.pixel_format.c_str(),
           config_.camera_width, config_.camera_height,
           config_.model_input_width, config_.model_input_height);

  if (zones.empty()) {
    ESP_LOGD(TAG, "Processing full image as no zones specified");
    CropZone full_zone{0, 0, config_.camera_width, config_.camera_height};
    ProcessResult result;
    if (config_.pixel_format == "JPEG") {
      ESP_LOGD(TAG, "Using JPEG decoder path");
      result = decode_and_process_jpeg(image, full_zone);
    } else {
      ESP_LOGD(TAG, "Using direct crop/resize path");
      result = crop_and_resize(image, full_zone);
    }
    if (result.data) {
      results.push_back(std::move(result));
    }
  } else {
    ESP_LOGD(TAG, "Processing %d crop zones", zones.size());
    for (const auto &zone : zones) {
      if (validate_zone(zone)) {
        ESP_LOGD(TAG, "Processing zone [%d,%d,%d,%d]", 
                zone.x1, zone.y1, zone.x2, zone.y2);
        ProcessResult result;
        if (config_.pixel_format == "JPEG") {
          result = decode_and_process_jpeg(image, zone);
        } else {
          result = crop_and_resize(image, zone);
        }
        if (result.data) {
          results.push_back(std::move(result));
        }
      }
    }
  }
  
  ESP_LOGD(TAG, "Generated %d processed image regions", results.size());
  return results;
}


#ifdef USE_JPEG
ImageProcessor::ProcessResult ImageProcessor::decode_and_process_jpeg(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone) {
  
  ProcessResult result{nullptr, 0};
  ESP_LOGD(TAG, "Starting JPEG decode for zone [%d,%d,%d,%d]",
           zone.x1, zone.y1, zone.x2, zone.y2);
  
  if (!validate_zone(zone)) {
    return result;
  }

  const uint8_t *jpeg_data = image->get_data_buffer();
  size_t jpeg_size = image->get_data_length();
  
  if (!jpeg_data || jpeg_size == 0) {
    ESP_LOGE(TAG, "Invalid JPEG image data");
    return result;
  }

  // Get image info first
  esp_jpeg_image_cfg_t jpeg_cfg = {
    .indata = const_cast<uint8_t*>(jpeg_data),
    .indata_size = jpeg_size,
    .outbuf = nullptr,
    .outbuf_size = 0,
    .out_format = JPEG_IMAGE_FORMAT_RGB888,
    .out_scale = JPEG_IMAGE_SCALE_0
  };

  esp_jpeg_image_output_t out_info;
  esp_err_t ret = esp_jpeg_get_image_info(&jpeg_cfg, &out_info);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to get JPEG info: %s", esp_err_to_name(ret));
    return result;
  }

  // Allocate buffer for decoded image
  std::unique_ptr<uint8_t[]> decoded_data(new uint8_t[out_info.width * out_info.height * 3]);

  // Setup for decoding
  jpeg_cfg.outbuf = decoded_data.get();
  jpeg_cfg.outbuf_size = out_info.width * out_info.height * 3;
  jpeg_cfg.flags.swap_color_bytes = 1; // Convert BGR to RGB

  // Decode the image
  ret = esp_jpeg_decode(&jpeg_cfg, &out_info);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "JPEG decode failed: %s", esp_err_to_name(ret));
    return result;
  }

  return crop_and_resize_from_decoded(decoded_data.get(), out_info.width, out_info.height, zone);
}
#endif

ImageProcessor::ProcessResult ImageProcessor::crop_and_resize_from_decoded(
    const uint8_t *decoded_data,
    int original_width,
    int original_height,
    const CropZone &zone) {
  
  ProcessResult result{nullptr, 0};
  
  const size_t output_size = config_.model_input_width * config_.model_input_height * 3;
  result.data.reset(new uint8_t[output_size]);
  result.size = output_size;
  
  const float x_ratio = static_cast<float>(zone.x2 - zone.x1) / config_.model_input_width;
  const float y_ratio = static_cast<float>(zone.y2 - zone.y1) / config_.model_input_height;

  for (int y = 0; y < config_.model_input_height; y++) {
    for (int x = 0; x < config_.model_input_width; x++) {
      const int src_x = zone.x1 + static_cast<int>(x * x_ratio);
      const int src_y = zone.y1 + static_cast<int>(y * y_ratio);
      
      const size_t src_idx = (src_y * original_width + src_x) * 3;
      const size_t dst_idx = (y * config_.model_input_width + x) * 3;
      
      result.data.get()[dst_idx] = decoded_data[src_idx];     // R
      result.data.get()[dst_idx+1] = decoded_data[src_idx+1]; // G
      result.data.get()[dst_idx+2] = decoded_data[src_idx+2]; // B
    }
  }

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
  ESP_LOGD(TAG, "Cropping/resizing zone [%d,%d,%d,%d] to %dx%d",
           zone.x1, zone.y1, zone.x2, zone.y2,
           config_.model_input_width, config_.model_input_height);
  
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
  } else { // GRAYSCALE, RAW, YUV420
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