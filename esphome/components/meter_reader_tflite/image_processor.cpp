#include "image_processor.h"
#include "esp_log.h"
#include "debug_utils.h"
#ifdef USE_JPEG
#include "jpeg_decoder.h"
#endif
#include <algorithm>

// ---- split_image_in_zone() --> process_zone() --> scale_cropped_region()

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "ImageProcessor";

ImageProcessor::ImageProcessor(const ImageProcessorConfig &config, 
                             ModelHandler* model_handler)
  : config_(config), model_handler_(model_handler) {
  
  if (!config_.validate()) {
    ESP_LOGE(TAG, "Invalid image processor configuration");
  }

  // Calculate bytes per pixel with detailed logging
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
#ifdef USE_JPEG
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

    // JPEG decoding with PSRAM allocation
    esp_jpeg_image_output_t out_info;
    esp_jpeg_image_cfg_t jpeg_cfg = {
      .indata = const_cast<uint8_t*>(jpeg_data),
      .indata_size = jpeg_size,
      .outbuf = nullptr,
      .outbuf_size = 0,
      .out_format = JPEG_IMAGE_FORMAT_RGB888,
      .out_scale = JPEG_IMAGE_SCALE_0,
      .flags = {.swap_color_bytes = 1}
    };

    // Get image info first
    if (esp_jpeg_get_image_info(&jpeg_cfg, &out_info) != ESP_OK) {
      ESP_LOGE(TAG, "JPEG info failed");
      return result;
    }

    // Allocate output buffer in PSRAM
    uint8_t* decoded_data = static_cast<uint8_t*>(
        heap_caps_malloc(out_info.width * out_info.height * 3, 
                        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    
    if (!decoded_data) {
      ESP_LOGE(TAG, "Failed to allocate PSRAM for JPEG decode");
      return result;
    }

    // Set up decoder with PSRAM buffer
    jpeg_cfg.outbuf = decoded_data;
    jpeg_cfg.outbuf_size = out_info.width * out_info.height * 3;

    // Create a custom deleter for the unique_ptr
    auto psram_deleter = [](uint8_t* p) {
      if (p) heap_caps_free(p);
    };

    std::unique_ptr<uint8_t[], decltype(psram_deleter)> decoded_buffer(decoded_data, psram_deleter);

    if (esp_jpeg_decode(&jpeg_cfg, &out_info) != ESP_OK) {
      ESP_LOGE(TAG, "JPEG decode failed");
      return result;
    }

    ESP_LOGD(TAG, "JPEG decoded successfully, processing cropped region");
    return scale_cropped_region(decoded_buffer.get(), out_info.width, out_info.height, zone);
#else
    ESP_LOGE(TAG, "JPEG support not compiled in");
    return result;
#endif
  }

  // Non-JPEG processing remains the same
  return scale_cropped_region(
      image->get_data_buffer(),
      config_.camera_width,
      config_.camera_height,
      zone);
}


bool ImageProcessor::validate_jpeg(const uint8_t* data, size_t size) {
    // Minimum JPEG size is 2 bytes for SOI marker plus some content
    if (size < 4) {
        ESP_LOGE(TAG, "JPEG too small (%zu bytes)", size);
        return false;
    }
    
    // Check for JPEG Start of Image marker (0xFFD8)
    if (data[0] != 0xFF || data[1] != 0xD8) {
        ESP_LOGE(TAG, "Missing JPEG SOI marker (got 0x%02X%02X)", data[0], data[1]);
        return false;
    }
    
    // Optional: Check for End of Image marker somewhere in the data
    // (Don't require it to be at the end in case we got truncated data)
    bool has_eoi = false;
    for (size_t i = 2; i < size - 1; i++) {
        if (data[i] == 0xFF && data[i+1] == 0xD9) {
            has_eoi = true;
            break;
        }
    }
    
    if (!has_eoi) {
        ESP_LOGW(TAG, "No JPEG EOI marker found");
        // Don't return false here as some cameras might send partial frames
    }
    
    return true;
}


/* ImageProcessor::ProcessResult ImageProcessor::scale_cropped_region(
    const uint8_t *src_data,
    int src_width,
    int src_height,
    const CropZone &zone) {
  
  ProcessResult result{nullptr, 0};
  const int crop_width = zone.x2 - zone.x1;
  const int crop_height = zone.y2 - zone.y1;
  
  const int model_width = model_handler_->get_input_width(); //20
  const int model_height = model_handler_->get_input_height(); // 32
  const int model_channels = model_handler_->get_input_channels(); // 3

  if (model_width <= 0 || model_height <= 0 || model_channels <= 0) {
    ESP_LOGE(TAG, "Invalid model dimensions: %dx%dx%d", 
            model_width, model_height, model_channels);
    return result;
  }

  ESP_LOGD(TAG, "Processing crop region:");
  ESP_LOGD(TAG, "  Source: %dx%d (crop %dx%d)", src_width, src_height, crop_width, crop_height);
  ESP_LOGD(TAG, "  Target: %dx%dx%d", model_width, model_height, model_channels);

  result.data.reset(new uint8_t[model_width * model_height * model_channels]());
  result.size = model_width * model_height * model_channels;

  float scale = std::min(
      static_cast<float>(model_width) / crop_width,
      static_cast<float>(model_height) / crop_height
  );

  int scaled_width = crop_width * scale;
  int scaled_height = crop_height * scale;
  int x_offset = (model_width - scaled_width) / 2;
  int y_offset = (model_height - scaled_height) / 2;

  ESP_LOGD(TAG, "Scaling parameters:");
  ESP_LOGD(TAG, "  Scale: %.2f", scale);
  ESP_LOGD(TAG, "  Scaled: %dx%d", scaled_width, scaled_height);
  ESP_LOGD(TAG, "  Offset: %d,%d", x_offset, y_offset);

  for (int y = 0; y < scaled_height; y++) {
    for (int x = 0; x < scaled_width; x++) {
      int src_x = zone.x1 + (x / scale);
      int src_y = zone.y1 + (y / scale);
      
      for (int c = 0; c < model_channels; c++) {
        size_t src_idx = (src_y * src_width + src_x) * bytes_per_pixel_ + (c % bytes_per_pixel_);
        size_t dst_idx = ((y + y_offset) * model_width + (x + x_offset)) * model_channels + c;
        result.data[dst_idx] = src_data[src_idx];
      }
    }
  }
  ESP_LOGD("ImageProcessor", "Processed region size: %zu bytes", result.size);
  
  ESP_LOGD(TAG, "Image processing completed successfully");
  return result;
} */

ImageProcessor::ProcessResult ImageProcessor::scale_cropped_region(
    const uint8_t *src_data,
    int src_width,
    int src_height,
    const CropZone &zone) {
		
  DURATION_START()
  
  ProcessResult result{nullptr, 0};
  const int model_width = model_handler_->get_input_width(); // 32
  const int model_height = model_handler_->get_input_height(); // 20
  const int model_channels = model_handler_->get_input_channels(); // 3
  
  // Allocate exact size needed by model
  result.data.reset(new uint8_t[model_width * model_height * model_channels]());
  result.size = model_width * model_height * model_channels;

  // Calculate scaling factors
  float width_scale = static_cast<float>(model_width) / (zone.x2 - zone.x1);
  float height_scale = static_cast<float>(model_height) / (zone.y2 - zone.y1);

  // Simple nearest-neighbor scaling
  for (int y = 0; y < model_height; y++) {
    for (int x = 0; x < model_width; x++) {
      int src_x = zone.x1 + static_cast<int>(x / width_scale);
      int src_y = zone.y1 + static_cast<int>(y / height_scale);
      
      for (int c = 0; c < model_channels; c++) {
        size_t src_idx = (src_y * src_width + src_x) * bytes_per_pixel_ + (c % bytes_per_pixel_);
        size_t dst_idx = (y * model_width + x) * model_channels + c;
        result.data[dst_idx] = src_data[src_idx];
      }
    }
  }
  
  DURATION_END("scale_cropped_region");
  return result;
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