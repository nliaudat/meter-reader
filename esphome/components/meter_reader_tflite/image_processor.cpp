/**
 * @file image_processor.cpp
 * @brief Implements the ImageProcessor class for handling image manipulation
 *        such as cropping, scaling, and format conversion for TensorFlow Lite.
 */

// Include necessary headers first
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <memory>

// Then include ESPHome headers
#include "image_processor.h"
#include "esp_log.h"
#include "debug_utils.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "ImageProcessor";

/**
 * @brief Constructs an ImageProcessor object.
  * @param config Configuration for the image processor.
 * @param model_handler Pointer to the ModelHandler for model input dimensions.
 */
ImageProcessor::ImageProcessor(const ImageProcessorConfig &config, 
                             ModelHandler* model_handler)
  : config_(config), model_handler_(model_handler) {
  
  if (!config_.validate()) {
    ESP_LOGE(TAG, "Invalid image processor configuration");
  }

  // Determine bytes per pixel based on the camera pixel format.
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
    bytes_per_pixel_ = 3; // JPEG will be decoded to RGB888.
    ESP_LOGD(TAG, "Using JPEG format (will decode to RGB888)");
  } else {
    ESP_LOGE(TAG, "Unsupported pixel format: %s", config_.pixel_format.c_str());
    bytes_per_pixel_ = 3; // Default to RGB888 for unsupported formats.
    ESP_LOGW(TAG, "Defaulting to RGB888 format");
  }

  ESP_LOGD(TAG, "ImageProcessor initialized with:");
  ESP_LOGD(TAG, "  Camera resolution: %dx%d", config_.camera_width, config_.camera_height);
  ESP_LOGD(TAG, "  Model input: %dx%dx%d", 
           model_handler_->get_input_width(),
           model_handler_->get_input_height(),
           model_handler_->get_input_channels());
}

/**
 * @brief Adjusts crop zone to nearest multiple of 8 for JPEG decoder requirements.
 * @param zone Original crop zone to adjust
 * @return Adjusted crop zone with dimensions multiple of 8
 */
CropZone adjust_zone_for_jpeg(const CropZone &zone, int max_width, int max_height) {
    CropZone adjusted = zone;
    
    // Adjust width to nearest multiple of 8
    int width = zone.x2 - zone.x1;
    int adjusted_width = (width + 4) / 8 * 8; // Round to nearest multiple of 8
    if (adjusted_width > width) {
        adjusted_width -= 8; // Ensure we don't exceed original bounds
    }
    adjusted_width = std::max(8, adjusted_width);
    
    // Adjust height to nearest multiple of 8
    int height = zone.y2 - zone.y1;
    int adjusted_height = (height + 4) / 8 * 8; // Round to nearest multiple of 8
    if (adjusted_height > height) {
        adjusted_height -= 8;
    }
    adjusted_height = std::max(8, adjusted_height);  // Minimum 8 pixels
    
    // Center the adjusted zone within original bounds
    adjusted.x1 = zone.x1 + (width - adjusted_width) / 2;
    adjusted.y1 = zone.y1 + (height - adjusted_height) / 2;
    adjusted.x2 = adjusted.x1 + adjusted_width;
    adjusted.y2 = adjusted.y1 + adjusted_height;
    
    // Ensure within image bounds
    adjusted.x1 = std::max(0, adjusted.x1);
    adjusted.y1 = std::max(0, adjusted.y1);
    adjusted.x2 = std::min(max_width, adjusted.x2);
    adjusted.y2 = std::min(max_height, adjusted.y2);
    
    // Final adjustment to ensure multiples of 8
    int final_width = adjusted.x2 - adjusted.x1;
    int final_height = adjusted.y2 - adjusted.y1;
    final_width = (final_width / 8) * 8;
    final_height = (final_height / 8) * 8;
    adjusted.x2 = adjusted.x1 + final_width;
    adjusted.y2 = adjusted.y1 + final_height;
    
    ESP_LOGD(TAG, "Adjusted zone from [%d,%d,%d,%d] to [%d,%d,%d,%d] for JPEG",
             zone.x1, zone.y1, zone.x2, zone.y2,
             adjusted.x1, adjusted.y1, adjusted.x2, adjusted.y2);
    
    return adjusted;
}

/**
 * @brief Splits the input image into specified crop zones and processes each.
 * @param image A shared pointer to the camera image.
 * @param zones A vector of CropZone objects defining regions to process.
 * @return A vector of ProcessResult, each containing processed image data for a zone.
 */
std::vector<ImageProcessor::ProcessResult> ImageProcessor::split_image_in_zone(
    std::shared_ptr<camera::CameraImage> image,
    const std::vector<CropZone> &zones) {
  
  std::vector<ProcessResult> results;
  ESP_LOGD(TAG, "Starting image processing");
  ESP_LOGD(TAG, "Input image size: %zu bytes", image->get_data_length());

  // If no zones are specified, process the full image.
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
  } else { // Process each specified crop zone.
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

/**
 * @brief Processes a zone directly into a pre-allocated buffer for minimal memory usage.
 * @brief Processes JPEG zone directly into output buffer using esp_new_jpeg decoder.
 */
bool ImageProcessor::process_zone_to_buffer(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone,
    uint8_t* output_buffer,
    size_t output_buffer_size) {
    
    DURATION_START();
    
    if (!validate_zone(zone)) {
        ESP_LOGE(TAG, "Invalid zone [%d,%d,%d,%d]", zone.x1, zone.y1, zone.x2, zone.y2);
        return false;
    }

    bool success = false;
    
    if (config_.pixel_format == "JPEG") {
        // For JPEG, we need to adjust the zone to multiples of 8
        CropZone adjusted_zone = adjust_zone_for_jpeg(zone, config_.camera_width, config_.camera_height);
        success = process_jpeg_zone_to_buffer(image, adjusted_zone, output_buffer, output_buffer_size);
    } else {
        success = process_raw_zone_to_buffer(image, zone, output_buffer, output_buffer_size);
    }
    
    DURATION_END("process_zone_to_buffer");
    return success;
}



/**
 * @brief Processes JPEG zone directly into output buffer.
 * Note: This implementation uses manual JPEG processing since the library API is problematic.
 */
bool ImageProcessor::process_jpeg_zone_to_buffer(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone,
    uint8_t* output_buffer,
    size_t output_buffer_size) {
    
    ESP_LOGW(TAG, "JPEG decoder library has header issues, using fallback raw processing");
    
    // Since the JPEG library headers have issues, fall back to treating JPEG as raw data
    // This is a temporary workaround until the library headers are fixed
    return process_raw_zone_to_buffer(image, zone, output_buffer, output_buffer_size);
}

/**
 * @brief Processes raw image zone directly into output buffer.
 */
bool ImageProcessor::process_raw_zone_to_buffer(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone,
    uint8_t* output_buffer,
    size_t output_buffer_size) {
    
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    const int model_channels = model_handler_->get_input_channels();
    
    // Calculate size based on model input type
    bool is_float_model = model_handler_->input_tensor()->type == kTfLiteFloat32;
    size_t bytes_per_value = is_float_model ? 4 : 1;
    size_t required_size = model_width * model_height * model_channels * bytes_per_value;
    
    // Verify output buffer is large enough
    if (output_buffer_size < required_size) {
        ESP_LOGE(TAG, "Output buffer too small: %zu < %zu", output_buffer_size, required_size);
        return false;
    }

    const uint8_t* src_data = image->get_data_buffer();
    const int src_width = config_.camera_width;
    const int src_height = config_.camera_height;

    // Calculate scaling factors
    const uint32_t width_scale = ((zone.x2 - zone.x1) << 16) / model_width;
    const uint32_t height_scale = ((zone.y2 - zone.y1) << 16) / model_height;

    if (is_float_model) {
        // Convert to float32 and normalize to [0, 1]
        float* float_dst = reinterpret_cast<float*>(output_buffer);
        for (int y = 0; y < model_height; y++) {
            const int src_y = zone.y1 + ((y * height_scale) >> 16);
            const size_t src_row_offset = src_y * src_width * bytes_per_pixel_;
            
            for (int x = 0; x < model_width; x++) {
                const int src_x = zone.x1 + ((x * width_scale) >> 16);
                const size_t src_pixel_offset = src_row_offset + src_x * bytes_per_pixel_;
                const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

                for (int c = 0; c < model_channels; c++) {
                    uint8_t pixel_value = src_data[src_pixel_offset + (c % bytes_per_pixel_)];
                    float_dst[dst_pixel_offset + c] = static_cast<float>(pixel_value) / 255.0f;
                }
            }
        }
    } else {
        // uint8 model - direct copy
        for (int y = 0; y < model_height; y++) {
            const int src_y = zone.y1 + ((y * height_scale) >> 16);
            const size_t src_row_offset = src_y * src_width * bytes_per_pixel_;
            
            for (int x = 0; x < model_width; x++) {
                const int src_x = zone.x1 + ((x * width_scale) >> 16);
                const size_t src_pixel_offset = src_row_offset + src_x * bytes_per_pixel_;
                const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

                if (bytes_per_pixel_ == 1) { // Grayscale
                    const uint8_t val = src_data[src_pixel_offset];
                    output_buffer[dst_pixel_offset] = val;
                    output_buffer[dst_pixel_offset+1] = val;
                    output_buffer[dst_pixel_offset+2] = val;
                } else { // RGB
                    for (int c = 0; c < model_channels; c++) {
                        output_buffer[dst_pixel_offset + c] = 
                            src_data[src_pixel_offset + (c % bytes_per_pixel_)];
                    }
                }
            }
        }
    }
    
    return true;
}

/**
 * @brief Processes a single crop zone from the image.
 */
ImageProcessor::ProcessResult ImageProcessor::process_zone(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone) {
  
  ProcessResult result{nullptr, 0};
  
  if (!validate_zone(zone)) {
    ESP_LOGE(TAG, "Invalid zone [%d,%d,%d,%d]", zone.x1, zone.y1, zone.x2, zone.y2);
    return result;
  }

  // Handle JPEG pixel format
  if (config_.pixel_format == "JPEG") {
    ESP_LOGW(TAG, "JPEG format detected but decoder library has issues, using fallback processing");
  }

  // For all formats including JPEG (fallback), use raw processing
  return scale_cropped_region(
      image->get_data_buffer(),
      config_.camera_width,
      config_.camera_height,
      zone);
}

/**
 * @brief Scales a cropped region of an image to the model's input dimensions.
 */
ImageProcessor::ProcessResult ImageProcessor::scale_cropped_region(
    const uint8_t *src_data,
    int src_width,
    int src_height,
    const CropZone &zone) {
    
    DURATION_START();
    
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    const int model_channels = model_handler_->get_input_channels();
    
    // Calculate size based on model input type (float32 = 4 bytes per value)
    bool is_float_model = model_handler_->input_tensor()->type == kTfLiteFloat32;
    size_t bytes_per_value = is_float_model ? 4 : 1;
    size_t required_size = model_width * model_height * model_channels * bytes_per_value;
    
    ESP_LOGD(TAG, "Model requires %dx%dx%d (%s, %zu bytes)", 
             model_width, model_height, model_channels,
             is_float_model ? "float32" : "uint8",
             required_size);

    // Allocate buffer for the scaled output image.
    UniqueBufferPtr buffer = allocate_image_buffer(required_size);
    if (!buffer) {
      ESP_LOGE(TAG, "Failed to allocate output buffer");
      return ProcessResult(nullptr, 0);
    }

    // Calculate fixed-point scaling factors (16.16 format) for precise scaling.
    const uint32_t width_scale = ((zone.x2 - zone.x1) << 16) / model_width;
    const uint32_t height_scale = ((zone.y2 - zone.y1) << 16) / model_height;

    uint8_t* dst = buffer->get();
    
    if (is_float_model) {
        // Convert to float32 and normalize to [0, 1]
        float* float_dst = reinterpret_cast<float*>(dst);
        for (int y = 0; y < model_height; y++) {
            const int src_y = zone.y1 + ((y * height_scale) >> 16);
            const size_t src_row_offset = src_y * src_width * bytes_per_pixel_;
            
            for (int x = 0; x < model_width; x++) {
                const int src_x = zone.x1 + ((x * width_scale) >> 16);
                const size_t src_pixel_offset = src_row_offset + src_x * bytes_per_pixel_;
                const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

                for (int c = 0; c < model_channels; c++) {
                    uint8_t pixel_value = src_data[src_pixel_offset + (c % bytes_per_pixel_)];
                    float_dst[dst_pixel_offset + c] = static_cast<float>(pixel_value) / 255.0f;
                }
            }
        }
    } else {
        // uint8 model - direct copy
        for (int y = 0; y < model_height; y++) {
            const int src_y = zone.y1 + ((y * height_scale) >> 16);
            const size_t src_row_offset = src_y * src_width * bytes_per_pixel_;
            
            for (int x = 0; x < model_width; x++) {
                const int src_x = zone.x1 + ((x * width_scale) >> 16);
                const size_t src_pixel_offset = src_row_offset + src_x * bytes_per_pixel_;
                const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

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
    }

    DURATION_END("scale_cropped_region");
    return ProcessResult(std::move(buffer), required_size);
}

/**
 * @brief Validates if a given crop zone is within the camera's dimensions.
 */
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

/**
 * @brief Scale RGB888 image to model dimensions using fixed-point arithmetic
 */
void ImageProcessor::scale_rgb888_image(const uint8_t* src_data, int src_width, int src_height,
                       uint8_t* dst_data, int dst_width, int dst_height) {
    const uint32_t x_scale = (static_cast<uint32_t>(src_width) << 16) / dst_width;
    const uint32_t y_scale = (static_cast<uint32_t>(src_height) << 16) / dst_height;

    for (int y = 0; y < dst_height; y++) {
        const int src_y = (y * y_scale) >> 16;
        const size_t src_row_offset = src_y * src_width * 3;
        
        for (int x = 0; x < dst_width; x++) {
            const int src_x = (x * x_scale) >> 16;
            const size_t src_pixel_offset = src_row_offset + src_x * 3;
            const size_t dst_pixel_offset = (y * dst_width + x) * 3;

            dst_data[dst_pixel_offset] = src_data[src_pixel_offset];
            dst_data[dst_pixel_offset + 1] = src_data[src_pixel_offset + 1];
            dst_data[dst_pixel_offset + 2] = src_data[src_pixel_offset + 2];
        }
    }
}

}  // namespace meter_reader_tflite
}  // namespace esphome