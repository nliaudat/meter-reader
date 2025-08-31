/**
 * @file image_processor.cpp
 * @brief Implements the ImageProcessor class for handling image manipulation
 *        such as cropping, scaling, and format conversion for TensorFlow Lite.
 */

#include "image_processor.h"
#include "esp_log.h"
#include "debug_utils.h"
#include <algorithm>

// Include JPEG headers
// #include "esp_jpeg_dec.h"
// #include "esp_jpeg_common.h"

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
  } else {
    // Process each specified crop zone.
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
        success = process_jpeg_zone_to_buffer(image, zone, output_buffer, output_buffer_size);
    } else {
        success = process_raw_zone_to_buffer(image, zone, output_buffer, output_buffer_size);
    }
    
    DURATION_END("process_zone_to_buffer");
    return success;
}

/**
 * @brief Processes JPEG zone directly into output buffer.
 */
bool ImageProcessor::process_jpeg_zone_to_buffer(
    std::shared_ptr<camera::CameraImage> image,
    const CropZone &zone,
    uint8_t* output_buffer,
    size_t output_buffer_size) {
    
    const uint8_t *jpeg_data = image->get_data_buffer();
    size_t jpeg_size = image->get_data_length();
    
    if (!jpeg_data || jpeg_size == 0) {
        ESP_LOGE(TAG, "Invalid JPEG data");
        return false;
    }

    // Configure JPEG decoder for single-step cropping and scaling
    jpeg_dec_config_t config = {
        .output_type = JPEG_PIXEL_FORMAT_RGB888,
        .scale = {
            .width = static_cast<uint16_t>(model_handler_->get_input_width()),
            .height = static_cast<uint16_t>(model_handler_->get_input_height())
        },
        .clipper = {
            .width = static_cast<uint16_t>(zone.x2 - zone.x1),
            .height = static_cast<uint16_t>(zone.y2 - zone.y1)
        },
        .rotate = JPEG_ROTATE_0D,
        .block_enable = false
    };

    jpeg_dec_handle_t decoder = nullptr;
    jpeg_error_t ret = jpeg_dec_open(&config, &decoder);
    
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to open JPEG decoder: %d", ret);
        return false;
    }

    // Prepare IO control
    jpeg_dec_io_t io = {
        .inbuf = const_cast<uint8_t*>(jpeg_data),
        .inbuf_len = jpeg_size,
        .inbuf_remain = jpeg_size,
        .outbuf = output_buffer,
        .out_size = output_buffer_size
    };

    // Parse header
    jpeg_dec_header_info_t header_info;
    ret = jpeg_dec_parse_header(decoder, &io, &header_info);
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to parse JPEG header: %d", ret);
        jpeg_dec_close(decoder);
        return false;
    }

    // Direct decode to output buffer
    ret = jpeg_dec_process(decoder, &io);
    jpeg_dec_close(decoder);

    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "JPEG decoding failed: %d", ret);
        return false;
    }

    return true;
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
    
    const uint8_t* src_data = image->get_data_buffer();
    const int src_width = config_.camera_width;
    const int src_height = config_.camera_height;
    
    // Verify output buffer is large enough
    const size_t required_size = model_width * model_height * model_channels;
    if (output_buffer_size < required_size) {
        ESP_LOGE(TAG, "Output buffer too small: %zu < %zu", output_buffer_size, required_size);
        return false;
    }

    // Calculate scaling factors
    const uint32_t width_scale = ((zone.x2 - zone.x1) << 16) / model_width;
    const uint32_t height_scale = ((zone.y2 - zone.y1) << 16) / model_height;

    // Process directly into output buffer
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

  // Handle JPEG pixel format using esp_new_jpeg library
  if (config_.pixel_format == "JPEG") {
    ESP_LOGD(TAG, "Decoding JPEG for zone [%d,%d,%d,%d]", zone.x1, zone.y1, zone.x2, zone.y2);
    const uint8_t *jpeg_data = image->get_data_buffer();
    size_t jpeg_size = image->get_data_length();
    
    if (!jpeg_data || jpeg_size == 0) {
      ESP_LOGE(TAG, "Invalid JPEG data");
      return result;
    }

    // Get model input dimensions
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    
    // Configure JPEG decoder - ONLY SCALING, NO CLIPPING
    // We'll handle cropping manually after decoding
    jpeg_dec_config_t config = {
        .output_type = JPEG_PIXEL_FORMAT_RGB888,
        .scale = {
            .width = static_cast<uint16_t>(model_width),
            .height = static_cast<uint16_t>(model_height)
        },
        .clipper = {
            .width = 0,  // NO CLIPPING - we'll crop after decoding
            .height = 0  // NO CLIPPING
        },
        .rotate = JPEG_ROTATE_0D,
        .block_enable = false
    };

    jpeg_dec_handle_t decoder = nullptr;
    jpeg_error_t ret = jpeg_dec_open(&config, &decoder);
    
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to open JPEG decoder: %d", ret);
        return result;
    }

    jpeg_dec_io_t io = {
        .inbuf = const_cast<uint8_t*>(jpeg_data),
        .inbuf_len = jpeg_size,
        .inbuf_remain = jpeg_size,
        .outbuf = nullptr,
        .out_size = 0
    };

    jpeg_dec_header_info_t header_info;
    ret = jpeg_dec_parse_header(decoder, &io, &header_info);
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to parse JPEG header: %d", ret);
        jpeg_dec_close(decoder);
        return result;
    }

    int outbuf_len = 0;
    ret = jpeg_dec_get_outbuf_len(decoder, &outbuf_len);
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to get output buffer size: %d", ret);
        jpeg_dec_close(decoder);
        return result;
    }

    // Allocate buffer for the full decoded image
    uint8_t *full_decoded_buf = (uint8_t*)heap_caps_malloc(outbuf_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!full_decoded_buf) {
        ESP_LOGE(TAG, "Failed to allocate output buffer");
        jpeg_dec_close(decoder);
        return result;
    }

    io.outbuf = full_decoded_buf;
    io.out_size = outbuf_len;

    ret = jpeg_dec_process(decoder, &io);
    jpeg_dec_close(decoder);

    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "JPEG decoding failed: %d", ret);
        heap_caps_free(full_decoded_buf);
        return result;
    }

    // Now manually crop the decoded image to the desired zone
    // Calculate crop coordinates in the decoded image space
    float scale_x = static_cast<float>(model_width) / config_.camera_width;
    float scale_y = static_cast<float>(model_height) / config_.camera_height;
    
    int crop_x = static_cast<int>(zone.x1 * scale_x);
    int crop_y = static_cast<int>(zone.y1 * scale_y);
    int crop_width = static_cast<int>((zone.x2 - zone.x1) * scale_x);
    int crop_height = static_cast<int>((zone.y2 - zone.y1) * scale_y);
    
    // Ensure crop region is within bounds
    crop_x = std::max(0, std::min(crop_x, model_width - 1));
    crop_y = std::max(0, std::min(crop_y, model_height - 1));
    crop_width = std::min(crop_width, model_width - crop_x);
    crop_height = std::min(crop_height, model_height - crop_y);

    // Allocate final output buffer (RGB888, 3 bytes per pixel)
    const size_t final_size = crop_width * crop_height * 3;
    uint8_t *final_buf = (uint8_t*)heap_caps_malloc(final_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!final_buf) {
        ESP_LOGE(TAG, "Failed to allocate final crop buffer");
        heap_caps_free(full_decoded_buf);
        return result;
    }

    // Perform manual cropping
    for (int y = 0; y < crop_height; y++) {
        const int src_y = crop_y + y;
        const size_t src_offset = src_y * model_width * 3;
        const size_t dst_offset = y * crop_width * 3;
        
        for (int x = 0; x < crop_width; x++) {
            const int src_x = crop_x + x;
            const size_t src_pixel = src_offset + src_x * 3;
            const size_t dst_pixel = dst_offset + x * 3;
            
            // Copy RGB pixels
            final_buf[dst_pixel] = full_decoded_buf[src_pixel];
            final_buf[dst_pixel + 1] = full_decoded_buf[src_pixel + 1];
            final_buf[dst_pixel + 2] = full_decoded_buf[src_pixel + 2];
        }
    }

    // Free the full decoded buffer
    heap_caps_free(full_decoded_buf);

    // Create a custom deleter for the final buffer
    auto final_deleter = [](uint8_t* p) {
        if (p) heap_caps_free(p);
    };
    
    std::unique_ptr<uint8_t[], decltype(final_deleter)> final_buffer(final_buf, final_deleter);
    
    return ProcessResult(
        std::unique_ptr<TrackedBuffer>(
            new TrackedBuffer(final_buffer.release(), true, false)),
        final_size);
  }

  // For non-JPEG formats
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
    const size_t required_size = model_width * model_height * model_channels;
    
    ESP_LOGD(TAG, "Model requires %dx%dx%d (%zu bytes)", model_width, model_height, model_channels, required_size);    

    // Allocate buffer for the scaled output image.
    UniqueBufferPtr buffer = allocate_image_buffer(required_size);
    if (!buffer) {
      ESP_LOGE(TAG, "Failed to allocate output buffer");
      return ProcessResult(nullptr, 0);
    }

    // Calculate fixed-point scaling factors (16.16 format) for precise scaling.
    const uint32_t width_scale = ((zone.x2 - zone.x1) << 16) / model_width;
    const uint32_t height_scale = ((zone.y2 - zone.y1) << 16) / model_height;

    uint8_t* dst = buffer->get(); // Destination buffer for the scaled image.
    
    // Optimized scaling loop to resize the image.
    for (int y = 0; y < model_height; y++) {
        const int src_y = zone.y1 + ((y * height_scale) >> 16); // Calculate source Y coordinate.
        const size_t src_row_offset = src_y * src_width * bytes_per_pixel_; // Offset to the source row.
        
        for (int x = 0; x < model_width; x++) {
            const int src_x = zone.x1 + ((x * width_scale) >> 16); // Calculate source X coordinate.
            const size_t src_pixel_offset = src_row_offset + src_x * bytes_per_pixel_; // Offset to the source pixel.
            const size_t dst_pixel_offset = (y * model_width + x) * model_channels; // Offset to the destination pixel.

            // Handle different pixel formats (Grayscale or RGB).
            if (bytes_per_pixel_ == 1) { // Grayscale
                const uint8_t val = src_data[src_pixel_offset];
                dst[dst_pixel_offset] = val;
                dst[dst_pixel_offset+1] = val;
                dst[dst_pixel_offset+2] = val; // Replicate grayscale to all channels for RGB model input.
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

}  // namespace meter_reader_tflite
}  // namespace esphome