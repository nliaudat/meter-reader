/**
 * @file image_processor.cpp
 * @brief Implements the ImageProcessor class for handling image manipulation
 *        such as cropping, scaling, and format conversion for TensorFlow Lite.
 */

#include "image_processor.h"
#include "esp_log.h"
#include "debug_utils.h"
#include <algorithm>
#include <cstring>

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
    adjusted_width = std::max(8, adjusted_width); // Minimum 8 pixels
    
    // Adjust height to nearest multiple of 8
    int height = zone.y2 - zone.y1;
    int adjusted_height = (height + 4) / 8 * 8; // Round to nearest multiple of 8
    if (adjusted_height > height) {
        adjusted_height -= 8;
    }
    adjusted_height = std::max(8, adjusted_height); // Minimum 8 pixels
    
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
 * @brief Processes JPEG zone directly into output buffer using esp_new_jpeg decoder.
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

    // Get model input dimensions
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    
    // Verify zone dimensions are multiples of 8 (JPEG decoder requirement)
    int zone_width = zone.x2 - zone.x1;
    int zone_height = zone.y2 - zone.y1;
    if (zone_width % 8 != 0 || zone_height % 8 != 0) {
        ESP_LOGE(TAG, "JPEG zone dimensions must be multiples of 8, got %dx%d", 
                zone_width, zone_height);
        return false;
    }

    // Configure JPEG decoder for cropping and scaling
    jpeg_dec_config_t config = DEFAULT_JPEG_DEC_CONFIG();
    config.output_type = JPEG_PIXEL_FORMAT_RGB888;
    config.scale.width = static_cast<uint16_t>(model_width);
    config.scale.height = static_cast<uint16_t>(model_height);
    config.clipper.width = static_cast<uint16_t>(zone.x2 - zone.x1);
    config.clipper.height = static_cast<uint16_t>(zone.y2 - zone.y1);
    config.rotate = JPEG_ROTATE_0D;
    config.block_enable = false;

    jpeg_dec_handle_t decoder = nullptr;
    jpeg_error_t ret = jpeg_dec_open(&config, &decoder);
    
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to open JPEG decoder: %d", ret);
        return false;
    }

    // Prepare IO control
    jpeg_dec_io_t io;
    memset(&io, 0, sizeof(io));
    io.inbuf = const_cast<uint8_t*>(jpeg_data);
    io.inbuf_len = jpeg_size;
    io.inbuf_remain = jpeg_size;
    io.outbuf = output_buffer;
    io.out_size = output_buffer_size;

    // Parse header
    jpeg_dec_header_info_t header_info;
    ret = jpeg_dec_parse_header(decoder, &io, &header_info);
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to parse JPEG header: %d", ret);
        jpeg_dec_close(decoder);
        return false;
    }

    // Verify output buffer size is sufficient
    int required_size = model_width * model_height * 3; // RGB888
    if (output_buffer_size < static_cast<size_t>(required_size)) {
        ESP_LOGE(TAG, "Output buffer too small: %zu < %d", output_buffer_size, required_size);
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

    // Adjust zone to meet JPEG decoder requirements (multiples of 8)
    CropZone adjusted_zone = adjust_zone_for_jpeg(zone, config_.camera_width, config_.camera_height);
    
    // Get model input dimensions
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    
    // Configure JPEG decoder with cropping and scaling
    jpeg_dec_config_t config = DEFAULT_JPEG_DEC_CONFIG();
    config.output_type = JPEG_PIXEL_FORMAT_RGB888;
    config.scale.width = static_cast<uint16_t>(model_width);
    config.scale.height = static_cast<uint16_t>(model_height);
    config.clipper.width = static_cast<uint16_t>(adjusted_zone.x2 - adjusted_zone.x1);
    config.clipper.height = static_cast<uint16_t>(adjusted_zone.y2 - adjusted_zone.y1);
    config.rotate = JPEG_ROTATE_0D;
    config.block_enable = false;

    jpeg_dec_handle_t decoder = nullptr;
    jpeg_error_t ret = jpeg_dec_open(&config, &decoder);
    
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to open JPEG decoder: %d", ret);
        return result;
    }

    jpeg_dec_io_t io;
    memset(&io, 0, sizeof(io));
    io.inbuf = const_cast<uint8_t*>(jpeg_data);
    io.inbuf_len = jpeg_size;
    io.inbuf_remain = jpeg_size;
    io.outbuf = nullptr;
    io.out_size = 0;

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

    // Verify output size matches expected model input
    const size_t expected_size = model_width * model_height * 3; // RGB888
    if (outbuf_len != expected_size) {
        ESP_LOGE(TAG, "Output buffer size mismatch: expected %zu, got %d", expected_size, outbuf_len);
        jpeg_dec_close(decoder);
        return result;
    }

    // Allocate aligned buffer for the decoded image
    uint8_t *decoded_buf = (uint8_t*)jpeg_calloc_align(outbuf_len, 16);
    if (!decoded_buf) {
        ESP_LOGE(TAG, "Failed to allocate output buffer");
        jpeg_dec_close(decoder);
        return result;
    }

    io.outbuf = decoded_buf;
    io.out_size = outbuf_len;

    ret = jpeg_dec_process(decoder, &io);
    jpeg_dec_close(decoder);

    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "JPEG decoding failed: %d", ret);
        jpeg_free_align(decoded_buf);
        return result;
    }

    // Create a custom deleter for the buffer
    auto final_deleter = [](uint8_t* p) {
        if (p) jpeg_free_align(p);
    };
    
    std::unique_ptr<uint8_t[], decltype(final_deleter)> final_buffer(decoded_buf, final_deleter);
    
    return ProcessResult(
        std::unique_ptr<TrackedBuffer>(
            new TrackedBuffer(final_buffer.release(), false, true)), // jpeg_aligned = true
        outbuf_len);
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