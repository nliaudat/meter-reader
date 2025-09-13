/**
 * @file image_processor.cpp
 * @brief Implements the ImageProcessor class for handling image manipulation
 *        such as cropping, scaling, and format conversion for TensorFlow Lite.
 */

// Prevent old JPEG headers from being included
#define ESP_JPEG_DEC_H
#define ESP_JPEG_COMMON_H

// Include esp_new_jpeg FIRST, before any ESPHome camera headers
#include <cstdint> //// for uint8_t, uint16_t (needed by esp new jpeg)
#include "esp_jpeg_dec.h"
#include "esp_jpeg_common.h"

// Then include other headers
#include "image_processor.h"
#include "esp_log.h"
#include "debug_utils.h"
#include <algorithm>
#include <cstring>


namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "ImageProcessor";

// JPEG error code to string conversion
const char* ImageProcessor::jpeg_error_to_string(jpeg_error_t error) const {
    switch(error) {
        case JPEG_ERR_OK:            return "OK";
        case JPEG_ERR_FAIL:          return "Device error or wrong termination of input stream";
        case JPEG_ERR_NO_MEM:        return "Insufficient memory for the image";
        case JPEG_ERR_NO_MORE_DATA:  return "Input data is not enough";
        case JPEG_ERR_INVALID_PARAM: return "Parameter error";
        case JPEG_ERR_BAD_DATA:      return "Data format error (may be damaged data)";
        case JPEG_ERR_UNSUPPORT_FMT: return "Right format but not supported";
        case JPEG_ERR_UNSUPPORT_STD: return "Not supported JPEG standard";
        default:                     return "Unknown error";
    }
}

/**
 * @brief Validates buffer size for the given context.
 */
bool ImageProcessor::validate_buffer_size(size_t required, size_t available, const char* context) const {
    if (available < required) {
        ESP_LOGE(TAG, "Buffer too small for %s: need %zu bytes, have %zu bytes", 
                context, required, available);
        return false;
    }
    return true;
}

/**
 * @brief Validates input image for processing.
 */
bool ImageProcessor::validate_input_image(std::shared_ptr<camera::CameraImage> image) const {
    if (!image) {
        ESP_LOGE(TAG, "Null image pointer");
        return false;
    }
    
    if (!image->get_data_buffer()) {
        ESP_LOGE(TAG, "Image data buffer is null");
        return false;
    }
    
    if (image->get_data_length() == 0) {
        ESP_LOGE(TAG, "Image data length is zero");
        return false;
    }
    
    // Additional format-specific validation
    if (config_.pixel_format == "JPEG") {
        // Basic JPEG signature check
        if (image->get_data_length() < 2) {
            ESP_LOGE(TAG, "JPEG data too short");
            return false;
        }
        
        const uint8_t* data = image->get_data_buffer();
        if (data[0] != 0xFF || data[1] != 0xD8) {
            ESP_LOGW(TAG, "Invalid JPEG signature: 0x%02X 0x%02X", data[0], data[1]);
            // Still try to process, but warn
        }
    }
    
    return true;
}

/**
 * @brief Adjusts crop zone to nearest multiple of 8 for JPEG decoder requirements.
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
 * @brief Constructs an ImageProcessor object.
 */
ImageProcessor::ImageProcessor(const ImageProcessorConfig &config, 
                             ModelHandler* model_handler)
  : config_(config), model_handler_(model_handler) {
  
  if (!config_.validate()) {
    ESP_LOGE(TAG, "Invalid image processor configuration");
    ESP_LOGE(TAG, "  Camera: %dx%d, Format: %s", 
             config_.camera_width, config_.camera_height, config_.pixel_format.c_str());
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
 */
std::vector<ImageProcessor::ProcessResult> ImageProcessor::split_image_in_zone(
    std::shared_ptr<camera::CameraImage> image,
    const std::vector<CropZone> &zones) {
  
  std::lock_guard<std::mutex> lock(processing_mutex_);
  std::vector<ProcessResult> results;
  
  stats_.total_frames++;
  uint32_t start_time = millis();
  
  ESP_LOGD(TAG, "Starting image processing");
  ESP_LOGD(TAG, "Input image size: %zu bytes", image->get_data_length());

  // Validate input image
  if (!validate_input_image(image)) {
    ESP_LOGE(TAG, "Invalid input image");
    stats_.failed_frames++;
    return results;
  }

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
      stats_.failed_frames++;
    }
  } else {
    // Process each specified crop zone.
    ESP_LOGD(TAG, "Processing %d crop zones", zones.size());
    bool all_zones_successful = true;
    
    for (size_t i = 0; i < zones.size(); i++) {
      ESP_LOGD(TAG, "Processing zone %d: [%d,%d,%d,%d]", 
              i+1, zones[i].x1, zones[i].y1, zones[i].x2, zones[i].y2);
      
      ProcessResult result = process_zone(image, zones[i]);
      if (result.data) {
        ESP_LOGD(TAG, "Zone %d processed successfully (%zu bytes)", i+1, result.size);
        results.push_back(std::move(result));
      } else {
        ESP_LOGE(TAG, "Failed to process zone %d", i+1);
        all_zones_successful = false;
      }
    }
    
    if (!all_zones_successful) {
      stats_.failed_frames++;
    }
  }
  
  // Update statistics
  uint32_t processing_time = millis() - start_time;
  stats_.total_processing_time_ms += processing_time;
  
  ESP_LOGD(TAG, "Image processing complete. Generated %d outputs in %lums", 
           results.size(), processing_time);
  ESP_LOGD(TAG, "Success rate: %.1f%%, Avg time: %.1fms", 
           stats_.get_success_rate(), stats_.get_avg_processing_time());
  
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
    
    std::lock_guard<std::mutex> lock(processing_mutex_);
    DURATION_START();
    
    if (!validate_input_image(image)) {
        ESP_LOGE(TAG, "Invalid input image");
        return false;
    }
    
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
 * @brief Processes JPEG zone by decoding the full image first, then cropping and scaling to model's input type
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

    // Get model input dimensions and type
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    const int model_channels = model_handler_->get_input_channels();
    
    // Get model input tensor to determine expected data type and size
    TfLiteTensor* input_tensor = model_handler_->input_tensor();
    if (!input_tensor) {
        ESP_LOGE(TAG, "Cannot determine model input type");
        return false;
    }
    
    TfLiteType input_type = input_tensor->type;
    size_t required_size = 0;
    
    if (input_type == kTfLiteFloat32) {
        required_size = model_width * model_height * model_channels * sizeof(float);
    } else if (input_type == kTfLiteUInt8) {
        required_size = model_width * model_height * model_channels;
    } else {
        ESP_LOGE(TAG, "Unsupported model input type: %d", input_type);
        return false;
    }
    
    // Verify output buffer size is sufficient
    if (!validate_buffer_size(required_size, output_buffer_size, "JPEG processing")) {
        ESP_LOGE(TAG, "Need %zu bytes for %s output, got %zu bytes", 
                 required_size, 
                 input_type == kTfLiteFloat32 ? "float32" : "uint8",
                 output_buffer_size);
        return false;
    }

    // Step 1: Decode the full JPEG image to RGB888
    ESP_LOGD(TAG, "Decoding full JPEG image first for %s input", 
             input_type == kTfLiteFloat32 ? "float32" : "uint8");
    
    jpeg_dec_config_t decode_config = DEFAULT_JPEG_DEC_CONFIG();
    decode_config.output_type = JPEG_PIXEL_FORMAT_RGB888;
    decode_config.scale.width = static_cast<uint16_t>(config_.camera_width);
    decode_config.scale.height = static_cast<uint16_t>(config_.camera_height);
    decode_config.clipper.width = static_cast<uint16_t>(config_.camera_width);
    decode_config.clipper.height = static_cast<uint16_t>(config_.camera_height);
    decode_config.rotate = JPEG_ROTATE_0D;
    decode_config.block_enable = false;

    jpeg_dec_handle_t decoder = nullptr;
    jpeg_error_t ret = jpeg_dec_open(&decode_config, &decoder);
    
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to open JPEG decoder: %s", jpeg_error_to_string(ret));
        stats_.jpeg_decoding_errors++;
        return false;
    }

    // Allocate 16-byte aligned buffer for full decoded image using jpeg_calloc_align
    size_t full_image_size = config_.camera_width * config_.camera_height * 3;
    uint8_t* full_image_buf = (uint8_t*)jpeg_calloc_align(full_image_size, 16);
    if (!full_image_buf) {
        ESP_LOGE(TAG, "Failed to allocate full image buffer with jpeg_calloc_align");
        jpeg_dec_close(decoder);
        return false;
    }

    // Prepare IO control for full decoding
    jpeg_dec_io_t io;
    memset(&io, 0, sizeof(io));
    io.inbuf = const_cast<uint8_t*>(jpeg_data);
    io.inbuf_len = jpeg_size;
    io.inbuf_remain = jpeg_size;
    io.outbuf = full_image_buf;
    io.out_size = full_image_size;

    // Parse header and decode full image
    jpeg_dec_header_info_t header_info;
    ret = jpeg_dec_parse_header(decoder, &io, &header_info);
    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to parse JPEG header: %s", jpeg_error_to_string(ret));
        jpeg_free_align(full_image_buf);
        jpeg_dec_close(decoder);
        stats_.jpeg_decoding_errors++;
        return false;
    }

    // Decode the full image
    ret = jpeg_dec_process(decoder, &io);
    jpeg_dec_close(decoder);

    if (ret != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "JPEG decoding failed: %s", jpeg_error_to_string(ret));
        jpeg_free_align(full_image_buf);
        stats_.jpeg_decoding_errors++;
        return false;
    }

    ESP_LOGD(TAG, "Full JPEG decoded successfully to %dx%d", config_.camera_width, config_.camera_height);

    // Step 2: Crop and scale from the full decoded image to model's expected type
    bool success = process_raw_zone_to_buffer_from_rgb(
        full_image_buf, 
        config_.camera_width, 
        config_.camera_height,
        zone, 
        output_buffer, 
        output_buffer_size
    );

    // Free the aligned buffer
    jpeg_free_align(full_image_buf);

    return success;
}


/**
 * @brief Process raw zone from already decoded RGB data and convert to model's expected input type
 */
bool ImageProcessor::process_raw_zone_to_buffer_from_rgb(
    const uint8_t* rgb_data,
    int src_width,
    int src_height,
    const CropZone &zone,
    uint8_t* output_buffer,
    size_t output_buffer_size) {
    
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    const int model_channels = model_handler_->get_input_channels();
    
    // Get model input tensor to determine expected data type
    TfLiteTensor* input_tensor = model_handler_->input_tensor();
    if (!input_tensor) {
        ESP_LOGE(TAG, "Cannot determine model input type - input tensor is null");
        return false;
    }
    
    TfLiteType input_type = input_tensor->type;
    size_t required_size = 0;
    
    if (input_type == kTfLiteFloat32) {
        // Float32 model: 4 bytes per value
        required_size = model_width * model_height * model_channels * sizeof(float);
        ESP_LOGD(TAG, "Model expects float32 input (%zu bytes)", required_size);
    } else if (input_type == kTfLiteUInt8) {
        // UInt8 model: 1 byte per value
        required_size = model_width * model_height * model_channels;
        ESP_LOGD(TAG, "Model expects uint8 input (%zu bytes)", required_size);
    } else {
        ESP_LOGE(TAG, "Unsupported model input type: %d", input_type);
        return false;
    }
    
    // Verify output buffer is large enough
    if (!validate_buffer_size(required_size, output_buffer_size, "model input processing")) {
        ESP_LOGE(TAG, "Need %zu bytes for output, got %zu bytes", required_size, output_buffer_size);
        return false;
    }

    // Validate the crop zone
    if (zone.x1 < 0 || zone.y1 < 0 || 
        zone.x2 > src_width || zone.y2 > src_height ||
        zone.x1 >= zone.x2 || zone.y1 >= zone.y2) {
        ESP_LOGE(TAG, "Invalid crop zone for source image %dx%d: [%d,%d,%d,%d]", 
                 src_width, src_height, zone.x1, zone.y1, zone.x2, zone.y2);
        return false;
    }

    // Get model configuration to check if normalization is needed
    const ModelConfig& config = model_handler_->get_config();
    bool normalize = config.normalize;
    
    ESP_LOGD(TAG, "Converting RGB to model input type %d (normalize: %s)", 
             input_type, normalize ? "true" : "false");

    // Calculate scaling factors (fixed-point 16.16 format)
    const int crop_width = zone.x2 - zone.x1;
    const int crop_height = zone.y2 - zone.y1;
    const uint32_t width_scale = (crop_width << 16) / model_width;
    const uint32_t height_scale = (crop_height << 16) / model_height;

    if (input_type == kTfLiteFloat32) {
        // Convert to float32
        float* float_output = reinterpret_cast<float*>(output_buffer);
        
        for (int y = 0; y < model_height; y++) {
            const int src_y = zone.y1 + ((y * height_scale) >> 16);
            const size_t src_row_offset = src_y * src_width * 3; // RGB888 has 3 bytes per pixel
            
            for (int x = 0; x < model_width; x++) {
                const int src_x = zone.x1 + ((x * width_scale) >> 16);
                const size_t src_pixel_offset = src_row_offset + src_x * 3;
                const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

                // Convert RGB values to float32
                for (int c = 0; c < model_channels && c < 3; c++) {
                    uint8_t pixel_value = rgb_data[src_pixel_offset + c];
                    
                    if (normalize) {
                        // Normalize uint8 [0,255] to float32 [0,1]
                        float_output[dst_pixel_offset + c] = static_cast<float>(pixel_value) / 255.0f;
                    } else {
                        // Direct conversion (if model expects 0-255 range)
                        float_output[dst_pixel_offset + c] = static_cast<float>(pixel_value);
                    }
                }
                
                // Handle cases where model expects more channels than available
                for (int c = 3; c < model_channels; c++) {
                    float_output[dst_pixel_offset + c] = 0.0f; // Pad with zeros
                }
            }
        }
        
        // Debug: log first few values to verify conversion
        ESP_LOGD(TAG, "First 5 float32 values:");
        for (int i = 0; i < 5 && i < required_size / sizeof(float); i++) {
            ESP_LOGD(TAG, "  [%d]: %.4f", i, float_output[i]);
        }
        
    } else if (input_type == kTfLiteUInt8) {
        // Convert to uint8 (direct copy with possible quantization handling)
        for (int y = 0; y < model_height; y++) {
            const int src_y = zone.y1 + ((y * height_scale) >> 16);
            const size_t src_row_offset = src_y * src_width * 3; // RGB888 has 3 bytes per pixel
            
            for (int x = 0; x < model_width; x++) {
                const int src_x = zone.x1 + ((x * width_scale) >> 16);
                const size_t src_pixel_offset = src_row_offset + src_x * 3;
                const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

                // Copy RGB values directly
                for (int c = 0; c < model_channels && c < 3; c++) {
                    output_buffer[dst_pixel_offset + c] = rgb_data[src_pixel_offset + c];
                }
                
                // Handle cases where model expects more channels than available
                for (int c = 3; c < model_channels; c++) {
                    output_buffer[dst_pixel_offset + c] = 0; // Pad with zeros
                }
            }
        }
        
        // Debug: log first few values to verify conversion
        ESP_LOGD(TAG, "First 5 uint8 values:");
        for (int i = 0; i < 5 && i < required_size; i++) {
            ESP_LOGD(TAG, "  [%d]: %u", i, output_buffer[i]);
        }
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
    if (!validate_buffer_size(required_size, output_buffer_size, "raw image processing")) {
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

  // Handle JPEG pixel format
  if (config_.pixel_format == "JPEG") {
    ESP_LOGD(TAG, "Decoding JPEG for zone [%d,%d,%d,%d]", zone.x1, zone.y1, zone.x2, zone.y2);
    
    // Get model input dimensions and type
    const int model_width = model_handler_->get_input_width();
    const int model_height = model_handler_->get_input_height();
    const int model_channels = model_handler_->get_input_channels();
    
    // Get model input tensor to determine expected data type and size
    TfLiteTensor* input_tensor = model_handler_->input_tensor();
    if (!input_tensor) {
        ESP_LOGE(TAG, "Cannot determine model input type");
        return result;
    }
    
    TfLiteType input_type = input_tensor->type;
    size_t required_size = 0;
    
    if (input_type == kTfLiteFloat32) {
        required_size = model_width * model_height * model_channels * sizeof(float);
    } else if (input_type == kTfLiteUInt8) {
        required_size = model_width * model_height * model_channels;
    } else {
        ESP_LOGE(TAG, "Unsupported model input type: %d", input_type);
        return result;
    }
    
    ESP_LOGD(TAG, "Allocating %zu bytes for %s input", 
             required_size, 
             input_type == kTfLiteFloat32 ? "float32" : "uint8");
    
    // Allocate output buffer with correct size
    UniqueBufferPtr buffer = allocate_image_buffer(required_size);
    if (!buffer) {
      ESP_LOGE(TAG, "Failed to allocate output buffer (%zu bytes)", required_size);
      return result;
    }
    
    // Use the new approach: decode full image first, then crop and scale
    if (process_jpeg_zone_to_buffer(image, zone, buffer->get(), required_size)) {
      return ProcessResult(std::move(buffer), required_size);
    } else {
      ESP_LOGE(TAG, "JPEG processing failed");
      return result;
    }
  }

  // For non-JPEG formats, use the existing approach
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
    
    // Get model input tensor to determine expected data type
    TfLiteTensor* input_tensor = model_handler_->input_tensor();
    if (!input_tensor) {
        ESP_LOGE(TAG, "Cannot determine model input type");
        return ProcessResult(nullptr, 0);
    }
    
    TfLiteType input_type = input_tensor->type;
    size_t required_size = 0;
    
    if (input_type == kTfLiteFloat32) {
        required_size = model_width * model_height * model_channels * sizeof(float);
    } else if (input_type == kTfLiteUInt8) {
        required_size = model_width * model_height * model_channels;
    } else {
        ESP_LOGE(TAG, "Unsupported model input type: %d", input_type);
        return ProcessResult(nullptr, 0);
    }
    
    ESP_LOGD(TAG, "Model requires %dx%dx%d %s (%zu bytes)", 
             model_width, model_height, model_channels,
             input_type == kTfLiteFloat32 ? "float32" : "uint8",
             required_size);    

    // Allocate buffer for the scaled output image.
    UniqueBufferPtr buffer = allocate_image_buffer(required_size);
    if (!buffer) {
      ESP_LOGE(TAG, "Failed to allocate output buffer (%zu bytes)", required_size);
      stats_.memory_allocation_errors++;
      return ProcessResult(nullptr, 0);
    }

    // Get model configuration
    const ModelConfig& config = model_handler_->get_config();
    bool normalize = config.normalize;

    // Calculate fixed-point scaling factors (16.16 format) for precise scaling.
    const uint32_t width_scale = ((zone.x2 - zone.x1) << 16) / model_width;
    const uint32_t height_scale = ((zone.y2 - zone.y1) << 16) / model_height;

    if (input_type == kTfLiteFloat32) {
        // Convert to float32
        float* dst = reinterpret_cast<float*>(buffer->get());
        
        for (int y = 0; y < model_height; y++) {
            const int src_y = zone.y1 + ((y * height_scale) >> 16);
            const size_t src_row_offset = src_y * src_width * bytes_per_pixel_;
            
            for (int x = 0; x < model_width; x++) {
                const int src_x = zone.x1 + ((x * width_scale) >> 16);
                const size_t src_pixel_offset = src_row_offset + src_x * bytes_per_pixel_;
                const size_t dst_pixel_offset = (y * model_width + x) * model_channels;

                if (bytes_per_pixel_ == 1) { // Grayscale
                    const uint8_t val = src_data[src_pixel_offset];
                    if (normalize) {
                        dst[dst_pixel_offset] = static_cast<float>(val) / 255.0f;
                        dst[dst_pixel_offset+1] = static_cast<float>(val) / 255.0f;
                        dst[dst_pixel_offset+2] = static_cast<float>(val) / 255.0f;
                    } else {
                        dst[dst_pixel_offset] = static_cast<float>(val);
                        dst[dst_pixel_offset+1] = static_cast<float>(val);
                        dst[dst_pixel_offset+2] = static_cast<float>(val);
                    }
                } else { // RGB
                    for (int c = 0; c < model_channels; c++) {
                        uint8_t pixel_value = src_data[src_pixel_offset + (c % bytes_per_pixel_)];
                        if (normalize) {
                            dst[dst_pixel_offset + c] = static_cast<float>(pixel_value) / 255.0f;
                        } else {
                            dst[dst_pixel_offset + c] = static_cast<float>(pixel_value);
                        }
                    }
                }
            }
        }
    } else if (input_type == kTfLiteUInt8) {
        // Convert to uint8
        uint8_t* dst = buffer->get();
        
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
  // Check if camera dimensions are valid
  if (config_.camera_width <= 0 || config_.camera_height <= 0) {
    ESP_LOGE(TAG, "Invalid camera dimensions: %dx%d", config_.camera_width, config_.camera_height);
    return false;
  }
  
  if (zone.x1 < 0 || zone.y1 < 0 || 
      zone.x2 > config_.camera_width || zone.y2 > config_.camera_height ||
      zone.x1 >= zone.x2 || zone.y1 >= zone.y2) {
    ESP_LOGE(TAG, "Invalid crop zone [%d,%d,%d,%d] (camera: %dx%d)", 
             zone.x1, zone.y1, zone.x2, zone.y2,
             config_.camera_width, config_.camera_height);
    return false;
  }
  
  // Check minimum size requirements
  int width = zone.x2 - zone.x1;
  int height = zone.y2 - zone.y1;
  
  if (width < 8 || height < 8) {
    ESP_LOGE(TAG, "Crop zone too small: %dx%d (minimum 8x8)", width, height);
    return false;
  }
  
  ESP_LOGD(TAG, "Valid crop zone [%d,%d,%d,%d] (%dx%d)", 
           zone.x1, zone.y1, zone.x2, zone.y2, width, height);
  return true;
}




}  // namespace meter_reader_tflite
}  // namespace esphome