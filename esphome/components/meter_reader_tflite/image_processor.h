#pragma once

#include "esphome/components/camera/camera.h"
#include "crop_zones.h"
#include "model_handler.h"
#include <memory>
#include <vector>
#include <string>

namespace esphome {
namespace meter_reader_tflite {
class ModelHandler;  // Forward declaration

struct ImageProcessorConfig {
  int camera_width;
  int camera_height;
  // int model_input_width; 
  // int model_input_height; 
  std::string pixel_format;
  
  bool validate() const {
    return camera_width > 0 && camera_height > 0 &&
           (pixel_format == "RGB888" || pixel_format == "GRAYSCALE" ||
            pixel_format == "RGB565" || pixel_format == "YUV422" ||
            pixel_format == "YUV420" || pixel_format == "JPEG" ||
            pixel_format == "RAW" || pixel_format == "RGB444" ||
            pixel_format == "RGB555");
  }
};

class ImageProcessor {
 public:
  // explicit ImageProcessor(const ImageProcessorConfig &config);
  // explicit ImageProcessor(const ImageProcessorConfig &config, esphome::meter_reader_tflite::ModelHandler* model_handler);
  explicit ImageProcessor(const ImageProcessorConfig &config, ModelHandler* model_handler);
  
    // get model dimensions
  int get_model_input_width() const { 
    return model_handler_ ? model_handler_->get_input_width() : 0; 
  }
  
  int get_model_input_height() const { 
    return model_handler_ ? model_handler_->get_input_height() : 0; 
  }
  
  int get_model_input_channels() const { 
    return model_handler_ ? model_handler_->get_input_channels() : 0; 
  }
  
  struct ProcessResult {
    std::unique_ptr<uint8_t[]> data;
    size_t size;
  };
  
  std::vector<ProcessResult> process_image(
      std::shared_ptr<camera::CameraImage> image,
      const std::vector<CropZone> &zones);

 protected:
  ProcessResult crop_image(
      const uint8_t *src_data,
      int src_width,
      int src_height,
      const CropZone &zone);
      
  ProcessResult resize_image(
      const uint8_t *src_data,
      int src_width,
      int src_height,
      int src_channels);

  ProcessResult process_zone(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone);

  ProcessResult process_zone_from_decoded(
      const uint8_t *decoded_data,
      int original_width,
      int original_height,
      const CropZone &zone);

#ifdef USE_JPEG
  ProcessResult decode_and_process_jpeg(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone);
#endif

  bool validate_zone(const CropZone &zone) const;

  ImageProcessorConfig config_;
  int bytes_per_pixel_;
  ModelHandler* model_handler_;
  
};

}  // namespace meter_reader_tflite
}  // namespace esphome