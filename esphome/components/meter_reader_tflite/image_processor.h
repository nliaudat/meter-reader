#pragma once

#include "esphome/components/camera/camera.h"
#include "crop_zones.h" 
#include <memory>
#include <vector>
#include <string>

namespace esphome {
namespace meter_reader_tflite {

struct ImageProcessorConfig {
  int camera_width;
  int camera_height;
  int model_input_width;
  int model_input_height;
  std::string pixel_format;
  
  bool validate() const {
    return camera_width > 0 && camera_height > 0 &&
           model_input_width > 0 && model_input_height > 0 &&
           (pixel_format == "RGB888" || pixel_format == "GRAYSCALE" ||
            pixel_format == "RGB565" || pixel_format == "YUV422" ||
            pixel_format == "YUV420" || pixel_format == "JPEG" ||
            pixel_format == "RAW" || pixel_format == "RGB444" ||
            pixel_format == "RGB555");
  }
};

class ImageProcessor {
 public:
  explicit ImageProcessor(const ImageProcessorConfig &config);
  
  struct ProcessResult {
    std::unique_ptr<uint8_t[]> data;
    size_t size;
  };
  
  std::vector<ProcessResult> process_image(
      std::shared_ptr<camera::CameraImage> image,
      const std::vector<CropZone> &zones);
  
  // ProcessResult process_zone(
      // std::shared_ptr<camera::CameraImage> image,
      // const CropZone &zone);

 protected:
  ProcessResult crop_and_resize(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone);

  ProcessResult decode_and_process_jpeg(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone);
      
  bool validate_zone(const CropZone &zone) const;

  ImageProcessorConfig config_;
  int bytes_per_pixel_;
  

protected:
  ProcessResult crop_and_resize_from_decoded(
      const uint8_t *decoded_data,
      int original_width,
      int original_height,
      const CropZone &zone);
  
};

}  // namespace meter_reader_tflite
}  // namespace esphome