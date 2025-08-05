#pragma once

#include "esphome/components/camera/camera.h"
#include "crop_zones.h"
#include "model_handler.h"
// #include "debug_utils.h"
#include <memory>
#include <vector>
#include <string>

namespace esphome {
namespace meter_reader_tflite {

struct ImageProcessorConfig {
  int camera_width;
  int camera_height;
  std::string pixel_format;
  
  bool validate() const {
    return camera_width > 0 && camera_height > 0 && !pixel_format.empty();
  }
};

class ImageProcessor {
 public:
  struct ProcessResult {
    std::unique_ptr<uint8_t[]> data;
    size_t size;
  };

  ImageProcessor(const ImageProcessorConfig &config, ModelHandler* model_handler);

  std::vector<ProcessResult> split_image_in_zone(
      std::shared_ptr<camera::CameraImage> image,
      const std::vector<CropZone> &zones = {});

 private:
  ProcessResult process_zone(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone);

  ProcessResult scale_cropped_region(
      const uint8_t *src_data,
      int src_width,
      int src_height,
      const CropZone &zone);

  bool validate_zone(const CropZone &zone) const;

  ImageProcessorConfig config_;
  ModelHandler* model_handler_;
  int bytes_per_pixel_;
};

}  // namespace meter_reader_tflite
}  // namespace esphome