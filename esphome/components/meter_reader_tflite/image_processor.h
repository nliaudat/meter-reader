#pragma once

#include "esphome/components/camera/camera.h"
#include "crop_zones.h"
#include "model_handler.h"
#include "esp_heap_caps.h"
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
    // Define the custom deleter first
    struct PSRAMDeleter {
        void operator()(uint8_t* p) const {
            if (p) {
                if (heap_caps_get_free_size(MALLOC_CAP_SPIRAM)) {
                    heap_caps_free(p);
                } else {
                    delete[] p;
                }
            }
        }
    };

    // Our custom unique_ptr type
    using UniqueBufferPtr = std::unique_ptr<uint8_t[], PSRAMDeleter>;

    struct ProcessResult {
        UniqueBufferPtr data;
        size_t size;
        
        ProcessResult() : data(nullptr), size(0) {}
        ProcessResult(UniqueBufferPtr&& ptr, size_t sz) 
            : data(std::move(ptr)), size(sz) {}
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
  bool validate_jpeg(const uint8_t* data, size_t size);
  bool jpeg_to_rgb888(const uint8_t* src, size_t src_len, uint8_t* dst, uint32_t width, uint32_t height);
  
    UniqueBufferPtr allocate_image_buffer(size_t size) {
        uint8_t* buf = (uint8_t*)heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!buf) {
            buf = new uint8_t[size];
        }
        return UniqueBufferPtr(buf);
    }

  ImageProcessorConfig config_;
  ModelHandler* model_handler_;
  int bytes_per_pixel_;
};

}  // namespace meter_reader_tflite
}  // namespace esphome