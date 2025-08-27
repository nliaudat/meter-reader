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
    // Simple wrapper that tracks allocation source
    class TrackedBuffer {
    public:
        TrackedBuffer(uint8_t* ptr, bool is_spiram) : ptr_(ptr), is_spiram_(is_spiram) {}
        ~TrackedBuffer() {
            if (ptr_) {
                if (is_spiram_) {
                    heap_caps_free(ptr_);
                } else {
                    delete[] ptr_;
                }
            }
        }
        
        uint8_t* get() { return ptr_; }
        const uint8_t* get() const { return ptr_; }
        uint8_t& operator[](size_t idx) { return ptr_[idx]; }
        const uint8_t& operator[](size_t idx) const { return ptr_[idx]; }
        
    private:
        uint8_t* ptr_;
        bool is_spiram_;
    };

    using UniqueBufferPtr = std::unique_ptr<TrackedBuffer>;

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
  bool jpeg_to_rgb888(const uint8_t* src, size_t src_len, uint8_t* dst);

  UniqueBufferPtr allocate_image_buffer(size_t size) {
      uint8_t* buf = nullptr;
      bool is_spiram = false;
      
      // Try to allocate from SPIRAM first
      #ifdef CONFIG_SPIRAM
      if (heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >= size) {
          buf = (uint8_t*)heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
          is_spiram = (buf != nullptr);
      }
      #endif
      
      // Fall back to internal RAM if SPIRAM allocation failed or is not available
      if (!buf) {
          buf = new (std::nothrow) uint8_t[size];
          is_spiram = false;
      }
      
      if (buf) {
          return std::unique_ptr<TrackedBuffer>(new TrackedBuffer(buf, is_spiram));
      }
      
      return nullptr;
  }

  ImageProcessorConfig config_;
  ModelHandler* model_handler_;
  int bytes_per_pixel_;
};

}  // namespace meter_reader_tflite
}  // namespace esphome