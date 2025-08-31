/**
 * @file image_processor.h
 * @brief Defines the ImageProcessor class for handling image manipulation
 *        such as cropping, scaling, and format conversion for TensorFlow Lite.
 */

#pragma once

#include "esphome/components/camera/camera.h"
#include "crop_zones.h"
#include "model_handler.h"
#include "esp_heap_caps.h"
#include <memory>
#include <vector>
#include <string>

#include "esp_jpeg_dec.h"
#include "esp_jpeg_common.h"



namespace esphome {
namespace meter_reader_tflite {

/**
 * @brief Configuration structure for the ImageProcessor.
 * Contains camera dimensions and pixel format.
 */
struct ImageProcessorConfig {
  int camera_width;  ///< Width of the camera image.
  int camera_height; ///< Height of the camera image.
  std::string pixel_format; ///< Pixel format of the camera image (e.g., "RGB888", "JPEG").
  
  /**
   * @brief Validates the configuration parameters.
   * @return True if the configuration is valid, false otherwise.
   */
  bool validate() const {
    return camera_width > 0 && camera_height > 0 && !pixel_format.empty();
  }
};

/**
 * @brief The ImageProcessor class handles image preprocessing tasks.
 * This includes splitting images into zones, decoding JPEGs, and scaling
 * regions to the model's input requirements.
 */
class ImageProcessor {
 public:
    /**
     * @brief A smart pointer wrapper for dynamically allocated buffers.
     * Manages memory allocated with `heap_caps_malloc` or `new[]`.
     */
    class TrackedBuffer {
    public:
        /**
         * @brief Constructs a TrackedBuffer.
         * @param ptr Pointer to the allocated memory.
         * @param is_spiram True if memory was allocated from SPIRAM, false otherwise.
         * @param is_jpeg_aligned True if memory was allocated with jpeg_calloc_align.
         */
        TrackedBuffer(uint8_t* ptr, bool is_spiram, bool is_jpeg_aligned = false) 
            : ptr_(ptr), is_spiram_(is_spiram), is_jpeg_aligned_(is_jpeg_aligned) {}
        
        /**
         * @brief Destructor. Frees the allocated memory.
         */
        ~TrackedBuffer() {
            if (ptr_) {
                if (is_jpeg_aligned_) {
                    jpeg_free_align(ptr_);
                } else if (is_spiram_) {
                    heap_caps_free(ptr_);
                } else {
                    delete[] ptr_;
                }
            }
        }
        
        /**
         * @brief Gets the raw pointer to the buffer.
         * @return Raw pointer to the buffer.
         */
        uint8_t* get() { return ptr_; }
        
        /**
         * @brief Gets the const raw pointer to the buffer.
         * @return Const raw pointer to the buffer.
         */
        const uint8_t* get() const { return ptr_; }
        
        /**
         * @brief Overloads the array subscript operator for mutable access.
         * @param idx Index of the element.
         * @return Reference to the element.
         */
        uint8_t& operator[](size_t idx) { return ptr_[idx]; }
        
        /**
         * @brief Overloads the array subscript operator for const access.
         * @param idx Index of the element.
         * @return Const reference to the element.
         */
        const uint8_t& operator[](size_t idx) const { return ptr_[idx]; }
        
    private:
        uint8_t* ptr_;      ///< Pointer to the buffer data.
        bool is_spiram_;    ///< Flag indicating if the buffer is in SPIRAM.
        bool is_jpeg_aligned_; ///< Flag indicating if the buffer was allocated with jpeg_calloc_align.
    };

    /// Unique pointer type for TrackedBuffer.
    using UniqueBufferPtr = std::unique_ptr<TrackedBuffer>;

    /**
     * @brief Structure to hold the result of image processing.
     * Contains a unique pointer to the processed data and its size.
     */
    struct ProcessResult {
        UniqueBufferPtr data; ///< Unique pointer to the processed image data.
        size_t size;          ///< Size of the processed image data in bytes.
        
        /**
         * @brief Default constructor.
         */
        ProcessResult() : data(nullptr), size(0) {}
        
        /**
         * @brief Constructor with data and size.
         * @param ptr Unique pointer to the data.
         * @param sz Size of the data.
         */
        ProcessResult(UniqueBufferPtr&& ptr, size_t sz) 
            : data(std::move(ptr)), size(sz) {}
    };

  /**
   * @brief Constructs an ImageProcessor object.
   * @param config Configuration for the image processor.
   * @param model_handler Pointer to the ModelHandler for model input dimensions.
   */
  ImageProcessor(const ImageProcessorConfig &config, ModelHandler* model_handler);

  /**
   * @brief Splits the input image into specified crop zones and processes each.
   * @param image A shared pointer to the camera image.
   * @param zones A vector of CropZone objects defining regions to process.
   * @return A vector of ProcessResult, each containing processed image data for a zone.
   */
  std::vector<ProcessResult> split_image_in_zone(
      std::shared_ptr<camera::CameraImage> image,
      const std::vector<CropZone> &zones = {});

  /**
   * @brief Processes a zone directly into a pre-allocated buffer for minimal memory usage.
   * @param image A shared pointer to the camera image.
   * @param zone The CropZone to process.
   * @param output_buffer Pre-allocated buffer to store the processed image.
   * @param output_buffer_size Size of the output buffer.
   * @return True if successful, false otherwise.
   */
  bool process_zone_to_buffer(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone,
      uint8_t* output_buffer,
      size_t output_buffer_size);

 private:
  /**
   * @brief Processes a single crop zone from the image.
   * Handles JPEG decoding and scaling if necessary.
   * @param image A shared pointer to the camera image.
   * @param zone The CropZone to process.
   * @return A ProcessResult containing the processed image data for the zone.
   */
  ProcessResult process_zone(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone);

  /**
   * @brief Scales a cropped region of an image to the model's input dimensions.
   * This function is used for non-JPEG formats or when esp_new_jpeg's scaling
   * is not directly applicable.
   * @param src_data Pointer to the source image data.
   * @param src_width Width of the source image.
   * @param src_height Height of the source image.
   * @param zone The CropZone defining the region to scale.
   * @return A ProcessResult containing the scaled image data.
   */
  ProcessResult scale_cropped_region(
      const uint8_t *src_data,
      int src_width,
      int src_height,
      const CropZone &zone);

  /**
   * @brief Processes JPEG zone directly into output buffer.
   */
  bool process_jpeg_zone_to_buffer(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone,
      uint8_t* output_buffer,
      size_t output_buffer_size);

  /**
   * @brief Processes raw image zone directly into output buffer.
   */
  bool process_raw_zone_to_buffer(
      std::shared_ptr<camera::CameraImage> image,
      const CropZone &zone,
      uint8_t* output_buffer,
      size_t output_buffer_size);

  /**
   * @brief Validates if a given crop zone is within the camera's dimensions.
   * @param zone The CropZone to validate.
   * @return True if the zone is valid, false otherwise.
   */
  bool validate_zone(const CropZone &zone) const;

  /**
   * @brief Allocates a buffer for image data, preferring SPIRAM if available.
   * @param size The size of the buffer to allocate in bytes.
   * @return A UniqueBufferPtr to the allocated memory, or nullptr if allocation fails.
   */
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
      
      // Fallback to internal RAM with proper alignment
      if (!buf) {
        buf = (uint8_t*)heap_caps_malloc(size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DEFAULT);
      }
      
      if (buf) {
          return std::unique_ptr<TrackedBuffer>(new TrackedBuffer(buf, is_spiram));
      }
      
      return nullptr;
  }

  ImageProcessorConfig config_; ///< Configuration for the image processor.
  ModelHandler* model_handler_; ///< Pointer to the ModelHandler instance.
  int bytes_per_pixel_;        ///< Number of bytes per pixel for the current pixel format.
};

}  // namespace meter_reader_tflite
}  // namespace esphome