#pragma once

#include "esphome/core/component.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/esp32_camera/esp32_camera.h"
#include "esphome/components/camera/camera.h"
// #include "esphome/core/hal.h"  // For millis() and other core functions
// #include "esphome/core/application.h" //for defer loading after boot
#include "model_handler.h"
#include "memory_manager.h"
#include "image_processor.h"
#include "crop_zones.h"
#include "model_config.h"
// #include "debug_utils.h"
#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <mutex> // needs CONFIG_FREERTOS_SUPPORT_STATIC_ALLOCATION=y # for frame_mutex_ in meter_reader_tflite.*
// #include <numeric> // for std::accumulate


#define DEBUG_DURATION  // Comment this out to disable duration debugging




namespace esphome {
namespace meter_reader_tflite {

class MeterReaderTFLite : public PollingComponent, public camera::CameraImageReader {
 public:
  void setup() override;
  void update() override;
  void loop() override;
  void request_and_process_image();
  
  // class destructor :
  ~MeterReaderTFLite() override;  // 'override' is important for virtual destructors
  
  float get_setup_priority() const override { return setup_priority::LATE; }

  // CameraImageReader implementation
  void set_image(std::shared_ptr<camera::CameraImage> image) override;
  size_t available() const override;
  uint8_t *peek_data_buffer() override;
  void consume_data(size_t consumed) override;
  void return_image() override;

  void set_confidence_threshold(float threshold) { confidence_threshold_ = threshold; }
  void set_tensor_arena_size(size_t size_bytes) { tensor_arena_size_requested_ = size_bytes; }
  void set_model(const uint8_t *model, size_t length);
  void get_camera_image(esp32_camera::ESP32Camera *camera);
  void set_value_sensor(sensor::Sensor *sensor);
  void set_crop_zones(const std::string &zones_json);
  void set_camera_image_format(int width, int height, const std::string &pixel_format);
  void set_camera(esp32_camera::ESP32Camera *camera) { camera_ = camera; }
  void set_model_config(const std::string &model_type);
  void dump_debug_info() {
    this->print_debug_info();
  }
  static void register_service(MeterReaderTFLite *comp) {
        comp->dump_debug_info();
  }
  void print_debug_info();
  
  int get_model_input_width() const { return model_handler_.get_input_width(); }
  int get_model_input_height() const { return model_handler_.get_input_height(); }
  int get_model_input_channels() const { return model_handler_.get_input_channels(); } 
  
/* #ifdef DEBUG_METER_READER_TFLITE
  void set_debug_image(const uint8_t* data, size_t size);
  void test_with_debug_image();
  void set_debug_mode(bool debug_mode);
#endif */
	void set_confidence_sensor(sensor::Sensor *sensor) { confidence_sensor_ = sensor; }

 protected:
  sensor::Sensor *confidence_sensor_{nullptr};
  bool allocate_tensor_arena();
  bool load_model();
  bool image_requested_ = false;
  void report_memory_status();
  size_t get_arena_peak_bytes() const;
  // void process_full_image();
  void process_full_image(std::shared_ptr<camera::CameraImage> frame);
  // bool process_model_result(const ProcessedImageResult& result, float* value, float* confidence);
  bool process_model_result(const ImageProcessor::ProcessResult& result, float* value, float* confidence);
  float combine_readings(const std::vector<float> &readings);
  // void preprocess_image(std::shared_ptr<camera::CameraImage> image);

  // Configuration parameters
  int camera_width_{0};
  int camera_height_{0};
  std::string pixel_format_{"RGB888"};
  float confidence_threshold_{0.7f};
  size_t tensor_arena_size_requested_{500 * 1024};
  std::string model_type_{"default"};
  
  // State variables
  size_t tensor_arena_size_actual_{0};
  bool model_loaded_{false};
  std::shared_ptr<camera::CameraImage> current_frame_;
  size_t image_offset_{0};
  const uint8_t *model_{nullptr};
  size_t model_length_{0};
  sensor::Sensor *value_sensor_{nullptr};
  esp32_camera::ESP32Camera *camera_{nullptr};
  bool debug_mode_ = false;

  // Component instances
  MemoryManager memory_manager_;
  ModelHandler model_handler_;
  std::unique_ptr<ImageProcessor> image_processor_;
  CropZoneHandler crop_zone_handler_;
  MemoryManager::AllocationResult tensor_arena_allocation_;
  
  std::shared_ptr<camera::CameraImage> debug_image_;
  
  // bool is_processing_image_ = false; // Todo : better if std::atomic<bool> is_processing_image_{false};
  std::atomic<bool> is_processing_{false};
  // QueueHandle_t frame_queue_;
  
/* #ifdef DEBUG_METER_READER_TFLITE
  // std::shared_ptr<camera::CameraImage> debug_image_;
    const std::vector<CropZone> debug_crop_zones_ = {
    {80, 233, 116, 307}, {144, 235, 180, 307},
    {202, 234, 238, 308}, {265, 233, 304, 306},
    {328, 232, 367, 311}, {393, 231, 433, 310},
    {460, 235, 499, 311}, {520, 235, 559, 342}
  };
#endif */

private:
  // Double buffering
  // std::array<std::shared_ptr<camera::CameraImage>, 2> frame_buffers_;
  // std::atomic<int> active_buffer_{0};
  // std::atomic<bool> processing_buffer_{false};
  
  void setup_camera_callback();
  
  struct QueuedFrame {
    std::shared_ptr<camera::CameraImage> frame;
    uint32_t timestamp;
  };
  QueueHandle_t frame_queue_;
  // std::mutex frame_mutex_;
  bool frame_requested_{false};
  
  // State flags
  // bool debug_mode_{false};
  // bool model_loaded_{false};
  // esp32_camera::ESP32Camera *camera_{nullptr};
  
  
};


/* #ifdef DEBUG_METER_READER_TFLITE
class DebugCameraImage : public camera::CameraImage {
 public:
  DebugCameraImage(const uint8_t* data, size_t size, int width, int height, const std::string& format)
    : data_(data, data + size), width_(width), height_(height), format_(format) {}

  uint8_t* get_data_buffer() override { return const_cast<uint8_t*>(data_.data()); }
  size_t get_data_length() override { return data_.size(); }
  bool was_requested_by(camera::CameraRequester requester) const override { return true; }
  
 private:
  std::vector<uint8_t> data_;
  int width_;
  int height_;
  std::string format_;
};
#endif */

}  // namespace meter_reader_tflite
}  // namespace esphome