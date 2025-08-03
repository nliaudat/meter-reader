#pragma once

#include "esphome/core/component.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/esp32_camera/esp32_camera.h"
#include "esphome/components/camera/camera.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "memory_manager.h"
#include "model_handler.h"
#include "image_processor.h"
#include "crop_zones.h"
#include <memory>

namespace esphome {
namespace meter_reader_tflite {

class MeterReaderTFLite : public PollingComponent, public camera::CameraImageReader {
 public:
  void setup() override;
  void update() override;
  void loop() override;
  
  float get_setup_priority() const override { return setup_priority::LATE; }

  // CameraImageReader implementation
  void set_image(std::shared_ptr<camera::CameraImage> image) override;
  size_t available() const override;
  uint8_t *peek_data_buffer() override;
  void consume_data(size_t consumed) override;
  void return_image() override;

  void set_input_size(int width, int height);
  void set_confidence_threshold(float threshold);
  void set_tensor_arena_size(size_t size_bytes);
  void set_model(const uint8_t *model, size_t length);
  void set_camera(esp32_camera::ESP32Camera *camera);
  void set_value_sensor(sensor::Sensor *sensor);
  void set_crop_zones(const std::string &zones_json);
  void set_camera_format(int width, int height, const std::string &pixel_format);
  
  int get_model_input_width() const { return model_handler_.get_input_width(); }
  int get_model_input_height() const { return model_handler_.get_input_height(); }
  int get_model_input_channels() const { return model_handler_.get_input_channels(); } 
  
// #ifdef DEBUG_METER_READER_TFLITE
  // void set_debug_image(std::shared_ptr<camera::CameraImage> image);
// #endif

#ifdef DEBUG_METER_READER_TFLITE
  void set_debug_image(const uint8_t* data, size_t size);
  void test_with_debug_image();
#endif

 protected:
  bool allocate_tensor_arena();
  bool load_model();
  void report_memory_status();
  size_t get_arena_peak_bytes() const;
  void process_image();

  // Configuration parameters
  int camera_width_{0};
  int camera_height_{0};
  // int model_input_width_{32};
  // int model_input_height_{32};
 
 
  
  std::string pixel_format_{"RGB888"};
  float confidence_threshold_{0.7f};
  size_t tensor_arena_size_requested_{500 * 1024};
  
  // State variables
  size_t tensor_arena_size_actual_{0};
  bool model_loaded_{false};
  std::shared_ptr<camera::CameraImage> current_image_;
  size_t image_offset_{0};
  const uint8_t *model_{nullptr};
  size_t model_length_{0};
  sensor::Sensor *value_sensor_{nullptr};
  esp32_camera::ESP32Camera *camera_{nullptr};

  // Component instances
  MemoryManager memory_manager_;
  ModelHandler model_handler_;
  std::unique_ptr<ImageProcessor> image_processor_;
  CropZoneHandler crop_zone_handler_;
  MemoryManager::AllocationResult tensor_arena_allocation_;
  
// #ifdef DEBUG_METER_READER_TFLITE
  // std::shared_ptr<camera::CameraImage> debug_image_;
  // bool load_debug_image();
// #endif

#ifdef DEBUG_METER_READER_TFLITE
  std::shared_ptr<camera::CameraImage> debug_image_;
#endif

};

#ifdef DEBUG_METER_READER_TFLITE
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
#endif

}  // namespace meter_reader_tflite
}  // namespace esphome