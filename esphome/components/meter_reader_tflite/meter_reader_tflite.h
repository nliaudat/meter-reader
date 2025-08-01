#pragma once

#include "esphome/core/component.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/esp32_camera/esp32_camera.h"
#include "esphome/components/camera/camera.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "esp_heap_caps.h"
#include "op_resolver.h"
#include <memory>
#include <vector>
#include <string>

namespace esphome {
namespace meter_reader_tflite {

struct CropZone {
  int x1;
  int y1;
  int x2;
  int y2;
};

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

  void set_input_size(int width, int height) {
    model_input_width_ = width;
    model_input_height_ = height;
  }
  void set_confidence_threshold(float threshold) { confidence_threshold_ = threshold; }
  void set_tensor_arena_size(size_t size_bytes) { tensor_arena_size_requested_ = size_bytes; }
  
  void set_model(const uint8_t *model, size_t length) {
    model_ = model;
    model_length_ = length;
  }

  void set_camera(esp32_camera::ESP32Camera *camera);
  void set_value_sensor(sensor::Sensor *sensor) { value_sensor_ = sensor; }
  
  void set_crop_zones(const std::string &zones_json);
  void set_camera_format(int width, int height, const std::string &pixel_format);

 protected:
  bool allocate_tensor_arena();
  bool load_model();
  void report_memory_status();
  size_t get_arena_peak_bytes() const;
  void process_image();
  void process_single_image(std::shared_ptr<camera::CameraImage> image);
  std::shared_ptr<camera::CameraImage> crop_and_resize_image(
      std::shared_ptr<camera::CameraImage> image, const CropZone &zone);
  void parse_crop_zones(const std::string &zones_json);

  esp32_camera::ESP32Camera *camera_{nullptr};
  int model_input_width_{32};
  int model_input_height_{32};
  float confidence_threshold_{0.7f};
  size_t tensor_arena_size_requested_{500 * 1024};
  size_t tensor_arena_size_actual_{0};
  sensor::Sensor *value_sensor_{nullptr};
  const uint8_t *model_{nullptr};
  size_t model_length_{0};
  
  std::shared_ptr<camera::CameraImage> current_image_;
  size_t image_offset_{0};
  std::vector<CropZone> crop_zones_;
  int camera_width_{0};
  int camera_height_{0};
  std::string pixel_format_{"RGB888"};
  int bytes_per_pixel_{3};
  bool model_loaded_{false};

  struct HeapCapsDeleter {
    void operator()(uint8_t *p) const { heap_caps_free(p); }
  };
  std::unique_ptr<uint8_t[], HeapCapsDeleter> tensor_arena_;
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
  const tflite::Model* tflite_model_{nullptr};
};

}  // namespace meter_reader_tflite
}  // namespace esphome