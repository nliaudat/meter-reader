#pragma once

#include "esphome.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <memory>
#include <vector>

namespace esphome {
namespace meter_reader_tflite {

class MeterReaderTFLite : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::LATE; }

  void set_model_path(const std::string &path) { model_path_ = path; }
  void set_input_size(int width, int height) {
    input_width_ = width;
    input_height_ = height;
  }
  void set_confidence_threshold(float threshold) { confidence_threshold_ = threshold; }
  void set_tensor_arena_size(size_t size_bytes) { tensor_arena_size_requested_ = size_bytes; }
  
  size_t get_actual_arena_size() const { return tensor_arena_ ? tensor_arena_size_actual_ : 0; }
  size_t get_model_size() const { return model_size_; }

 protected:
  bool allocate_tensor_arena();
  bool load_model();
  bool run_inference();
  void report_memory_status();
  void preprocess_image(uint8_t* image_data);
  void process_output(float* output_data);

  std::string model_path_;
  int input_width_{96};
  int input_height_{96};
  float confidence_threshold_{0.7f};
  size_t tensor_arena_size_requested_{800 * 1024}; // 800KB default
  size_t tensor_arena_size_actual_{0};
  size_t model_size_{0};
  
  std::unique_ptr<uint8_t[]> tensor_arena_;
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
  const tflite::Model* model_{nullptr};
};

}  // namespace meter_reader_tflite
}  // namespace esphome