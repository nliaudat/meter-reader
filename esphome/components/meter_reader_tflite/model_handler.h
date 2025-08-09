#pragma once

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "op_resolver.h"
// #include "debug_utils.h"
#include <memory>
#include <string>

namespace esphome {
namespace meter_reader_tflite {

struct ModelConfig {
  std::string description;
  std::string output_processing; // "softmax_scale10", "softmax", "direct_class"
  float scale_factor;
  std::string input_type;
  int input_channels = 3;
  std::pair<int, int> input_size = {0, 0};
  bool normalize = false;
  bool invert = false;
};

constexpr size_t MAX_OPERATORS = 30;

class ModelHandler {
 public:
  bool load_model(const uint8_t *model_data, size_t model_size, 
                uint8_t* tensor_arena, size_t tensor_arena_size,
                const ModelConfig &config);
                
  // bool invoke_model(const uint8_t *input_data, size_t input_size, 
                  // float* output_value, float* output_confidence);
  bool invoke_model(const uint8_t* input_data, size_t input_size);
  const float* get_output() const { return model_output_; }
  int get_output_size() const { return output_size_; }
                  
  size_t get_arena_peak_bytes() const;
  
  TfLiteTensor* input_tensor() const;
  TfLiteTensor* output_tensor() const;
  
  int get_input_width() const {
    if (!interpreter_ || !input_tensor()) return 0;
    if (input_tensor()->dims->size < 2) return 0;
    return input_tensor()->dims->data[1];
  }

  int get_input_height() const {
    if (!interpreter_ || !input_tensor()) return 0;
    if (input_tensor()->dims->size < 3) return 0;
    return input_tensor()->dims->data[2];
  }

  int get_input_channels() const {
    if (!interpreter_ || !input_tensor()) return 0;
    if (input_tensor()->dims->size < 4) return 0;
    return input_tensor()->dims->data[3];
  }

  const ModelConfig& get_config() const { return config_; }
  void set_config(const ModelConfig &config) { config_ = config; }
  
    // const float* model_output_ = nullptr;
    // int output_size_ = 0;


 protected:
  float process_output(const float* output_data) const;
  bool validate_model_config();

  const tflite::Model* tflite_model_{nullptr};
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
  ModelConfig config_;
  
  
private:

    const float* model_output_ = nullptr;
    int output_size_ = 0;
  
};

}  // namespace meter_reader_tflite
}  // namespace esphome