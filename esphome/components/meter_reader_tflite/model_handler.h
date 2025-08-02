// model_handler.h
#pragma once

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "op_resolver.h"
#include <memory>

namespace esphome {
namespace meter_reader_tflite {

constexpr size_t MAX_OPERATORS = 30;
// static const char *const TAG = "ModelHandler";

class ModelHandler {
 public:
  bool load_model(const uint8_t *model_data, size_t model_size, 
                uint8_t* tensor_arena, size_t tensor_arena_size);
  bool invoke_model(const uint8_t *input_data, size_t input_size, 
                  float* output_value, float* output_confidence);
  size_t get_arena_peak_bytes() const;
  

  TfLiteTensor* input_tensor() const;
  TfLiteTensor* output_tensor() const;
  
    int get_input_width() const {
        if (!interpreter_ || !input_tensor()) return 0;
        if (input_tensor()->dims->size < 2) return 0;
        return input_tensor()->dims->data[1];  // Width is typically dimension 1
    }

    int get_input_height() const {
        if (!interpreter_ || !input_tensor()) return 0;
        if (input_tensor()->dims->size < 3) return 0;
        return input_tensor()->dims->data[2];  // Height is typically dimension 2
    }

    int get_input_channels() const {
        if (!interpreter_ || !input_tensor()) return 0;
        if (input_tensor()->dims->size < 4) return 0;
        return input_tensor()->dims->data[3];  // Channels is typically dimension 3
    }

 protected:
  const tflite::Model* tflite_model_{nullptr};
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
};

}  // namespace meter_reader_tflite
}  // namespace esphome