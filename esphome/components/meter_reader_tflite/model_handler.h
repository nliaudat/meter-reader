// model_handler.h
#pragma once

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "op_resolver.h"
#include <memory>

namespace esphome {
namespace meter_reader_tflite {

constexpr size_t MAX_OPERATORS = 90;  // Define the constant here

class ModelHandler {
 public:
  bool load_model(const uint8_t *model_data, size_t model_size, 
                uint8_t* tensor_arena, size_t tensor_arena_size);
  bool invoke_model(const uint8_t *input_data, size_t input_size, 
                  float* output_value, float* output_confidence);
  size_t get_arena_peak_bytes() const;
  
  // Add these helper methods
  TfLiteTensor* input_tensor() const;
  TfLiteTensor* output_tensor() const;

 protected:
  const tflite::Model* tflite_model_{nullptr};
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
};

}  // namespace meter_reader_tflite
}  // namespace esphome