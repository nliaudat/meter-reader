#pragma once

#include <string>
#include <unordered_map>
#include "model_handler.h"

namespace esphome {
namespace meter_reader_tflite {

static const std::unordered_map<std::string, ModelConfig> MODEL_CONFIGS = {
    {"class100-0180", 
        ModelConfig{
            .description = "dig-class100-0180",
            .output_processing = "softmax_jomjol", //"softmax_scale10",// "logits_jomjol", //model trained with from_logits=True
            .scale_factor = 10.0f, // For 100-class models (0.0-9.9)
            .input_type = "float32", //"uint8", // Model is float32, not quantized!
            .input_channels = 3,
            .input_order = "RGB", //Keras ImageDataGenerator typically uses RGB order
            .input_size = {32, 20}, // Explicitly set expected size
            .normalize = false //.normalize = false 
        }
    },
    {"class100-0173", 
        ModelConfig{
            .description = "dig-class100-0173",
            .output_processing = "softmax_scale10",
            .scale_factor = 10.0f,
            .input_type = "float32",  // Quantized models use uint8 "float32",
            .input_channels = 3,
            .input_order = "BGR",
            .input_size = {32, 20}, 
            .normalize = false //.normalize = false      // Quantization handles scaling 
        }
    },
    {"class10-0900", 
        ModelConfig{
            .description = "dig-cont_0900",
            .output_processing = "softmax",
            .scale_factor = 1.0f,
            .input_type = "float32",  // Quantized models use uint8 "float32",
            .input_channels = 3,
            .input_order = "BGR",
            .input_size = {32, 20}, 
            .normalize = false //.normalize = false      // Quantization handles scaling 
        }
    },
    {"class10-0810", 
        ModelConfig{
            .description = "dig-cont_0810",
            .output_processing = "softmax",
            .scale_factor = 1.0f,
            .input_type = "float32",  // Quantized models use uint8 "float32",
            .input_channels = 3,
            .input_order = "BGR",
            .input_size = {32, 20},
            .normalize = false //.normalize = false      // Quantization handles scaling 
        }
    },
    {"mnist", 
        ModelConfig{
            .description = "MNIST Digit Classifier",
            .output_processing = "direct_class",
            .scale_factor = 1.0f,
            .input_type = "float32",
            .input_channels = 1,
            .input_order = "RGB",
            .input_size = {28, 28},
            .normalize = true,
            .invert = true
        }
    }
};


static const ModelConfig DEFAULT_MODEL_CONFIG = MODEL_CONFIGS.at("class100-0180");

}  // namespace meter_reader_tflite
}  // namespace esphome