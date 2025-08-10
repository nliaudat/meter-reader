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
            .output_processing = "softmax_scale10",
            .scale_factor = 10.0f,
            .input_type = "float32",
            .input_channels = 3,
			.normalize = true
        }
    },
    {"class100-0173", 
        ModelConfig{
            .description = "dig-class100-0173",
            .output_processing = "softmax_scale10",
            .scale_factor = 10.0f,
            .input_type = "float32",
            .input_channels = 3,
			.normalize = true
        }
    },
    {"class10-0900", 
        ModelConfig{
            .description = "dig-cont_0900",
            .output_processing = "softmax",
            .scale_factor = 1.0f,
            .input_type = "float32",
            .input_channels = 3,
			.normalize = true
        }
    },
    {"class10-0810", 
        ModelConfig{
            .description = "dig-cont_0810",
            .output_processing = "softmax",
            .scale_factor = 1.0f,
            .input_type = "float32",
            .input_channels = 3,
			.normalize = true
        }
    },
    {"mnist", 
        ModelConfig{
            .description = "MNIST Digit Classifier",
            .output_processing = "direct_class",
            .scale_factor = 1.0f,
            .input_type = "float32",
            .input_channels = 1,
            .input_size = {28, 28},
            .normalize = true,
            .invert = true
        }
    }
};


static const ModelConfig DEFAULT_MODEL_CONFIG = MODEL_CONFIGS.at("class100-0180");

}  // namespace meter_reader_tflite
}  // namespace esphome