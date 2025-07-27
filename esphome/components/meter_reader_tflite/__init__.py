import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.const import CONF_ID
from esphome.core import CORE, logging
from esphome.components import esp32 

DEPENDENCIES = ['esp32']
AUTO_LOAD = []

CONF_TENSOR_ARENA_SIZE = 'tensor_arena_size'
CONF_MODEL_PATH = 'model_path'
CONF_INPUT_WIDTH = 'input_width'
CONF_INPUT_HEIGHT = 'input_height'
CONF_CONFIDENCE_THRESHOLD = 'confidence_threshold'

_LOGGER = logging.getLogger(__name__)
meter_reader_tflite_ns = cg.esphome_ns.namespace('meter_reader_tflite')
MeterReaderTFLite = meter_reader_tflite_ns.class_('MeterReaderTFLite', cg.Component)

def validate_tensor_arena(value):
    """Convert string like '800KB' to bytes and validate range."""
    if isinstance(value, str):
        try:
            if value.endswith('KB'):
                value = int(float(value[:-2]) * 1024)
            elif value.endswith('MB'):
                value = int(float(value[:-2]) * 1024 * 1024)
            else:
                value = int(value)
        except ValueError as e:
            raise cv.Invalid(f"Invalid tensor arena size: {str(e)}")
    
    value = int(value)
    if value < 400 * 1024:
        raise cv.Invalid("Tensor arena must be at least 400KB")
    if value > 800 * 1024:
        raise cv.Invalid("Maximum tensor arena size is 800KB")
    return value

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(MeterReaderTFLite),
    cv.Required(CONF_MODEL_PATH): cv.string,
    cv.Optional(CONF_INPUT_WIDTH, default=96): cv.int_,
    cv.Optional(CONF_INPUT_HEIGHT, default=96): cv.int_,
    cv.Optional(CONF_CONFIDENCE_THRESHOLD, default=0.7): cv.float_,
    cv.Optional(CONF_TENSOR_ARENA_SIZE, default='800KB'): cv.All(
        cv.string,
        validate_tensor_arena
    ),
}).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    """Generate the C++ code for the component."""
    
    ########################################### VERY IMPORTANT ! Load esp-tflite-micro external dependency 
    
    # Add IDF component dependency
    esp32.add_idf_component(
        name="espressif/esp-tflite-micro",
        ref="1.3.3~1"  # Using same version as micro_wake_word
    )
    
    #Add required build flags
    cg.add_build_flag("-DTF_LITE_STATIC_MEMORY")
    cg.add_build_flag("-DTF_LITE_DISABLE_X86_NEON")
    cg.add_build_flag("-DESP_NN")
    
    # cg.add_library("kahrendt/ESPMicroSpeechFeatures", "1.1.0")
    # cg.add_library("espressif/esp-tflite-micro", "1.3.3")
    
    # esp32 = cg.esphome_core.target_platform == "esp32"

    # if esp32:
        # cg.add_build_flag("-DTF_LITE_STATIC_MEMORY")
        # cg.add_build_flag("-DTF_LITE_DISABLE_X86_NEON")
        # cg.add_build_flag("-DESP_NN")
        # cg.add_define("USE_METER_READER_TFLITE")

        # esp32_target = cg.get_variable(cg.App).target
        # cg.add_idf_component(name="espressif/esp-tflite-micro", ref="1.3.3~1")
        


    # Create and register component
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    
    # Set configuration values
    cg.add(var.set_model_path(config[CONF_MODEL_PATH]))
    cg.add(var.set_input_size(config[CONF_INPUT_WIDTH], config[CONF_INPUT_HEIGHT]))
    cg.add(var.set_confidence_threshold(config[CONF_CONFIDENCE_THRESHOLD]))
    
    arena_size = validate_tensor_arena(config[CONF_TENSOR_ARENA_SIZE])
    cg.add(var.set_tensor_arena_size(arena_size))
    
def validate_config(config):
    import os
    ESP_LOGI("Custom Component", "Search path: %s", os.path.abspath(os.path.dirname(__file__)))
    return config
