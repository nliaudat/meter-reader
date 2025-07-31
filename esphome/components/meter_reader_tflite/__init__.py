"""Component to use TensorFlow Lite Micro to read a meter."""
import esphome.codegen as cg
import esphome.config_validation as cv
import os
from esphome.const import CONF_ID
from esphome.core import CORE, HexInt
from esphome.components import esp32

CODEOWNERS = ["@nl"]
DEPENDENCIES = ['esp32']
AUTO_LOAD = []

CONF_TENSOR_ARENA_SIZE = 'tensor_arena_size'
CONF_MODEL = 'model'
CONF_INPUT_WIDTH = 'input_width'
CONF_INPUT_HEIGHT = 'input_height'
CONF_CONFIDENCE_THRESHOLD = 'confidence_threshold'
CONF_RAW_DATA_ID = 'raw_data_id'

meter_reader_tflite_ns = cg.esphome_ns.namespace('meter_reader_tflite')
MeterReaderTFLite = meter_reader_tflite_ns.class_('MeterReaderTFLite', cg.Component)


def datasize_to_bytes(value):
    """Parse a data size string with units like KB, MB to bytes."""
    try:
        value = str(value).upper().strip()
        if value.endswith('KB'):
            return int(float(value[:-2]) * 1024)
        if value.endswith('MB'):
            return int(float(value[:-2]) * 1024 * 1024)
        if value.endswith('B'):
            return int(value[:-1])
        return int(value)
    except ValueError as e:
        raise cv.Invalid(f"Invalid data size: {e}") from e

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(MeterReaderTFLite),
    cv.Required(CONF_MODEL): cv.file_,
    cv.Optional(CONF_INPUT_WIDTH, default=96): cv.positive_int,
    cv.Optional(CONF_INPUT_HEIGHT, default=96): cv.positive_int,
    cv.Optional(CONF_CONFIDENCE_THRESHOLD, default=0.7): cv.float_range(
        min=0.0, max=1.0
    ),
    cv.Optional(CONF_TENSOR_ARENA_SIZE, default='800KB'): cv.All(
        datasize_to_bytes,
        cv.Range(min=400 * 1024, max=800 * 1024)
    ),
    cv.GenerateID(CONF_RAW_DATA_ID): cv.declare_id(cg.uint8),
}).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    """Code generation for the component."""
    # Add IDF component and build flags
    esp32.add_idf_component(
        name="espressif/esp-tflite-micro",
        ref="1.3.3~1"
    )
    
    cg.add_build_flag("-DTF_LITE_STATIC_MEMORY")
    cg.add_build_flag("-DTF_LITE_DISABLE_X86_NEON")
    cg.add_build_flag("-DESP_NN")

    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    
    model_path = os.path.join(CORE.config_dir, config[CONF_MODEL])
    
    # Read the model file as binary data
    with open(model_path, "rb") as f:
        model_data = f.read()
    
    # Create a progmem array for the model data
    rhs = [HexInt(x) for x in model_data]
    prog_arr = cg.progmem_array(config[CONF_RAW_DATA_ID], rhs)
    
    cg.add(var.set_model(prog_arr, len(model_data)))
    cg.add(var.set_input_size(config[CONF_INPUT_WIDTH], config[CONF_INPUT_HEIGHT]))
    cg.add(var.set_confidence_threshold(config[CONF_CONFIDENCE_THRESHOLD]))
    
    # The config value is already an integer thanks to the schema validator
    cg.add(var.set_tensor_arena_size(config[CONF_TENSOR_ARENA_SIZE]))