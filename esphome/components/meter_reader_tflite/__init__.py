"""Component to use TensorFlow Lite Micro to read a meter."""
import esphome.codegen as cg
import esphome.config_validation as cv
import os
from esphome.const import CONF_ID, CONF_MODEL, CONF_SENSOR
from esphome.core import CORE, HexInt
#from esphome.components import esp32, camera, sensor
from esphome.components import esp32, sensor
import esphome.components.esp32_camera as esp32_camera
from esphome.cpp_generator import RawExpression

CODEOWNERS = ["@nl"]
DEPENDENCIES = ['esp32', 'camera']
AUTO_LOAD = ['sensor']

CONF_CAMERA_ID = 'camera_id'
CONF_TENSOR_ARENA_SIZE = 'tensor_arena_size'
# CONF_MODEL_INPUT_WIDTH = 'model_input_width'
# CONF_MODEL_INPUT_HEIGHT = 'model_input_height'
CONF_CONFIDENCE_THRESHOLD = 'confidence_threshold'
CONF_RAW_DATA_ID = 'raw_data_id'
CONF_DEBUG = 'debug'
# CONF_DEBUG_IMAGE_PATH = 'debug_image_path'

meter_reader_tflite_ns = cg.esphome_ns.namespace('meter_reader_tflite')
MeterReaderTFLite = meter_reader_tflite_ns.class_('MeterReaderTFLite', cg.PollingComponent)

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
    #cv.Required(CONF_CAMERA_ID): cv.use_id(camera.Camera),
    cv.Required(CONF_CAMERA_ID): cv.use_id(esp32_camera.ESP32Camera),
    # cv.Optional(CONF_MODEL_INPUT_WIDTH, default=32): cv.positive_int,
    # cv.Optional(CONF_MODEL_INPUT_HEIGHT, default=32): cv.positive_int,
    cv.Optional(CONF_CONFIDENCE_THRESHOLD, default=0.7): cv.float_range(
        min=0.0, max=1.0
    ),
    cv.Optional(CONF_TENSOR_ARENA_SIZE, default='800KB'): cv.All( ### TODO : change with no range
        datasize_to_bytes,
        cv.Range(min=100 * 1024, max=800 * 1024)
    ),
    cv.Optional(CONF_SENSOR): sensor.sensor_schema(accuracy_decimals=2),
    cv.GenerateID(CONF_RAW_DATA_ID): cv.declare_id(cg.uint8),
    cv.Optional(CONF_DEBUG, default=False): cv.boolean, 
}).extend(cv.polling_component_schema('60s'))

async def to_code(config):
    """Code generation for the component."""
    # Add IDF component and build flags
    esp32.add_idf_component(
        name="espressif/esp-tflite-micro",
        ref="1.3.3~1"
    )
    
    # Get pixel format from substitutions
    pixel_format = CORE.config['substitutions'].get('camera_pixel_format', 'RGB888')
    
    # If pixel format is JPEG, add JPEG decoder component and define
    if pixel_format == "JPEG":
        cg.add_define("USE_JPEG")
        esp32.add_idf_component(
            name="espressif/esp_jpeg",
            ref="1.3.1"
        )
        
    cg.add_build_flag("-DTF_LITE_STATIC_MEMORY")
    cg.add_build_flag("-DTF_LITE_DISABLE_X86_NEON")
    cg.add_build_flag("-DESP_NN")

    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cam = await cg.get_variable(config[CONF_CAMERA_ID])
    cg.add(var.set_camera(cam))

    if CONF_SENSOR in config:
        sens = await sensor.new_sensor(config[CONF_SENSOR])
        cg.add(var.set_value_sensor(sens))
    
    model_path = config[CONF_MODEL]
    
    # Read the model file as binary data
    with open(model_path, "rb") as f:
        model_data = f.read()
    
    # Create a progmem array for the model data
    rhs = [HexInt(x) for x in model_data]
    prog_arr = cg.progmem_array(config[CONF_RAW_DATA_ID], rhs)
    
    cg.add(var.set_model(prog_arr, len(model_data)))
    # cg.add(var.set_input_size(config[CONF_MODEL_INPUT_WIDTH], config[CONF_MODEL_INPUT_HEIGHT]))
    cg.add(var.set_confidence_threshold(config[CONF_CONFIDENCE_THRESHOLD]))
    
    # The config value is already an integer thanks to the schema validator
    cg.add(var.set_tensor_arena_size(config[CONF_TENSOR_ARENA_SIZE]))
    
    # Get camera resolution from substitutions
    width, height = 800, 600  # Defaults
    if CORE.config['substitutions'].get('camera_resolution'):
        res = CORE.config['substitutions']['camera_resolution']
        if 'x' in res:
            width, height = map(int, res.split('x'))
    
    # Get pixel format from substitutions
    pixel_format = CORE.config['substitutions'].get('camera_pixel_format', 'RGB888')
    
    # Set camera format
    cg.add(var.set_camera_format(width, height, pixel_format))
    

    if config[CONF_DEBUG]:
        cg.add_define("DEBUG_METER_READER_TFLITE")
        cg.add(var.set_debug_mode(True))
        
        # Load debug image
        component_dir = os.path.dirname(os.path.abspath(__file__))
        debug_image_path = os.path.join(component_dir, "debug.jpg")
        
        if not os.path.exists(debug_image_path):
            raise cv.Invalid(f"Debug image not found at {debug_image_path}")
        
        with open(debug_image_path, "rb") as f:
            debug_image_data = f.read()
        
        # Create debug image array
        debug_image_id = f"{config[CONF_ID]}_debug_image"
        cg.add_global(
            cg.RawStatement(
                f"static const uint8_t {debug_image_id}[] = {{{','.join(f'0x{x:02x}' for x in debug_image_data)}}};"
            )
        )
        
        cg.add(
            var.set_debug_image(
                cg.RawExpression(debug_image_id),
                len(debug_image_data)
            )
        )
        
        # Process debug image immediately
        # cg.add(var.test_with_debug_image())