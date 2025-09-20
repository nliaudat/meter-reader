"""Component to use TensorFlow Lite Micro to read a meter."""
import esphome.codegen as cg
import esphome.config_validation as cv
import os
import zlib # for crc model check
from esphome.const import CONF_ID, CONF_MODEL, CONF_SENSOR
from esphome.core import CORE, HexInt
#from esphome.components import esp32, camera, sensor
from esphome.components import esp32, sensor
import esphome.components.esp32_camera as esp32_camera
from esphome.cpp_generator import RawExpression
from esphome.components import globals

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
CONF_DEBUG_IMAGE = 'debug_image'
CONF_DEBUG_OUT_PROCESSED_IMAGE_TO_SERIAL = 'debug_image_out_serial'
# CONF_DEBUG_DURATION = 'debug_duration' // can be enabled in  meter_reader_tflite.h #define DEBUG_DURATION
# CONF_DEBUG_IMAGE_PATH = 'debug_image_path'
CONF_SENSOR = 'meter_reader_value_sensor' 


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
    cv.Optional(CONF_DEBUG_IMAGE, default=False): cv.boolean, 
    cv.Optional(CONF_DEBUG_OUT_PROCESSED_IMAGE_TO_SERIAL, default=False): cv.boolean,
    cv.Optional('crop_zones_global'): cv.use_id(globals.GlobalsComponent),
}).extend(cv.polling_component_schema('60s'))

async def to_code(config):
    """Code generation for the component."""
    # Add IDF component and build flags
    esp32.add_idf_component(
        name="espressif/esp-tflite-micro",
        ref="*"  # Gets the latest version available ref="~1.3.4"
    )
    
    esp32.add_idf_component(
        name="espressif/esp-nn",
        ref="*" 
    )
    
    
    
    pixel_format = CORE.config["substitutions"].get("camera_pixel_format", "RGB888")
    
    # if pixel_format == "JPEG":
        # cg.add_define("USE_JPEG")
        # esp32.add_idf_component(
            # name="espressif/esp_jpeg",
            # ref="1.3.1"
        # )
        


    esp32.add_idf_component(
        name="espressif/esp_new_jpeg",
        ref="0.6.1"
    )
        
    # esp32.add_idf_component( ## not used actually
        # name="espressif/esp-dl",
        # ref="3.1.5"  # image preprocessor : https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_preprocessor.cpp
    # )
        

    cg.add_build_flag("-DTF_LITE_STATIC_MEMORY")
    cg.add_build_flag("-DTF_LITE_DISABLE_X86_NEON")
    cg.add_build_flag("-DESP_NN")
    cg.add_build_flag("-DUSE_ESP32_CAMERA_CONV")
    
    # Enable esp-nn optimizations
    cg.add_build_flag("-DOPTIMIZED_KERNEL=esp_nn")
    # cg.add_build_flag("-DUSE_ESP_NN_O1")
    # cg.add_build_flag("-DUSE_ESP_NN_O2")
    
    # Force reference kernels (debug without esp-nn)
    # cg.add_build_flag("-DOPTIMIZED_KERNEL=reference")
    
    # cg.add_build_flag("-DUSE_ESP_DL")
    
    # Enable higher precision modes in esp-nn
    # cg.add_build_flag("-DESP_NN_USE_HIGHER_PRECISION")
    # cg.add_build_flag("-DESP_NN_DISABLE_APPROXIMATIONS")
    
    ## debug only
    # cg.add_build_flag("-DTF_LITE_SHOW_OPERATIONS")
    # cg.add_build_flag("-DTF_LITE_ENABLE_DEBUG_OUTPUT")


    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cam = await cg.get_variable(config[CONF_CAMERA_ID])
    # cg.add(var.get_camera_image(cam)) #set camera
    cg.add(var.set_camera(cam))

    if CONF_SENSOR in config:
        sens = await sensor.new_sensor(config[CONF_SENSOR])
        cg.add(var.set_value_sensor(sens))
    
    model_path = config[CONF_MODEL]
    
    # Read the model file as binary data
    with open(model_path, "rb") as f:
        model_data = f.read()
        
    # Compute CRC32
    crc32_val = zlib.crc32(model_data) & 0xFFFFFFFF
    # Emit define
    cg.add_define("MODEL_CRC32", HexInt(crc32_val)) 
    
    # Create a progmem array for the model data
    rhs = [HexInt(x) for x in model_data]
    prog_arr = cg.progmem_array(config[CONF_RAW_DATA_ID], rhs)
    
    cg.add(var.set_model(prog_arr, len(model_data)))
    cg.add(var.set_confidence_threshold(config[CONF_CONFIDENCE_THRESHOLD]))
    
    # The config value is already an integer thanks to the schema validator
    cg.add(var.set_tensor_arena_size(config[CONF_TENSOR_ARENA_SIZE]))
    
    # Get camera resolution from substitutions
    width, height = 800, 600  # Defaults
    if CORE.config["substitutions"].get("camera_resolution"):
        res = CORE.config["substitutions"]["camera_resolution"]
        if 'x' in res:
            width, height = map(int, res.split('x'))
    
    pixel_format = CORE.config["substitutions"].get("camera_pixel_format", "RGB888")
    if pixel_format == "JPEG":   cg.add(var.set_camera_image_format(width, height, pixel_format))
    
    # register debug service (called by service: meter_reader_tflite_my_reader_debug)
    cg.add_define("USE_SERVICE_DEBUG")
    var = await cg.get_variable(config[CONF_ID])
    # template = """
    # register_service("%s_debug", 
        # [](%s *comp) { comp->dump_debug_info(); },
        # %s);
    # """ % (config[CONF_ID], 
           # "esphome::meter_reader_tflite::MeterReaderTFLite",
           # config[CONF_ID])
           
    # cg.add_global(cg.RawStatement(template))

    if config.get(CONF_DEBUG_IMAGE, False):
        cg.add_define("DEBUG_METER_READER_TFLITE")
        cg.add(var.set_debug_mode(True))
        
        # Set camera format first
        cg.add(var.set_camera_image_format(640, 480, "JPEG"))
        
        # Load debug image and set it
        component_dir = os.path.dirname(os.path.abspath(__file__))
        debug_image_path = os.path.join(component_dir, "debug.jpg")
        
        if not os.path.exists(debug_image_path):
            raise cv.Invalid(f"Debug image not found at {debug_image_path}")
        else:
            with open(debug_image_path, "rb") as f:
                debug_image_data = f.read()
        
        debug_image_id = f"{config[CONF_ID]}_debug_image"
        cg.add_global(
            cg.RawStatement(
               f"static const uint8_t {debug_image_id}[] = {{{', '.join(f'0x{x:02x}' for x in debug_image_data)}}};"
            )
        )
        
        # cg.add(
            # cg.RawStatement(
                # f"{config[CONF_ID]}->set_debug_image({debug_image_id}, sizeof({debug_image_id}));"
            # )
        # )
        
        cg.add(
            var.set_debug_image(
                cg.RawExpression(debug_image_id),
                len(debug_image_data)
            )
        )
        
        # Process debug image immediately
        # cg.add(var.test_with_debug_image())
        
    if config.get(CONF_DEBUG, False):
        cg.add_define("DEBUG_METER_READER_TFLITE")
        cg.add(var.set_debug_mode(True))
        
    if config.get(CONF_DEBUG_OUT_PROCESSED_IMAGE_TO_SERIAL, False):
        cg.add_define("DEBUG_OUT_PROCESSED_IMAGE_TO_SERIAL")

        
    if 'crop_zones_global' in config:
        crop_global = await cg.get_variable(config['crop_zones_global'])
        # Instead of passing the component, pass its value
        cg.add(var.set_crop_zones_global_string(crop_global.value()))
        
        
    # cg.add_define("USE_SERVICE_PARAM_TEST")
    
    # //Register the service using ESPHome's service registration
    # template = cg.RawStatement(f"""
        # register_service({config[CONF_ID]}_test_parameters, []({config[CONF_ID]} *comp) {{
            # comp->debug_test_parameters();
        # }});
    # """)
    # cg.add(template)