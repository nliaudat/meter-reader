# Meter Reader TFLite - ESPHome Component

## Overview

The `meter_reader_tflite` is an ESPHome custom component that uses TensorFlow Lite Micro to read meter values from camera images. It captures images from an ESP32 camera, processes them through a TFLite model, and extracts meter readings with confidence scores.

**Repository**: https://github.com/nliaudat/meter-reader/tree/main/esphome/components/meter_reader_tflite

## Features

- **TensorFlow Lite Micro Integration**: Runs ML inference directly on ESP32
- **Multi-zone Processing**: Supports multiple crop zones for digit extraction
- **Flexible Image Formats**: RGB888, JPEG, RGB565, and grayscale support
- **Configurable Models**: Pre-configured for various model types (100-class, 10-class, MNIST)
- **Memory Efficient**: Smart memory management with SPIRAM support
- **Debug Capabilities**: Extensive logging and image analysis tools
- **Global Configuration**: Dynamic crop zones via global variables

## Installation

### Method 1: Git Submodule (Recommended)

```bash
cd your-esphome-config-directory
mkdir -p custom_components
cd custom_components
git submodule add https://github.com/nliaudat/meter-reader.git
```

### Method 2: Manual Download

1. Download the component from the GitHub repository
2. Extract to your ESPHome custom components directory:
```
config/esphome/custom_components/meter_reader_tflite/
```

### File Structure

Ensure you have this structure:
```
config/esphome/custom_components/
└── meter_reader_tflite/
    ├── __init__.py
    ├── meter_reader_tflite.h
    ├── meter_reader_tflite.cpp
    ├── model_handler.h
    ├── model_handler.cpp
    ├── image_processor.h
    ├── image_processor.cpp
    ├── crop_zones.h
    ├── crop_zones.cpp
    ├── memory_manager.h
    ├── memory_manager.cpp
    ├── model_config.h
    ├── op_resolver.h
    ├── debug_utils.h
    └── manifest.json
```

## Dependencies

The component automatically adds these ESP-IDF components during build:
- `espressif/esp-tflite-micro` (~1.3.4)
- `espressif/esp-nn` (~1.1.2)
- `espressif/esp_new_jpeg` (0.6.1)

## Configuration

### Basic Configuration

```yaml
external_components:
  - source:
      type: local
      path: custom_components/meter_reader_tflite

esp32_camera:
  id: cam0
  name: "Meter Camera"
  resolution: 800x600
  pixel_format: RGB888

meter_reader_tflite:
  id: mr0
  model: "model.tflite"
  camera_id: cam0
  confidence_threshold: 0.7
  tensor_arena_size: 100KB
  update_interval: 60s
  meter_reader_value_sensor:
    name: "Meter Reading"
```

### Advanced Configuration

```yaml
meter_reader_tflite:
  id: mr0
  model: "model.tflite"
  camera_id: cam0
  confidence_threshold: 0.8
  tensor_arena_size: 500KB
  update_interval: 30s
  debug: true
  debug_image: false
  debug_image_out_serial: false
  
  meter_reader_value_sensor:
    name: "Meter Value"
    id: meter_value
    accuracy_decimals: 2
    
  confidence_sensor:
    name: "Confidence Score"
    id: confidence_score
    accuracy_decimals: 3

globals:
  - id: crop_zones_global
    type: string
    initial_value: '[[80,233,116,307],[144,235,180,307],[202,234,238,308],[265,233,304,306],[328,232,367,311],[393,231,433,310],[460,235,499,311],[520,235,559,342]]'

meter_reader_tflite:
  # ... other config
  crop_zones_global: globals.crop_zones_global
```

## Model Support

### Pre-configured Models

The component supports these model types (defined in `model_config.h`):

1. **class100-0180/0173**: 100-class models (0.0-9.9 scale)
   - Input: 32x20x3 RGB
   - Processing: softmax_jomjol
   - Scale factor: 10.0

2. **class10-0900/0810**: 10-class digit classifiers
   - Input: 32x20x3 RGB  
   - Processing: softmax_jomjol
   - Scale factor: 1.0

3. **mnist**: MNIST-style models
   - Input: 28x28x1 grayscale
   - Processing: direct_class
   - Normalization: true

### Custom Models

To use a custom model, add its configuration to `model_config.h`:

```cpp
{"custom_model", 
    ModelConfig{
        .description = "Custom Model",
        .output_processing = "softmax_jomjol",
        .scale_factor = 1.0f,
        .input_type = "float32",
        .input_channels = 3,
        .input_order = "RGB",
        .input_size = {32, 32},
        .normalize = false
    }
}
```

## Crop Zones Format

Crop zones define regions for digit extraction in JSON format:

```json
[
  [x1, y1, x2, y2],  // Digit 1
  [x1, y1, x2, y2],  // Digit 2
  // ... more digits
]
```

- **x1, y1**: Top-left coordinates
- **x2, y2**: Bottom-right coordinates
- Zones are processed left-to-right as digit sequence

## Debug Features

### Serial Debug Output

Enable detailed logging:
```yaml
meter_reader_tflite:
  # ... other config
  debug: true
  debug_image_out_serial: true
```

Debug output includes:
- Raw model outputs and confidence scores
- Image processing statistics  
- Memory usage reports
- Zone-by-zone analysis
- ASCII art previews of processed images

### Debug Image Testing

Use a static image for testing:
```yaml
meter_reader_tflite:
  # ... other config
  debug_image: true
```

Place `debug.jpg` in the component directory for offline testing.

## API Reference

### Component Methods

```cpp
// Main component class
class MeterReaderTFLite : public PollingComponent, public camera::CameraImageReader {
public:
    void setup() override;
    void update() override;
    void loop() override;
    
    // Configuration setters
    void set_model(const uint8_t *model, size_t length);
    void set_camera(esp32_camera::ESP32Camera *camera);
    void set_value_sensor(sensor::Sensor *sensor);
    void set_confidence_sensor(sensor::Sensor *sensor);
    void set_crop_zones(const std::string &zones_json);
    void set_camera_image_format(int width, int height, const std::string &pixel_format);
    
    // Debug methods
    void print_debug_info();
    void set_debug_mode(bool debug_mode);
};
```

### Key Classes

1. **ModelHandler**: TFLite model loading and inference
2. **ImageProcessor**: Image preprocessing and scaling
3. **CropZoneHandler**: Crop zone management
4. **MemoryManager**: Tensor arena allocation

## Performance Tuning

### Memory Optimization

- **Tensor Arena**: Start with 50KB, increase if model fails to load
- **Image Resolution**: Use lowest practical camera resolution
- **Crop Zones**: Define precise zones to minimize processing area

### Model Optimization

- Use quantized models for better performance
- Match input dimensions to actual digit size
- Prefer 100-class models for decimal precision
- Test with different output processing methods

## Troubleshooting

### Common Issues

1. **Model Loading Failure**
   - Check model file path and size
   - Verify tensor arena size is sufficient
   - Check ESP32 flash size limitations

2. **Low Confidence Scores**
   - Adjust crop zone coordinates
   - Check lighting conditions
   - Verify model matches digit style

3. **Memory Allocation Errors**
   - Reduce tensor arena size
   - Lower camera resolution
   - Enable SPIRAM if available

### Debugging Steps

1. Enable debug mode and check serial output
2. Verify crop zone coordinates match camera resolution
3. Test with debug image before live camera
4. Monitor memory usage statistics

## Example Use Cases

### Water Meter Reading
```yaml
meter_reader_tflite:
  id: water_meter
  model: "water_meter.tflite"
  camera_id: cam0
  confidence_threshold: 0.6
  tensor_arena_size: 512KB
  update_interval: 120s
```

### Electricity Meter
```yaml
meter_reader_tflite:
  id: power_meter
  model: "power_meter.tflite" 
  camera_id: cam0
  confidence_threshold: 0.8
  tensor_arena_size: 512KB
  update_interval: 30s
```

## Files Included

- **`__init__.py`**: ESPHome component configuration and code generation
- **`meter_reader_tflite.h/cpp`**: Main component implementation
- **`model_handler.h/cpp`**: TFLite model management
- **`image_processor.h/cpp`**: Image preprocessing pipeline
- **`crop_zones.h/cpp`**: Crop zone parsing and management
- **`memory_manager.h/cpp`**: Memory allocation utilities
- **`model_config.h`**: Predefined model configurations
- **`op_resolver.h`**: TFLite operation registration
- **`debug_utils.h`**: Debugging macros and utilities
- **`manifest.json`**: Component metadata

## Support

For issues and questions:
1. Check the debug output with `debug: true`
2. Verify all file dependencies are present
3. Ensure compatible ESP32 camera configuration
4. Check serial logs for specific error messages

## License

This component is provided as-is for ESPHome integration. Ensure compliance with TensorFlow Lite and ESP-IDF component licenses when deploying.

---

*Note: This component requires an ESP32 with camera support and sufficient memory for TensorFlow Lite operations.*