# ESPHome Meter Reader TFLite Component

> AI-powered meter reading using TensorFlow Lite on ESP32 cameras

[![ESPHome](https://img.shields.io/badge/ESPHome-Compatible-brightgreen)](https://esphome.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🚀 What is this?

This ESPHome component enables automatic meter reading using machine learning. It captures images from an ESP32 camera, processes them through a TensorFlow Lite model, and extracts numerical readings from utility meters (water, electricity, gas, etc.).

## ✨ Key Features

- **🤖 AI-Powered**: Uses TensorFlow Lite Micro for on-device inference
- **📷 Multi-Format Support**: Works with RGB888, JPEG, RGB565, and grayscale images
- **🎯 Smart Cropping**: Multiple configurable zones for digit extraction
- **⚡ Real-time Processing**: Runs directly on ESP32 without cloud dependency
- **🔧 Easy Configuration**: Simple YAML configuration with ESPHome
- **🐛 Extensive Debugging**: Built-in debugging tools and image analysis

## 🏁 Quick Start

### 1. Installation

Add this to your ESPHome configuration:

```yaml
external_components:
  - source: 
      type: git
      url: https://github.com/nliaudat/meter-reader
      ref: main
    components: [meter_reader_tflite]
```

### 2. Basic Configuration

```yaml
# Configure camera
esp32_camera:
  id: my_camera
  name: "Meter Camera"
  resolution: 800x600
  pixel_format: RGB888

# Configure meter reader
meter_reader_tflite:
  id: meter_reader
  model: "model.tflite"  # Your trained model file
  camera_id: my_camera
  confidence_threshold: 0.7
  tensor_arena_size: 512KB
  update_interval: 60s
  
  # Optional: Publish readings as sensor
  meter_reader_value_sensor:
    name: "Water Meter Reading"
    id: water_meter_value
```

### 3. Add Your Model

Place your trained `.tflite` model file in the same directory as your ESPHome configuration.

## 📋 Prerequisites

- **ESP32 board** with camera support
- **ESPHome 2023.12** or newer
- **Trained TFLite model** for digit recognition
- **Camera** focused on your meter display

## 🎯 How It Works

1. **Image Capture**: ESP32 camera takes a picture of the meter
2. **Zone Processing**: Image is split into predefined digit zones
3. **AI Inference**: Each zone is processed by the TensorFlow Lite model
4. **Digit Recognition**: Model identifies numbers in each zone
5. **Value Assembly**: Individual digits are combined into final reading
6. **Sensor Output**: Result published as ESPHome sensor

## ⚙️ Configuration Examples

### Basic Water Meter

```yaml
meter_reader_tflite:
  id: water_meter
  model: "water_model.tflite"
  camera_id: my_camera
  confidence_threshold: 0.6
  tensor_arena_size: 400KB
  update_interval: 120s
  
  meter_reader_value_sensor:
    name: "Water Consumption"
    unit_of_measurement: "m³"
    accuracy_decimals: 2
```

### Advanced Electricity Meter with Debugging

```yaml
meter_reader_tflite:
  id: power_meter
  model: "power_model.tflite"
  camera_id: my_camera
  confidence_threshold: 0.8
  tensor_arena_size: 600KB
  update_interval: 30s
  debug: true
  debug_image_out_serial: true
  
  meter_reader_value_sensor:
    name: "Power Consumption"
    unit_of_measurement: "kWh"
    accuracy_decimals: 1
    
  confidence_sensor:
    name: "Reading Confidence"
    accuracy_decimals: 3
```

### Custom Crop Zones

```yaml
globals:
  - id: crop_zones
    type: string
    initial_value: '[[100,200,140,280],[160,200,200,280],[220,200,260,280]]'

meter_reader_tflite:
  # ... other config
  crop_zones_global: globals.crop_zones
```

## 🔧 Model Training Tips

### Recommended Model Specifications
- **Input size**: 32x20 pixels (RGB)
- **Output classes**: 100 (for 0.0-9.9 range) or 10 (digits 0-9)
- **Quantization**: Use int8 quantization for better performance
- **Training data**: Include various lighting conditions and angles

### Sample Model Configurations
The component includes pre-configured support for:
- **100-class models** (0.0-9.9) with decimal precision
- **10-class models** (digits 0-9) for whole numbers
- **MNIST-style models** for grayscale digit recognition

## 🐛 Troubleshooting

### Common Issues

**❌ Model won't load**
```yaml
# Solution: Increase tensor arena size
meter_reader_tflite:
  tensor_arena_size: 800KB  # Increase from 512KB
```

**❌ Low confidence scores**
```yaml
# Solution: Adjust confidence threshold and check lighting
meter_reader_tflite:
  confidence_threshold: 0.5  # Lower threshold
  debug: true               # Enable debug to see processing
```

**❌ Memory allocation errors**
```yaml
# Solution: Reduce image resolution
esp32_camera:
  resolution: 640x480  # Lower resolution
```

### Debugging Steps

1. **Enable debug mode**:
   ```yaml
   meter_reader_tflite:
     debug: true
     debug_image_out_serial: true
   ```

2. **Check serial output** for detailed processing information

3. **Verify crop zones** match your meter's digit positions

4. **Test with different lighting** conditions

## 📊 Performance Notes

- **Processing time**: 200-1000ms per reading (depends on image size and model)
- **Memory usage**: 400-800KB for tensor arena  image buffers
- **Accuracy**: Typically 90% with well-trained models and good lighting
- **Power usage**: Minimal when using appropriate update intervals

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
git clone https://github.com/nliaudat/meter-reader.git
cd meter-reader/esphome/components/meter_reader_tflite
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [ESPHome](https://esphome.io/)
- Uses [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- Camera support via [ESP32 Camera component](https://github.com/espressif/esp32-camera)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/nliaudat/meter-reader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nliaudat/meter-reader/discussions)
- **ESPHome Community**: [ESPHome Discord](https://discord.gg/K7jNqSbb)

---

**Happy meter reading!** 📊✨

*If this project helps you, please give it a ⭐ on GitHub!*