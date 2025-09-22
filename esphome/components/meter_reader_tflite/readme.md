# ESPHome Meter Reader TFLite Component

> General-purpose TensorFlow Lite Micro implementation for ESP32 with camera support

[![ESPHome](https://img.shields.io/badge/ESPHome-Compatible-brightgreen)](https://esphome.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🚀 What is this?

A general-purpose ESPHome component for running TensorFlow Lite Micro models on ESP32 devices. While originally designed for meter reading, it can be used for any computer vision task including object detection, image classification, and custom AI applications.

## ✨ Key Features

- **🤖 TensorFlow Lite Micro**: Full TFLite Micro runtime support
- **📷 Camera Integration**: Seamless ESP32 camera integration
- **🖼️ Image Preprocessing**: Automatic cropping, scaling, and format conversion
- **⚡ Optimized Performance**: ESP-NN accelerated operations
- **🎯 Multi-Zone Processing**: Process multiple regions of interest
- **🔧 Flexible Configuration**: Support for various model types and input formats
- **🐛 Advanced Debugging**: Real-time image analysis and model output inspection

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
  name: "AI Camera"
  resolution: 640x480
  pixel_format: JPEG

# Configure TFLite component
meter_reader_tflite:
  id: tflite_processor
  model: "model.tflite"  # Your TensorFlow Lite model
  camera_id: my_camera
  tensor_arena_size: 512KB
  update_interval: 60s
```

### 3. Add Your Model

Place your trained `.tflite` model file in the same directory as your ESPHome configuration.

## 📋 Prerequisites

- **ESP32 board** with camera support
- **ESPHome 2025.08** or newer
- **TensorFlow Lite model** (quantized recommended)

## 🎯 Use Cases

### Computer Vision Applications
- **Object Detection**: Identify objects in camera frames
- **Image Classification**: Categorize images into classes
- **Anomaly Detection**: Detect unusual patterns or events
- **Quality Control**: Inspect products or components
- **Gesture Recognition**: Recognize hand gestures or movements

### Meter Reading (Original Purpose)
- Water, electricity, gas meter digit recognition
- Analog gauge reading
- Digital display extraction

## ⚙️ Configuration Examples

### Basic Object Detection

```yaml
meter_reader_tflite:
  id: object_detector
  model: "object_model.tflite"
  camera_id: my_camera
  tensor_arena_size: 512KB
  update_interval: 30s
  confidence_threshold: 0.7
```

### Image Classification

```yaml
meter_reader_tflite:
  id: image_classifier
  model: "dig-class100-0180-s2-q.tflite"  # Must be in same directory as YAML
  camera_id: my_camera
  tensor_arena_size: 512KB
  update_interval: 60s
  
  # Custom sensor for classification results
  sensor:
    - name: "Detection Confidence"
      id: detection_confidence
```

### Multi-Zone Processing

```yaml
globals:
  - id: detection_zones
    type: string
    initial_value: '[[0,0,400,300],[400,0,800,300],[0,300,400,600],[400,300,800,600]]'

meter_reader_tflite:
  id: multi_zone_detector
  model: "detection_model.tflite"
  camera_id: my_camera
  tensor_arena_size: 512KB
  crop_zones_global: globals.detection_zones
```

### Advanced Configuration with Debugging

```yaml
meter_reader_tflite:
  id: advanced_processor
  model: "custom_model.tflite"
  camera_id: my_camera
  tensor_arena_size: 512KB
  update_interval: 5s
  debug: true
  debug_image_out_serial: true
  confidence_threshold: 0.8
```

## 🔧 Model Configuration

### Supported Model Types

The component automatically detects and configures for various model architectures:

```cpp
// Built-in model configurations (model_config.h)
{"class100-0180", ...}     // 100-class classification (0.0-9.9)
{"class10-0900", ...}      // 10-class classification (0-9)  
{"mnist", ...}             // MNIST-style grayscale models
```

### Custom Model Support

For custom models, the component auto-detects:
- Input dimensions (width, height, channels)
- Data type (float32, uint8)
- Output processing requirements

### Input Formats
- **RGB888**: 3-channel color images
- **GRAYSCALE**: 1-channel monochrome
- **JPEG**: Automatic decoding to RGB888
- **RGB565**: Automatic conversion

## 🎮 API Usage

### Accessing Model Outputs

```yaml
# Example: Create sensors from model outputs
sensor:
  - platform: template
    name: "Model Output 1"
    id: output_1
    lambda: |-
      // Access model outputs in lambdas
      return id(tflite_processor).get_output_value(0);
      
  - platform: template  
    name: "Model Confidence"
    id: output_confidence
    lambda: |-
      return id(tflite_processor).get_confidence();
```

### Custom Output Processing

```cpp
// Example C lambda for custom output handling
auto my_output_processor = [](float* outputs, int output_count) -> float {
    // Custom logic for your model outputs
    float result = 0.0f;
    for (int i = 0; i < output_count; i) {
        result = outputs[i] * custom_weights[i];
    }
    return result;
};
```

## ⚡ Performance Optimization

### Memory Settings

```yaml
meter_reader_tflite:
  tensor_arena_size: 512KB  # Default, adjust based on model size
  
  # For larger models:
  # tensor_arena_size: 768KB
  # tensor_arena_size: 1024KB
```

### Camera Optimization

```yaml
esp32_camera:
  resolution: 640x480      # Lower resolution for faster processing
  pixel_format: RGB888     # Direct processing, no conversion needed
  jpeg_quality: 10         # If using JPEG, lower quality for speed
  framerate: 1 fps         # Reduce frame rate for periodic processing
```

## 🐛 Debugging & Development

### Enable Debug Output

```yaml
meter_reader_tflite:
  debug: true
  debug_image_out_serial: true
```

Debug output includes:
- Model input/output values
- Processing timing statistics
- Memory usage reports
- Image analysis data
- Zone processing details

### Serial Monitoring

```bash
# Monitor debug output
esphome logs your-config.yaml

# Expected debug output example:
# DEBUG: Model input: 32x20x3, output: 10 classes
# DEBUG: Processing time: 45ms
# DEBUG: Zone 1 confidence: 0.92
# DEBUG: Memory usage: 412KB/512KB
```

## 🔍 Troubleshooting

### Common Issues & Solutions

**❌ Model loading fails**
```yaml
# Increase tensor arena size
meter_reader_tflite:
  tensor_arena_size: 768KB  # Increase from 512KB
```

**❌ Poor inference results**
```yaml
# Check model compatibility and preprocessing
meter_reader_tflite:
  debug: true  # Enable debug to see input data
```

**❌ Camera frame issues**
```yaml
# Adjust camera settings
esp32_camera:
  resolution: 320x240  # Lower resolution
  framerate: 1 fps     # Reduce frame rate
```

### Performance Tips

1. **Use quantized models** (int8) for better performance
2. **Match input dimensions** to your actual use case
3. **Enable ESP-NN optimizations** (enabled by default)
4. **Use appropriate tensor_arena_size** (start with 512KB)
5. **Optimize update interval** based on your application needs

## 📊 Technical Specifications

- **TensorFlow Lite Version**: Micro 1.3.4
- **ESP-NN Optimization**: Enabled by default
- **Memory Usage**: 512KB tensor arena (configurable)
- **Processing Speed**: 10-1000ms per inference (model-dependent)
- **Supported Operations**: Conv2D, FullyConnected, Pooling, Activation functions, etc.
- **Input Types**: float32, int8, uint8

## 🤝 Contributing

This component is designed to be extensible for various TFLite applications. Contributions welcome!

### Extension Ideas
- Object detection bounding box support
- Multi-model switching
- Custom preprocessing pipelines
- Cloud model updates
- Advanced output decoding

## 📄 License

* Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA)
* No commercial use
* The AI model from haverland is under Apache Licence

## 🌟 Acknowledgments

- TensorFlow Lite Micro team
- ESPHome community
- ESP-NN for performance optimizations

---

**Transform your ESP32 camera into an AI vision system!** 🤖✨

*For questions and support, check the GitHub repository discussions section.*