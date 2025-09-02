# Agents Guide for `meter_reader_tflite`

## Overview
- **Component name**: `meter_reader_tflite`
- **Type**: ESPHome external component (custom domain, not a sensor platform)
- **Purpose**: Run **TensorFlow Lite Micro** on ESP32 using an **ESP32 camera** to read meter values and optionally publish them as a sensor.
- **Implements**: `MeterReaderTFLite` (C++), derived from `PollingComponent`.

This component loads a `.tflite` model into program memory, captures images from an ESP32 camera, runs inference, and (optionally) exposes the reading via a standard ESPHome sensor entity.

---

## Dependencies (from code)
- ESPHome components:
  - `esp32`
  - `camera` (via `esp32_camera` import)
  - `sensor` (auto-loaded if you enable the sensor output)
- ESP-IDF components (added automatically during codegen):
  - `espressif/esp-tflite-micro` @ `1.3.3~1`
  - `espressif/esp_new_jpeg` @ `0.6.1`
- Build flags set by the component:
  - `-DTF_LITE_STATIC_MEMORY`
  - `-DTF_LITE_DISABLE_X86_NEON`
  - `-DESP_NN`
  - `-DUSE_ESP32_CAMERA_CONV`

---

## Configuration schema (as implemented)
Top-level key: **`meter_reader_tflite:`** (polling component schema, default `update_interval: 60s`)

| Option | Type | Default | Notes |
|---|---|---|---|
| `id` | ID | — | Required component ID. |
| `model` | file path | — | Required path to a `.tflite` file; baked into a PROGMEM array. |
| `camera_id` | ID (to `esp32_camera`) | — | Required camera reference (`esp32_camera.ESP32Camera`). |
| `confidence_threshold` | float (0–1) | `0.7` | Minimum confidence to accept results. |
| `tensor_arena_size` | size (B/KB/MB) | `800KB` | Parsed from strings like `512KB`, validated to **100KB–800KB**. |
| `meter_reader_value_sensor` | sensor schema | — | Optional sensor to publish readings; sets `accuracy_decimals: 2`. |
| `debug` | boolean | `false` | If true, loads and processes a bundled `debug.jpg` instead of live camera. |
| `update_interval` | time | `60s` | From polling component schema. |

**Notes on substitutions used by the component:**
- `substitutions.camera_resolution` (e.g., `"800x600"`): parsed to set width/height. Defaults to `800x600` when unspecified.
- `substitutions.camera_pixel_format` (`RGB888` or `JPEG`): default `RGB888`. If set to `JPEG`, the component configures JPEG input via `set_camera_image_format(width, height, "JPEG")`.

---

## Minimal working example (YAML)

```yaml
external_components:
  - source:
      type: local
      path: ./esphome/components
    components: [meter_reader_tflite]

substitutions:
  camera_resolution: "800x600"
  camera_pixel_format: "RGB888"  # or "JPEG"

esp32_camera:
  id: cam0
  name: "Meter Camera"
  resolution: 800x600
  pixel_format: RGB888  # match substitutions if you use them

# Component configuration (top-level key, not a sensor platform)
meter_reader_tflite:
  id: mr0
  model: "model.tflite"
  camera_id: cam0
  confidence_threshold: 0.8
  tensor_arena_size: 512KB
  update_interval: 60s
  meter_reader_value_sensor:
    name: "Meter Reading"
  debug: false
```

> If `camera_pixel_format` substitution is `"JPEG"`, the component will automatically call `set_camera_image_format(width, height, "JPEG")`. In debug mode it will also set `set_camera_image_format(640, 480, "JPEG")` for the bundled debug image run.

---

## Behavior details

1. **Model provisioning**
   - The `.tflite` file is read at compile time and converted to a `static PROGMEM` byte array.
   - The array and its length are passed into the component via `set_model()`.

2. **Camera setup**
   - The referenced `esp32_camera` component is injected via `set_camera(...)`.
   - When `camera_pixel_format == "JPEG"`, `set_camera_image_format(width, height, "JPEG")` is applied.

3. **Inference cycle**
   - On each `update_interval`, the component captures an image and runs inference using TFLite Micro with a statically allocated **tensor arena** (`tensor_arena_size`).
   - Results are filtered by `confidence_threshold`.
   - If `meter_reader_value_sensor` is configured, the value is published with `accuracy_decimals: 2`.

4. **Debug mode**
   - When `debug: true`:
     - `DEBUG_METER_READER_TFLITE` is defined.
     - A `debug.jpg` file is required in the component directory; it is embedded into a `static const uint8_t[]` at compile time.
     - The component calls `set_debug_image(...)` and immediately triggers `test_with_debug_image()`.
     - Camera format is forced to JPEG at `640x480` for this test path.

5. **Services / Defines**
   - `USE_SERVICE_DEBUG` is defined by the component. (The corresponding C++ code can use this to expose a service such as `meter_reader_tflite_<id>_debug`.)

---

## Practical tips
- Start with `tensor_arena_size: 512KB` and adjust if the model fails due to arena exhaustion; max allowed by schema is **800KB**.
- Keep `camera_pixel_format: RGB888` unless your pipeline explicitly requires JPEG. RGB input avoids extra JPEG decode work.
- Ensure `model.tflite` is reasonably small (Flash and RAM are limited). Quantized models are recommended for ESP32.
- If enabling `debug: true`, place a valid `debug.jpg` in `esphome/components/meter_reader_tflite/` before compiling, otherwise validation will fail.

---

## File layout (suggested)
```
esphome/
└── components/
    └── meter_reader_tflite/
        ├── __init__.py          # (this schema/codegen file)
        ├── meter_reader_tflite.h
        ├── meter_reader_tflite.cpp
        └── debug.jpg            # required only when debug: true
```

---

## Changelog pointers
- Adds IDF components: `esp-tflite-micro@1.3.3~1`, `esp_new_jpeg@0.6.1`
- Defines: `TF_LITE_STATIC_MEMORY`, `TF_LITE_DISABLE_X86_NEON`, `ESP_NN`, `USE_ESP32_CAMERA_CONV`, `USE_SERVICE_DEBUG` (always), `DEBUG_METER_READER_TFLITE` (when `debug: true`).
- Accepts substitutions: `camera_resolution`, `camera_pixel_format`.
