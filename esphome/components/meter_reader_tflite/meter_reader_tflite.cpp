#include "meter_reader_tflite.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <cstdarg> // for va_list
#include <cstdio>  // for vsnprintf
#include <cstring> // for strchr, strlen
#include <set>     // for std::set

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "meter_reader_tflite";

// Helper function to dump tensor contents for debugging.
static void hexdump_tensor(const char *tag, const TfLiteTensor *tensor) {
  if (tensor == nullptr) {
    ESP_LOGW(tag, "Attempted to hexdump a null tensor.");
    return;
  }
  // The 'name' field is removed in newer TFLite versions.
  ESP_LOGD(tag, "Hexdump of tensor (%zu bytes, type %d):", tensor->bytes, tensor->type);
  ESP_LOG_BUFFER_HEXDUMP(tag, tensor->data.data, tensor->bytes, ESP_LOG_DEBUG);
}

void MeterReaderTFLite::setup() {
  ESP_LOGI(TAG, "Setting up Meter Reader TFLite...");
  // Add a delay to allow other components (like the logger and camera) to initialize
  // before we attempt the resource-intensive model loading. This can prevent
  // crashes on startup.
  this->set_timeout(1000, [this]() {
    if (!this->load_model()) {
      ESP_LOGE(TAG, "Failed to load model. Marking component as failed.");
      this->mark_failed();
      return;
    }
    this->model_loaded_ = true;
    ESP_LOGI(TAG, "Meter Reader TFLite setup complete");
  });
}

bool MeterReaderTFLite::load_model() {
  ESP_LOGD(TAG, "load_model: start");
  if (model_ == nullptr || model_length_ == 0) {
    ESP_LOGE(TAG, "No model data available");
    return false;
  }

  ESP_LOGI(TAG, "Loading model (%zu bytes)", model_length_);

  // Log PSRAM status, as it's crucial for large tensor arenas
  if (heap_caps_get_total_size(MALLOC_CAP_SPIRAM) > 0) {
    ESP_LOGI(TAG, "PSRAM is available.");
  } else {
    ESP_LOGW(TAG, "PSRAM not available. Large tensor arenas may fail to allocate from internal RAM.");
  }

  ESP_LOGD(TAG, "load_model: calling GetModel()");
  tflite_model_ = tflite::GetModel(model_);
  if (tflite_model_ == nullptr) {
    ESP_LOGE(TAG, "Failed to get model from buffer. The model data may be corrupt or invalid.");
    return false;
  }

  ESP_LOGD(TAG, "load_model: checking schema version");
  if (tflite_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch");
    return false;
  }

  ESP_LOGD(TAG, "load_model: allocating tensor arena");
  if (!allocate_tensor_arena()) {
    return false;
  }

  ESP_LOGD(TAG, "load_model: creating op resolver");
  // Create resolver with automatic operation detection.
  // Using a larger number to support a wide variety of models.
  // This can be tuned down for a specific model to save a little RAM.
  static tflite::MicroMutableOpResolver<90> resolver;

  ESP_LOGD(TAG, "load_model: parsing operators");
  // Get model subgraph and operators
  ESP_LOGD(TAG, "Parsing model operators...");
  const auto* subgraphs = tflite_model_->subgraphs();
  if (subgraphs->size() != 1) {
    ESP_LOGE(TAG, "Only single subgraph models are supported");
    return false;
  }

  const auto* ops = (*subgraphs)[0]->operators();
  const auto* opcodes = tflite_model_->operator_codes();

  std::set<int> added_ops;

  // Add required operations to the resolver
  for (size_t i = 0; i < ops->size(); i++) {
    const auto* op = (*ops)[i];
    const auto* opcode = (*opcodes)[op->opcode_index()];
    const auto builtin_code = opcode->builtin_code();

    if (added_ops.count(builtin_code)) {
      continue;  // Operator already added, skip.
    }

    const char *op_name = tflite::EnumNameBuiltinOperator(builtin_code);
    ESP_LOGD(TAG, "Model requires op: %s", op_name);

    TfLiteStatus add_status = kTfLiteError;
    switch (builtin_code) {
      // Manually ordered for commonality
      case tflite::BuiltinOperator_CONV_2D:
        add_status = resolver.AddConv2D();
        break;
      case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        add_status = resolver.AddDepthwiseConv2D();
        break;
      case tflite::BuiltinOperator_FULLY_CONNECTED:
        add_status = resolver.AddFullyConnected();
        break;
      case tflite::BuiltinOperator_ADD:
        add_status = resolver.AddAdd();
        break;
      case tflite::BuiltinOperator_MUL:
        add_status = resolver.AddMul();
        break;
      case tflite::BuiltinOperator_MAX_POOL_2D:
        add_status = resolver.AddMaxPool2D();
        break;
      case tflite::BuiltinOperator_AVERAGE_POOL_2D:
        add_status = resolver.AddAveragePool2D();
        break;
      case tflite::BuiltinOperator_RESHAPE:
        add_status = resolver.AddReshape();
        break;
      case tflite::BuiltinOperator_QUANTIZE:
        add_status = resolver.AddQuantize();
        break;
      case tflite::BuiltinOperator_DEQUANTIZE:
        add_status = resolver.AddDequantize();
        break;
      case tflite::BuiltinOperator_SOFTMAX:
        add_status = resolver.AddSoftmax();
        break;
      case tflite::BuiltinOperator_RELU:
        add_status = resolver.AddRelu();
        break;
      case tflite::BuiltinOperator_RELU6:
        add_status = resolver.AddRelu6();
        break;
      case tflite::BuiltinOperator_LOGISTIC:
        add_status = resolver.AddLogistic();
        break;
      case tflite::BuiltinOperator_SUB:
        add_status = resolver.AddSub();
        break;
      case tflite::BuiltinOperator_CONCATENATION:
        add_status = resolver.AddConcatenation();
        break;
      case tflite::BuiltinOperator_MEAN:
        add_status = resolver.AddMean();
        break;
      case tflite::BuiltinOperator_PAD:
        add_status = resolver.AddPad();
        break;
      case tflite::BuiltinOperator_PADV2:
        add_status = resolver.AddPadV2();
        break;
      case tflite::BuiltinOperator_STRIDED_SLICE:
        add_status = resolver.AddStridedSlice();
        break;

      // The rest in alphabetical order for completeness
      case tflite::BuiltinOperator_ABS:
        add_status = resolver.AddAbs();
        break;
      case tflite::BuiltinOperator_ADD_N:
        add_status = resolver.AddAddN();
        break;
      case tflite::BuiltinOperator_ARG_MAX:
        add_status = resolver.AddArgMax();
        break;
      case tflite::BuiltinOperator_ARG_MIN:
        add_status = resolver.AddArgMin();
        break;
      case tflite::BuiltinOperator_ASSIGN_VARIABLE:
        add_status = resolver.AddAssignVariable();
        break;
      case tflite::BuiltinOperator_BATCH_TO_SPACE_ND:
        add_status = resolver.AddBatchToSpaceNd();
        break;
      case tflite::BuiltinOperator_BROADCAST_ARGS:
        add_status = resolver.AddBroadcastArgs();
        break;
      case tflite::BuiltinOperator_BROADCAST_TO:
        add_status = resolver.AddBroadcastTo();
        break;
      case tflite::BuiltinOperator_CALL_ONCE:
        add_status = resolver.AddCallOnce();
        break;
      case tflite::BuiltinOperator_CAST:
        add_status = resolver.AddCast();
        break;
      case tflite::BuiltinOperator_CEIL:
        add_status = resolver.AddCeil();
        break;
      case tflite::BuiltinOperator_COS:
        add_status = resolver.AddCos();
        break;
      case tflite::BuiltinOperator_CUMSUM:
        add_status = resolver.AddCumSum();
        break;
      case tflite::BuiltinOperator_DEPTH_TO_SPACE:
        add_status = resolver.AddDepthToSpace();
        break;
      case tflite::BuiltinOperator_DIV:
        add_status = resolver.AddDiv();
        break;
      case tflite::BuiltinOperator_ELU:
        add_status = resolver.AddElu();
        break;
      case tflite::BuiltinOperator_EQUAL:
        add_status = resolver.AddEqual();
        break;
      case tflite::BuiltinOperator_EXP:
        add_status = resolver.AddExp();
        break;
      case tflite::BuiltinOperator_EXPAND_DIMS:
        add_status = resolver.AddExpandDims();
        break;
      case tflite::BuiltinOperator_FILL:
        add_status = resolver.AddFill();
        break;
      case tflite::BuiltinOperator_FLOOR:
        add_status = resolver.AddFloor();
        break;
      case tflite::BuiltinOperator_FLOOR_DIV:
        add_status = resolver.AddFloorDiv();
        break;
      case tflite::BuiltinOperator_FLOOR_MOD:
        add_status = resolver.AddFloorMod();
        break;
      case tflite::BuiltinOperator_GATHER:
        add_status = resolver.AddGather();
        break;
      case tflite::BuiltinOperator_GATHER_ND:
        add_status = resolver.AddGatherNd();
        break;
      case tflite::BuiltinOperator_GREATER:
        add_status = resolver.AddGreater();
        break;
      case tflite::BuiltinOperator_GREATER_EQUAL:
        add_status = resolver.AddGreaterEqual();
        break;
      case tflite::BuiltinOperator_HARD_SWISH:
        add_status = resolver.AddHardSwish();
        break;
      case tflite::BuiltinOperator_IF:
        add_status = resolver.AddIf();
        break;
      case tflite::BuiltinOperator_L2_NORMALIZATION:
        add_status = resolver.AddL2Normalization();
        break;
      case tflite::BuiltinOperator_L2_POOL_2D:
        add_status = resolver.AddL2Pool2D();
        break;
      case tflite::BuiltinOperator_LEAKY_RELU:
        add_status = resolver.AddLeakyRelu();
        break;
      case tflite::BuiltinOperator_LESS:
        add_status = resolver.AddLess();
        break;
      case tflite::BuiltinOperator_LESS_EQUAL:
        add_status = resolver.AddLessEqual();
        break;
      case tflite::BuiltinOperator_LOG:
        add_status = resolver.AddLog();
        break;
      case tflite::BuiltinOperator_LOGICAL_AND:
        add_status = resolver.AddLogicalAnd();
        break;
      case tflite::BuiltinOperator_LOGICAL_NOT:
        add_status = resolver.AddLogicalNot();
        break;
      case tflite::BuiltinOperator_LOGICAL_OR:
        add_status = resolver.AddLogicalOr();
        break;
      case tflite::BuiltinOperator_LOG_SOFTMAX:
        add_status = resolver.AddLogSoftmax();
        break;
      case tflite::BuiltinOperator_MAXIMUM:
        add_status = resolver.AddMaximum();
        break;
      case tflite::BuiltinOperator_MINIMUM:
        add_status = resolver.AddMinimum();
        break;
      case tflite::BuiltinOperator_MIRROR_PAD:
        add_status = resolver.AddMirrorPad();
        break;
      case tflite::BuiltinOperator_NEG:
        add_status = resolver.AddNeg();
        break;
      case tflite::BuiltinOperator_NOT_EQUAL:
        add_status = resolver.AddNotEqual();
        break;
      case tflite::BuiltinOperator_PACK:
        add_status = resolver.AddPack();
        break;
      case tflite::BuiltinOperator_READ_VARIABLE:
        add_status = resolver.AddReadVariable();
        break;
      case tflite::BuiltinOperator_SPLIT_V:
        add_status = resolver.AddSplitV();
        break;
      case tflite::BuiltinOperator_VAR_HANDLE:
        add_status = resolver.AddVarHandle();
        break;
      default:
        ESP_LOGE(TAG, "Unsupported operator: %s (%d)", op_name, builtin_code);
        return false;
    }

    if (add_status != kTfLiteOk) {
      ESP_LOGE(TAG, "Failed to add operator %s to resolver. "
                    "This may be because the MicroMutableOpResolver's template size is too small.", op_name);
      return false;
    }
    // Mark this operator as added so we don't try to add it again.
    added_ops.insert(builtin_code);
  }

  ESP_LOGD(TAG, "load_model: creating interpreter");
  ESP_LOGD(TAG, "Creating interpreter...");
  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      tflite_model_,
      resolver,
      tensor_arena_.get(),
      tensor_arena_size_actual_);

  ESP_LOGD(TAG, "load_model: allocating tensors");
  ESP_LOGD(TAG, "Allocating tensors...");
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    // The error reporter will have already logged the detailed reason.
    ESP_LOGE(TAG, "Failed to allocate tensors. Check logs for details from tflite_micro.");
    return false;
  }

  ESP_LOGD(TAG, "load_model: success");
  ESP_LOGI(TAG, "Model loaded successfully");
  report_memory_status();
  return true;
}

bool MeterReaderTFLite::allocate_tensor_arena() {
  #ifdef ESP_NN
  ESP_LOGI(TAG, "ESP-NN optimizations are enabled");
  #else
  ESP_LOGW(TAG, "ESP-NN not enabled - using default kernels");
  #endif

  tensor_arena_size_actual_ = tensor_arena_size_requested_;

  // Allocate from PSRAM if available, otherwise fall back to internal RAM.
  // Using heap_caps_malloc is more robust for large allocations on ESP32.
  uint8_t *arena_ptr = (uint8_t *) heap_caps_malloc(tensor_arena_size_actual_, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  if (arena_ptr == nullptr) {
    ESP_LOGW(TAG, "Could not allocate tensor arena from PSRAM, trying internal RAM.");
    arena_ptr = (uint8_t *) heap_caps_malloc(tensor_arena_size_actual_, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }

  if (arena_ptr == nullptr) {
    ESP_LOGE(TAG, "Failed to allocate tensor arena from both PSRAM and internal RAM. "
                  "Try reducing tensor_arena_size.");
    return false;
  }

  // Use reset() to assign the raw pointer to the unique_ptr with the custom deleter.
  tensor_arena_.reset(arena_ptr);

  return true;
}

void MeterReaderTFLite::report_memory_status() {
  size_t free_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  ESP_LOGI(TAG, "Memory Status:");
  ESP_LOGI(TAG, "  Requested Arena: %zuB (%.1fKB)", 
          tensor_arena_size_requested_, tensor_arena_size_requested_/1024.0f);
  ESP_LOGI(TAG, "  Allocated Arena: %zuB (%.1fKB)", 
          tensor_arena_size_actual_, tensor_arena_size_actual_/1024.0f);
  ESP_LOGI(TAG, "  Free Heap: %zuB (%.1fKB)", free_heap, free_heap/1024.0f);
  
  if (model_length_ > 0) {
    float ratio = static_cast<float>(tensor_arena_size_actual_) / model_length_;
    ESP_LOGI(TAG, "  Arena/Model Ratio: %.1fx", ratio);
  }
}

void MeterReaderTFLite::loop() {
  // Inference logic will go here
}

}  // namespace meter_reader_tflite
}  // namespace esphome