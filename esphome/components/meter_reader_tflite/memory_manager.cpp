#include "memory_manager.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "MemoryManager";

MemoryManager::AllocationResult MemoryManager::allocate_tensor_arena(size_t requested_size) {
  ESP_LOGD(TAG, "Allocating tensor arena (%zu bytes)", requested_size);
  
  AllocationResult result;
  result.actual_size = requested_size;

  uint8_t *arena_ptr = static_cast<uint8_t*>(
      heap_caps_malloc(requested_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  
  if (!arena_ptr) {
    ESP_LOGW(TAG, "PSRAM allocation failed, trying internal RAM");
    arena_ptr = static_cast<uint8_t*>(
        heap_caps_malloc(requested_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  }

  if (arena_ptr) {
    result.data.reset(arena_ptr);
    ESP_LOGD(TAG, "Successfully allocated tensor arena");
  } else {
    ESP_LOGE(TAG, "Failed to allocate tensor arena");
    result.actual_size = 0;
  }

  return result;
}

void MemoryManager::report_memory_status(size_t requested_size, 
                                       size_t allocated_size,
                                       size_t peak_usage,
                                       size_t model_size) {
  size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  
  if (total_psram > 0) {
    ESP_LOGI(TAG, "PSRAM is available.");
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "  PSRAM: %zuB free of %zuB total (%.1fKB / %.1fKB)", 
             free_psram, total_psram, free_psram / 1024.0f, total_psram / 1024.0f);
  } else {
    ESP_LOGW(TAG, "PSRAM not available");
  }

  ESP_LOGI(TAG, "Memory Status:");
  ESP_LOGI(TAG, "  Requested Arena: %zuB (%.1fKB)", requested_size, requested_size / 1024.0f);
  ESP_LOGI(TAG, "  Allocated Arena: %zuB (%.1fKB)", allocated_size, allocated_size / 1024.0f);
  ESP_LOGI(TAG, "  Arena Peak Usage: %zuB (%.1fKB)", peak_usage, peak_usage / 1024.0f);
  ESP_LOGI(TAG, "  Free Internal Heap: %zuB (%.1fKB)", free_internal, free_internal / 1024.0f);

  if (model_size > 0) {
    float ratio = static_cast<float>(allocated_size) / model_size;
    ESP_LOGI(TAG, "  Arena/Model Ratio: %.1fx", ratio);
  }
}

}  // namespace meter_reader_tflite
}  // namespace esphome