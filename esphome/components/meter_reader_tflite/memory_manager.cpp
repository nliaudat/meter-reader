// memory_manager.cpp
#include "memory_manager.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "MemoryManager";

MemoryManager::AllocationResult MemoryManager::allocate_tensor_arena(size_t requested_size) {
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
  } else {
    ESP_LOGE(TAG, "Failed to allocate tensor arena");
    result.actual_size = 0;
  }

  return result;
}

void MemoryManager::report_memory_status() {
  size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  
  ESP_LOGI(TAG, "Memory Status:");
  ESP_LOGI(TAG, "  Free Internal: %.1fKB", free_internal / 1024.0f);
  
  if (total_psram > 0) {
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "  Free PSRAM: %.1fKB of %.1fKB", 
             free_psram / 1024.0f, total_psram / 1024.0f);
  }
}

}  // namespace meter_reader_tflite
}  // namespace esphome