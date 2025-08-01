// memory_manager.h
#pragma once

#include <memory>
#include <cstdint>
#include "esphome/core/component.h"

namespace esphome {
namespace meter_reader_tflite {

class MemoryManager {
 public:
  struct AllocationResult {
    std::unique_ptr<uint8_t[]> data;
    size_t actual_size;
    
    operator bool() const { return static_cast<bool>(data); }
  };

  static AllocationResult allocate_tensor_arena(size_t requested_size);
  static void report_memory_status();
};

}  // namespace meter_reader_tflite
}  // namespace esphome