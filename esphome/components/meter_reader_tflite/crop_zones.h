#pragma once

#include "esphome/core/component.h"
#include <vector>
#include <string>

namespace esphome {
namespace meter_reader_tflite {

struct CropZone {
  int x1;
  int y1;
  int x2;
  int y2;
};

class CropZoneHandler {
 public:
  void parse_zones(const std::string &zones_json);
  const std::vector<CropZone>& get_zones() const { return zones_; }
  void set_default_zone(int width, int height);

 protected:
  std::vector<CropZone> zones_;
};

}  // namespace meter_reader_tflite
}  // namespace esphome