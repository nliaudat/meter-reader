#include "crop_zones.h"
#include "esp_log.h"
#include <algorithm>

namespace esphome {
namespace meter_reader_tflite {

static const char *const TAG = "CropZoneHandler";

void CropZoneHandler::parse_zones(const std::string &zones_json) {
  zones_.clear();
  std::string stripped = zones_json;
  
  // Remove whitespace
  stripped.erase(std::remove(stripped.begin(), stripped.end(), ' '), stripped.end());
  stripped.erase(std::remove(stripped.begin(), stripped.end(), '\n'), stripped.end());
  stripped.erase(std::remove(stripped.begin(), stripped.end(), '\r'), stripped.end());
  stripped.erase(std::remove(stripped.begin(), stripped.end(), '\t'), stripped.end());

  if (stripped.empty() || stripped == "[]") {
    ESP_LOGD(TAG, "No crop zones defined");
    return;
  }

  // Remove outer brackets
  if (stripped.front() == '[' && stripped.back() == ']') {
    stripped = stripped.substr(1, stripped.length() - 2);
  }

  size_t pos = 0;
  while (pos < stripped.length()) {
    size_t start = stripped.find('[', pos);
    if (start == std::string::npos) break;
    size_t end = stripped.find(']', start);
    if (end == std::string::npos) break;

    std::string zone_str = stripped.substr(start + 1, end - start - 1);
    std::vector<int> coords;
    size_t coord_start = 0;
    
    while (true) {
      size_t comma_pos = zone_str.find(',', coord_start);
      std::string num_str = zone_str.substr(coord_start, comma_pos - coord_start);
      coords.push_back(std::stoi(num_str));
      
      if (comma_pos == std::string::npos) break;
      coord_start = comma_pos + 1;
    }

    if (coords.size() == 4) {
      zones_.push_back({coords[0], coords[1], coords[2], coords[3]});
    } else {
      ESP_LOGE(TAG, "Invalid zone format: %s", zone_str.c_str());
    }

    pos = end + 1;
  }

  ESP_LOGI(TAG, "Parsed %d crop zones", zones_.size());
}

}  // namespace meter_reader_tflite
}  // namespace esphome