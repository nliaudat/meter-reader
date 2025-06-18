# Meter Reader API Documentation

## Base URL
```
http://localhost:5000
```

---

## Endpoints

### 1. Process Meter Readings
**`GET | POST /api/json_response`**  
Process an image of a meter and return digit readings from specified regions.

#### GET Request
```
GET /api/json_response?image_url=<URL>&regions_source=<REGIONS>[&model_path=<PATH>]
```

#### POST Request
```http
POST /api/json_response
Content-Type: application/json
```

**Request Body:**
```json
{
  "image_source": "string",
  "regions_source": "string|array",
  "model_path": "string"
}
```

| Parameter             | Required | Type          | Description                              | Example                                     |
|----------------------|----------|---------------|------------------------------------------|---------------------------------------------|
| image_url/image_source | Yes    | String        | Image URL or file path                   | `"http://192.168.1.113/meter.jpg"`          |
| regions_source       | Yes      | String/Array  | Regions as JSON string, array, or path   | `"[[100,150,200,250]]"`                     |
| model_path           | No       | String        | Path to TFLite model                     | `"custom_model.tflite"`                     |

#### Success Response
```json
{
  "success": true,
  "regions_processed": 8,
  "raw_readings": [5.3, 2.8, 7.1, 3.4],
  "processed_readings": [5, 3, 7, 3],
  "confidence_scores": [0.95, 0.89, 0.92, 0.87],
  "final_reading": 5373
}
```

#### Error Response
```json
{
  "success": false,
  "error": "Missing parameter",
  "details": "image_source"
}
```

---

### 2. Web Interface  
**`GET /`**  
Web interface for manual processing with image upload.

---

### 3. Region Drawing Tool  
**`GET /draw_regions`**  
Interactive tool for defining meter digit regions.

---

## Examples

### cURL GET Request
```bash
curl "http://localhost:5000/api/json_response?image_url=http://192.168.1.113/meter.jpg&regions_source=[[85,232,121,309],[146,230,182,308]]"
```

### Python POST Request
```python
import requests

url = "http://localhost:5000/api/json_response"
data = {
    "image_source": "meter.jpg",
    "regions_source": [[85,232,121,309], [146,230,182,308]]
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Troubleshooting

| Error                    | Solution                                  |
|--------------------------|-------------------------------------------|
| 415 Unsupported Media Type | Add `Content-Type: application/json` header |
| 400 Missing Parameters     | Verify all required parameters are included |
| Empty Regions             | Check region coordinates are within image bounds |
| Low Confidence Scores     | Ensure clear image of meter digits       |

---

## Notes

- Coordinates format: `[x1, y1, x2, y2]`
- Model expects digits to be 0-9 (10 wraps to 0)
- Confidence scores range: 0 (low) to 1 (high)
