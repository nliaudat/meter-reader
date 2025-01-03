# Flask Meter Reader

**Meter Reader** is a Flask-based web application that uses TensorFlow Lite to predict meter readings from images. It allows users to upload an image or provide an image URL, define regions of interest, and get the predicted readings for each region.

---

## Features

- **Image Processing**:
  - Upload images or provide image URLs.
  - Automatically process images to extract meter readings.

- **Region Definition**:
  - Define regions of interest (ROIs) using a JSON file or a string representation of a list.
  - Draw regions interactively using the **Draw Regions** feature.

- **Meter Reading Prediction**:
  - Predict meter readings for each region using a TensorFlow Lite model.
  - Display raw and processed readings, along with a final concatenated reading.

- **User-Friendly Interface**:
  - Simple and intuitive web interface.
  - Real-time feedback and error handling.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow Lite
- Flask
- OpenCV
- NumPy

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nliaudat/meter-reader.git
   cd meter-reader/flask_meter_reader
   ```

2. **Set Up a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```bash
    python app.py
    ```

5. **Access the Application:

        Open your browser and navigate to http://localhost:5000/.

    Upload an Image:

        Provide an image URL or upload an image file.

        Define regions of interest using a JSON file or a string representation of a list (e.g., [[x1, y1, x2, y2], [x1, y1, x2, y2]]).

        Click Process to predict meter readings.

    View Results:

        The processed image, raw readings, processed readings, and final reading will be displayed.

Draw Regions Page (/draw_regions)

    Draw Regions Interactively:

        Upload an image or provide an image URL.

        Draw regions on the canvas by clicking to define points.

        Save the regions to regions.json.

Example
Input

    Image URL: http://example.com/meter_image.jpg

    Regions: [[80,233,116,307],[144,235,180,307],[202,234,238,308]]

Output

    Processed Image: Displayed with regions highlighted.

    Raw Readings: [5.3, 6.7, 7.1]

    Processed Readings: [5, 7, 7]

    Final Reading: 577
