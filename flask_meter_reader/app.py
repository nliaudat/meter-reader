from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import tensorflow as tf
import os
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Initialize the MeterReader class
class MeterReader:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

    def preprocess_image(self, image):
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image

    def predict(self, image):
        input_image = self.preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        meter_reading = np.argmax(output[0]) / 10
        return meter_reading

# Load regions from a text file
def load_regions(file_path):
    regions = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                x1, y1, x2, y2 = map(int, line.strip().split(','))
                regions.append((x1, y1, x2, y2))
    except FileNotFoundError:
        flash("Regions file not found.", "error")
    except ValueError:
        flash("Invalid format in regions file. Expected 4 integers per line.", "error")
    return regions

# Save regions to a text file
def save_regions(file_path, regions):
    try:
        with open(file_path, "w") as f:
            for region in regions:
                f.write(f"{region[0]},{region[1]},{region[2]},{region[3]}\n")
        flash("Regions saved successfully.", "success")
    except Exception as e:
        flash(f"Error saving regions: {e}", "error")

# Load image from a local file or URL
def load_image(image_source):
    if image_source.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_source)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        except requests.exceptions.RequestException as e:
            flash(f"Error downloading image: {e}", "error")
            return None
    else:
        if not os.path.exists(image_source):
            flash("Local image file not found.", "error")
            return None
        image = cv2.imread(image_source)
        if image is None:
            flash("Unable to load image.", "error")
        return image

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the image source (file upload or URL)
        image_source = None
        filename = None

        # Check if a file was uploaded
        if "image_file" in request.files:
            file = request.files["image_file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                file_path = os.path.join("static", filename)
                file.save(file_path)
                image_source = file_path

        # If no file was uploaded, check for an image URL
        if not image_source:
            image_source = request.form.get("image_url")
            if not image_source:
                flash("Please provide an image URL or upload a file.", "error")
                return redirect(url_for("index"))
            else:
                # Handle image URL
                try:
                    response = requests.get(image_source)
                    response.raise_for_status()
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        flash("Unable to load image from URL.", "error")
                        return redirect(url_for("index"))
                    # Save the image to the static folder
                    filename = secure_filename(os.path.basename(image_source))
                    file_path = os.path.join("static", filename)
                    cv2.imwrite(file_path, image)
                    image_source = file_path
                except requests.exceptions.RequestException as e:
                    flash(f"Error downloading image: {e}", "error")
                    return redirect(url_for("index"))

        # Load the image
        image = load_image(image_source)
        if image is None:
            return redirect(url_for("index"))

        # Load regions
        regions = load_regions("regions.txt")
        if not regions:
            flash("No regions defined. Please draw regions first.", "error")
            return redirect(url_for("index"))

        # Initialize the MeterReader
        meter_reader = MeterReader("model.tflite")

        # Extract regions and predict readings
        raw_meter_readings = []
        processed_meter_readings = []
        for region in regions:
            x1, y1, x2, y2 = region
            region_image = image[y1:y2, x1:x2]
            raw_reading = meter_reader.predict(region_image)
            raw_meter_readings.append(raw_reading)
            processed_reading = round(raw_reading)
            if processed_reading == 10:
                processed_reading = 0
            processed_meter_readings.append(processed_reading)

        # Concatenate the processed readings
        concatenated_readings = int(''.join(map(str, processed_meter_readings)))

        # Visualize the results
        result_image = image.copy()
        for region, reading in zip(regions, raw_meter_readings):
            x1, y1, x2, y2 = region
            cv2.putText(result_image, f"{reading:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save the result image
        result_filename = f"result_{os.path.basename(image_source)}"
        result_path = os.path.join("static", result_filename)
        cv2.imwrite(result_path, result_image)

        # Render the result page
        return render_template(
            "index.html",
            result_image=result_filename,
            raw_readings=raw_meter_readings,
            processed_readings=processed_meter_readings,
            final_reading=concatenated_readings,
        )

    return render_template("index.html")

# Route for drawing regions
@app.route("/draw_regions", methods=["GET", "POST"])
def draw_regions():
    if request.method == "POST":
        # Get the image source (file upload or URL)
        image_source = None
        filename = None

        # Check if a file was uploaded
        if "image_file" in request.files:
            file = request.files["image_file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                file_path = os.path.join("static", filename)
                file.save(file_path)
                image_source = file_path

        # If no file was uploaded, check for an image URL
        if not image_source:
            image_source = request.form.get("image_url")
            if not image_source:
                flash("Please provide an image URL or upload a file.", "error")
                return redirect(url_for("draw_regions"))
            else:
                # Handle image URL
                try:
                    response = requests.get(image_source)
                    response.raise_for_status()
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        flash("Unable to load image from URL.", "error")
                        return redirect(url_for("draw_regions"))
                    # Save the image to the static folder
                    filename = secure_filename(os.path.basename(image_source))
                    file_path = os.path.join("static", filename)
                    cv2.imwrite(file_path, image)
                    image_source = file_path
                except requests.exceptions.RequestException as e:
                    flash(f"Error downloading image: {e}", "error")
                    return redirect(url_for("draw_regions"))

        # Load the image
        image = load_image(image_source)
        if image is None:
            return redirect(url_for("draw_regions"))

        # Save the image path for drawing
        return render_template("draw_regions.html", image_source=filename)

    return render_template("draw_regions.html")

# Route to save regions
@app.route("/save_regions", methods=["POST"])
def save_regions_route():
    regions = request.json.get("regions")
    if not regions:
        flash("No regions provided.", "error")
        return redirect(url_for("draw_regions"))

    # Save regions to the file
    save_regions("regions.txt", regions)
    return redirect(url_for("index"))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)