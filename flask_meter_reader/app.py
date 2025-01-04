from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import cv2
import numpy as np
import tensorflow as tf
# from tflite_runtime.interpreter import Interpreter
import logging
import os
import json
import ast
import requests
from werkzeug.utils import secure_filename

# Set up logging for better output control
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = "supersecretkey"

class MeterReader:
    def __init__(self, model_path):
        """
        Initialize the MeterReader with a TensorFlow Lite model.
        
        Args:
            model_path (str): Path to the TensorFlow Lite model file.
        """
        print(f"Loading model from: {os.path.abspath(model_path)}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path) # for tensorflow
        # self.interpreter = Interpreter(model_path=model_path)  # for tflite-runtime
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape for preprocessing
        self.input_shape = self.input_details[0]['shape'][1:3]

    def preprocess_image(self, image):
        """
        Preprocess the image for the TensorFlow Lite model.
        
        Args:
            image (numpy.ndarray): Input image (RGB format).
        
        Returns:
            numpy.ndarray: Preprocessed image (normalized, resized).
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided for preprocessing.")

        # Resize the image to the model's input size
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

        # Normalize the image to [0, 1] (uncomment if required by the model)
        # image = image / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image

    def predict(self, image):
        """
        Predict the meter reading from the input image.
        
        Args:
            image (numpy.ndarray): Input image (RGB format).
        
        Returns:
            float: Predicted meter reading.
        """
        # Preprocess the image
        input_image = self.preprocess_image(image)

        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Extract the predicted meter reading
        # Using argmax to handle classification output (e.g., 10 classes for digits 0-9)
        meter_reading = np.argmax(output[0]) / 10

        return meter_reading


def load_regions(regions_source):
    """
    Load regions from a file or a list of coordinates.
    
    Args:
        regions_source (str): Path to the JSON file or a string representation of a list of regions.
        
    Returns:
        list: List of tuples defining the regions (x1, y1, x2, y2).
    """
    try:
        # Check if the input is a file
        if os.path.exists(regions_source):
            with open(regions_source, "r") as f:
                regions = json.load(f)
        else:
            # Assume the input is a string representation of a list
            # Remove any outer quotes and parse the list
            regions_source = regions_source.strip().strip('"').strip("'")
            regions = ast.literal_eval(regions_source)
        
        # Ensure each region has 4 values (x1, y1, x2, y2)
        return [tuple(region) for region in regions if len(region) == 4]
    except (FileNotFoundError, json.JSONDecodeError, SyntaxError, ValueError) as e:
        logging.error(f"Error loading regions: {e}")
        return []
        
# Save regions to a JSON file
def save_regions(file_path, regions):
    try:
        with open(file_path, "w") as f:
            json.dump(regions, f)
        flash("Regions saved successfully.", "success")
    except Exception as e:
        flash(f"Error saving regions: {e}", "error")

def load_image(image_source):
    """
    Load an image from a local file or a remote URL.
    
    Args:
        image_source (str): Path to the local image file or URL of the remote image.
        
    Returns:
        numpy.ndarray: Image in OpenCV format, or None if loading fails.
    """
    if image_source.startswith(('http://', 'https://')):
        # Load image from a remote URL
        try:
            response = requests.get(image_source)
            response.raise_for_status()  # Raise an error for bad responses (e.g., 404)
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image from {image_source}: {e}")
            return None
    else:
        # Load image from a local file
        if not os.path.exists(image_source):
            logging.error(f"Local image file {image_source} not found.")
            return None
        image = cv2.imread(image_source)
        if image is None:
            logging.error(f"Unable to load image from {image_source}.")
        return image


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Home route to handle image upload and prediction.
    """
    if request.method == "POST":
        # Handle POST request (file upload or URL)
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
        regions_source = request.form.get("regions_source", "regions.json")
        regions = load_regions(regions_source)
        if not regions:
            flash("No regions defined. Please provide regions.", "error")
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

    # Handle GET request with query parameters
    if request.method == "GET":
        image_source = request.args.get("image_url")
        regions_source = request.args.get("regions_source")

        if image_source and regions_source:
            # Load the image
            image = load_image(image_source)
            if image is None:
                flash("Unable to load image from URL.", "error")
                return redirect(url_for("index"))

            # Load regions
            regions = load_regions(regions_source)
            if not regions:
                flash("No regions defined. Please provide regions.", "error")
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

    # Save regions to the JSON file
    save_regions("regions.json", regions)
    return jsonify({"message": "Regions saved successfully."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)