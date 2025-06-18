# Import required libraries
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import cv2  # OpenCV for image processing
import numpy as np  # Numerical computing
# import tensorflow as tf  # TensorFlow (commented out as using tflite-runtime)
from tflite_runtime.interpreter import Interpreter  # TensorFlow Lite interpreter
import logging  # For logging messages
import os  # For file system operations
import json  # For JSON parsing and serialization
import ast  # For safely evaluating strings containing Python expressions
import requests  # For making HTTP requests (to fetch remote images)
from werkzeug.utils import secure_filename  # For secure filename handling

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Define softmax function for TensorFlow Lite compatibility
def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    Args:
        x: Input array of logits
    Returns:
        Array of probabilities where the sum of values equals 1
    """
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Secret key for session management

class MeterReader:
    """
    Class for reading meter values using a TensorFlow Lite model.
    Handles model loading, image preprocessing, and prediction.
    """
    
    def __init__(self, model_path):
        """
        Initialize the MeterReader with a TensorFlow Lite model.
        Args:
            model_path: Path to the TensorFlow Lite model file
        """
        print(f"Loading model from: {os.path.abspath(model_path)}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the TensorFlow Lite model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()  # Allocate tensors for model

        # Get model input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape (excluding batch dimension)
        self.input_shape = self.input_details[0]['shape'][1:3]

    def preprocess_image(self, image):
        """
        Preprocess image for model input.
        Args:
            image: Input image in numpy array format
        Returns:
            Preprocessed image ready for model inference
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided for preprocessing.")

        # Resize image to model's expected input size
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

        # Add batch dimension and convert to float32
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image

    def predict(self, image):
        """
        Predict meter reading from an image.
        Args:
            image: Input image in numpy array format
        Returns:
            tuple: (meter_reading, confidence_score)
        """
        # Preprocess the image
        input_image = self.preprocess_image(image)

        # Set input tensor and run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()

        # Get output tensor (logits)
        logits = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Convert logits to probabilities using softmax
        probabilities = softmax(logits[0])

        # Get predicted class and confidence score
        predicted_class = np.argmax(probabilities)
        confidence_score = np.max(probabilities)

        # Convert class index to meter reading (0-99 -> 0.0-9.9)
        meter_reading = predicted_class / 10
        return meter_reading, confidence_score

def load_regions(regions_source):
    """
    Load regions from a file or string representation.
    Args:
        regions_source: Path to JSON file or string representation of regions
    Returns:
        List of tuples defining regions (x1, y1, x2, y2)
    """
    try:
        # Check if source is a file
        if os.path.exists(regions_source):
            with open(regions_source, "r") as f:
                regions = json.load(f)
        else:
            # Handle string representation
            regions_source = regions_source.strip().strip('"').strip("'")
            regions = ast.literal_eval(regions_source)
        
        # Ensure each region has exactly 4 coordinates
        return [tuple(region) for region in regions if len(region) == 4]
    except (FileNotFoundError, json.JSONDecodeError, SyntaxError, ValueError) as e:
        logging.error(f"Error loading regions: {e}")
        return []

def save_regions(file_path, regions):
    """
    Save regions to a JSON file.
    Args:
        file_path: Path to save the JSON file
        regions: List of regions to save
    """
    try:
        with open(file_path, "w") as f:
            json.dump(regions, f)
        flash("Regions saved successfully.", "success")
    except Exception as e:
        flash(f"Error saving regions: {e}", "error")

def load_image(image_source):
    """
    Load image from local file or remote URL.
    Args:
        image_source: Path to local file or URL
    Returns:
        Image as numpy array or None if loading fails
    """
    if image_source.startswith(('http://', 'https://')):
        # Handle remote URL
        try:
            response = requests.get(image_source)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image from {image_source}: {e}")
            return None
    else:
        # Handle local file
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
    Main route for web interface.
    Handles image upload/URL submission and displays results.
    """
    if request.method == "POST":
        # Handle file upload
        image_source = None
        filename = None
        if "image_file" in request.files:
            file = request.files["image_file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                file_path = os.path.join("static", filename)
                file.save(file_path)
                image_source = file_path

        # Handle image URL if no file uploaded
        if not image_source:
            image_source = request.form.get("image_url")
            if not image_source:
                flash("Please provide an image URL or upload a file.", "error")
                return redirect(url_for("index"))
            else:
                try:
                    # Download and save remote image
                    response = requests.get(image_source)
                    response.raise_for_status()
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        flash("Unable to load image from URL.", "error")
                        return redirect(url_for("index"))
                    filename = secure_filename(os.path.basename(image_source))
                    file_path = os.path.join("static", filename)
                    cv2.imwrite(file_path, image)
                    image_source = file_path
                except requests.exceptions.RequestException as e:
                    flash(f"Error downloading image: {e}", "error")
                    return redirect(url_for("index"))

        # Load image and validate
        image = load_image(image_source)
        if image is None:
            return redirect(url_for("index"))

        # Load regions
        regions_source = request.form.get("regions_source", "regions.json")
        regions = load_regions(regions_source)
        if not regions:
            flash("No regions defined. Please provide regions.", "error")
            return redirect(url_for("index"))

        # Initialize meter reader and process regions
        meter_reader = MeterReader("model.tflite")
        raw_meter_readings = []
        processed_meter_readings = []
        confidence_scores = []
        
        for region in regions:
            x1, y1, x2, y2 = region
            region_image = image[y1:y2, x1:x2]
            raw_reading, confidence = meter_reader.predict(region_image)
            raw_meter_readings.append(raw_reading)
            confidence_scores.append(confidence)

            # Process reading (round and handle 10->0 wrap-around)
            processed_reading = round(raw_reading)
            if processed_reading == 10:
                processed_reading = 0
            processed_meter_readings.append(processed_reading)

        # Concatenate processed readings
        concatenated_readings = int(''.join(map(str, processed_meter_readings)))

        # Create result visualization
        result_image = image.copy()
        for i, (region, reading) in enumerate(zip(regions, raw_meter_readings)):
            x1, y1, x2, y2 = region
            # Draw bounding box and reading text
            cv2.putText(result_image, f"{reading:.1f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optionally show confidence score
            if not request.form.get("no_confidence"):
                confidence = confidence_scores[i]
                cv2.putText(result_image, f"{int(round(confidence * 100))}%", 
                           (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 165, 255), 2)

        # Save result image
        result_filename = f"result_{os.path.basename(image_source)}"
        result_path = os.path.join("static", result_filename)
        cv2.imwrite(result_path, result_image)

        # Render results template
        return render_template(
            "index.html",
            result_image=result_filename,
            raw_readings=raw_meter_readings,
            processed_readings=processed_meter_readings,
            confidence_scores=[int(round(score * 100)) for score in confidence_scores],
            final_reading=concatenated_readings,
        )

    return render_template("index.html")

@app.route("/draw_regions", methods=["GET", "POST"])
def draw_regions():
    """
    Route for drawing regions on an image.
    Handles image upload and serves drawing interface.
    """
    if request.method == "POST":
        # Handle file upload
        image_source = None
        filename = None
        if "image_file" in request.files:
            file = request.files["image_file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                file_path = os.path.join("static", filename)
                file.save(file_path)
                image_source = file_path

        # Handle image URL if no file uploaded
        if not image_source:
            image_source = request.form.get("image_url")
            if not image_source:
                flash("Please provide an image URL or upload a file.", "error")
                return redirect(url_for("draw_regions"))
            else:
                try:
                    # Download and save remote image
                    response = requests.get(image_source)
                    response.raise_for_status()
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        flash("Unable to load image from URL.", "error")
                        return redirect(url_for("draw_regions"))
                    filename = secure_filename(os.path.basename(image_source))
                    file_path = os.path.join("static", filename)
                    cv2.imwrite(file_path, image)
                    image_source = file_path
                except requests.exceptions.RequestException as e:
                    flash(f"Error downloading image: {e}", "error")
                    return redirect(url_for("draw_regions"))

        # Load image and validate
        image = load_image(image_source)
        if image is None:
            return redirect(url_for("draw_regions"))

        # Serve drawing interface with image
        return render_template("draw_regions.html", image_source=filename)

    return render_template("draw_regions.html")

@app.route("/save_regions", methods=["POST"])
def save_regions_route():
    """
    API endpoint for saving drawn regions.
    Accepts JSON with regions data.
    """
    regions = request.json.get("regions")
    if not regions:
        flash("No regions provided.", "error")
        return redirect(url_for("draw_regions"))

    # Save regions using existing function
    save_regions("regions.json", regions)
    return jsonify({"message": "Regions saved successfully."})

    
@app.route('/api/json_response', methods=["GET", "POST"])
def json_response():
    """
    API endpoint to process meter readings from an image and regions.
    Accepts JSON input with:
    - image_source (required): URL or file path of the image
    - regions (required): List of regions or path to JSON file
    - model_path (optional): Path to the model file (default: model.tflite)
    
    Returns JSON response with:
    - success: Boolean indicating success status
    - raw_readings: List of raw predictions
    - processed_readings: List of processed predictions
    - confidence_scores: List of confidence scores
    - final_reading: Concatenated processed readings
    - error: Optional error message if something went wrong
    """
    try:
        # Determine if we're handling GET or POST request
        if request.method == "GET":
            # Extract parameters from URL query string for GET request
            data = {
                "image_source": request.args.get("image_url"),
                "regions_source": request.args.get("regions_source"),
                "model_path": request.args.get("model_path", "model.tflite")
            }
        else:
            # Parse JSON body for POST request
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No JSON data provided"}), 400

        # Validate required parameters
        if not data.get("image_source") and not data.get("image_url"):
            return jsonify({"success": False, "error": "Missing required parameter: image_source/image_url"}), 400
        if not data.get("regions_source"):
            return jsonify({"success": False, "error": "Missing required parameter: regions_source"}), 400

        # Use image_url if image_source wasn't provided (for GET requests)
        image_source = data.get("image_source") or data.get("image_url")
        regions_source = data.get("regions_source")
        model_path = data.get("model_path", "model.tflite")

        # Rest of your existing processing code...
        # Load image using existing function
        image = load_image(image_source)
        if image is None:
            return jsonify({"success": False, "error": f"Unable to load image from {image_source}"}), 400

        # Load regions - handle string representation for GET requests
        regions = load_regions(regions_source)
        if not regions:
            return jsonify({"success": False, "error": "No valid regions provided"}), 400

        # Initialize meter reader
        try:
            meter_reader = MeterReader(model_path)
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to initialize model: {str(e)}"}), 500

        # Process each region
        raw_readings = []
        processed_readings = []
        confidence_scores = []
        valid_regions = []

        for i, (x1, y1, x2, y2) in enumerate(regions):
            try:
                # Validate region coordinates
                if (x1 >= x2 or y1 >= y2 or 
                    x1 < 0 or y1 < 0 or 
                    x2 > image.shape[1] or y2 > image.shape[0]):
                    logging.warning(f"Invalid region {i}: [{x1}, {y1}, {x2}, {y2}]")
                    continue

                region_image = image[y1:y2, x1:x2]
                if region_image.size == 0:
                    logging.warning(f"Empty region {i}: [{x1}, {y1}, {x2}, {y2}]")
                    continue

                # Get prediction using existing method
                raw_reading, confidence = meter_reader.predict(region_image)
                raw_readings.append(float(raw_reading))
                confidence_scores.append(float(confidence))

                # Process reading (same logic as web interface)
                processed_reading = round(raw_reading)
                if processed_reading == 10:
                    processed_reading = 0
                processed_readings.append(int(processed_reading))
                
                valid_regions.append([x1, y1, x2, y2])

            except Exception as e:
                logging.error(f"Error processing region {i}: {str(e)}")
                continue

        if not valid_regions:
            return jsonify({"success": False, "error": "No valid regions could be processed"}), 400

        # Calculate final reading
        try:
            final_reading = int(''.join(map(str, processed_readings)))
        except Exception as e:
            logging.error(f"Error concatenating readings: {str(e)}")
            final_reading = -1

        # Return success response
        return jsonify({
            "success": True,
            "regions_processed": len(valid_regions),
            "raw_readings": raw_readings,
            "processed_readings": processed_readings,
            "confidence_scores": confidence_scores,
            "final_reading": final_reading
        }), 200

    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500


# Main entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    

