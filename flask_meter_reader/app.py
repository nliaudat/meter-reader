# Import required libraries
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import logging
import os
import json
import ast
import requests
from werkzeug.utils import secure_filename

# Model Configuration 
MODELS = {
    "class100-0180": {
        "path": os.path.abspath(os.path.join("models", "dig-class100-0180-s2-q.tflite")),
        "description": "dig-class100-0180",
        "output_processing": "softmax_scale10",
        "scale_factor": 10,
        "input_type": "float32",
        "source": {
            "name": "haverland/Tenth-of-step-of-a-meter-digit",
            "url": "https://github.com/haverland/Tenth-of-step-of-a-meter-digit/releases",
            "license": "MIT",
            "notes": "Optimized for meter digit classification"
        }
    },
    "class100-0173": {
        "path": os.path.abspath(os.path.join("models", "dig-class100-0173-s2-q.tflite")),
        "description": "dig-class100-0173",
        "output_processing": "softmax_scale10",
        "scale_factor": 10,
        "input_type": "float32",
        "source": {
            "name": "haverland/Tenth-of-step-of-a-meter-digit",
            "url": "https://github.com/haverland/Tenth-of-step-of-a-meter-digit/releases",
            "license": "MIT",
            "notes": "Optimized for meter digit classification"
        }
    },
    "class10-0900": {
        "path": os.path.abspath(os.path.join("models", "dig-cont_0900_s3_q.tflite")),
        "description": "dig-cont_0900",
        "output_processing": "softmax",
        "scale_factor": 1,
        "input_type": "float32",
        "source": {
            "name": "jomjol/AI-on-the-edge-device",
            "url": "https://github.com/jomjol/AI-on-the-edge-device",
            "license": "Apache 2.0",
            "notes": "From AI-on-the-edge-device project config"
        }
    },
    "class10-0810": {
        "path": os.path.abspath(os.path.join("models", "dig-cont_0810_s3_q.tflite")),
        "description": "dig-cont_0810",
        "output_processing": "softmax",
        "scale_factor": 1,
        "input_type": "float32",
        "source": {
            "name": "jomjol/AI-on-the-edge-device",
            "url": "https://github.com/jomjol/AI-on-the-edge-device",
            "license": "Apache 2.0",
            "notes": "From AI-on-the-edge-device project config"
        }
    },
    "mnist": {
        "path": os.path.abspath(os.path.join("models", "mnist.tflite")),
        "description": "MNIST Digit Classifier",
        "output_processing": "direct_class",
        "scale_factor": 1,
        "input_channels": 1,
        "input_size": (28, 28),
        "normalize": True,
        "invert": True,
        "source": {
            "name": "alankrantas/MNIST-Live-Detection-TFLite",
            "url": "https://github.com/alankrantas/MNIST-Live-Detection-TFLite",
            "license": "MIT",
            "notes": "Standard MNIST classifier converted to TFLite"
        }
    }
}
DEFAULT_MODEL = "class100-0180"


# Set up logging configuration
logging.basicConfig(level=logging.INFO)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

app = Flask(__name__)
app.secret_key = "supersecretkey"

class MeterReader:
    def __init__(self, model_type=None):
        """Initialize with model type (classification/continuous/mnist)"""
        model_type = model_type or DEFAULT_MODEL
        try:
            self.model_config = MODELS[model_type]
        except KeyError:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODELS.keys())}")
            
        model_path = self.model_config["path"]
        print(f"Loading {model_type} model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

    def preprocess_image(self, image):
        """Preprocess image according to model input requirements."""
        if not hasattr(self, 'input_details') or len(self.input_details) == 0:
            raise ValueError("Model not properly initialized - missing input details")

        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided")

        input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']

        # Handle different input shapes
        if len(input_shape) == 4:
            _, height, width, channels = input_shape
        elif len(input_shape) == 3:
            _, height, width = input_shape
            channels = 1
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        # Resize to expected size
        image = cv2.resize(image, (width, height))

        # Handle grayscale or color
        if channels == 1:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(input_shape) == 4:
                image = np.expand_dims(image, axis=(0, -1))  # (1, H, W, 1)
            else:
                image = np.expand_dims(image, axis=0)        # (1, H, W)
        elif channels == 3:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = np.expand_dims(image, axis=0)            # (1, H, W, 3)
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")

        # Apply normalization if configured
        if getattr(self.model_config, 'invert', False):
            image = cv2.bitwise_not(image)
        if getattr(self.model_config, 'normalize', False):
            image = image.astype('float32') / 255.0

        # Special handling for MNIST (invert colors)
        if "mnist" in self.model_config.get("path", "").lower():
            image = cv2.bitwise_not(image)

        return image.astype(input_dtype)

    def predict(self, image):
        """Predict meter reading from an image with proper output processing."""
        try:
            input_image = self.preprocess_image(image)
            
            # Set input tensor and run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            self.interpreter.invoke()
            
            # Get model output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process output based on model configuration
            output_processing = self.model_config.get("output_processing", "softmax_scale10")
            
            if output_processing == "direct_class":
                # For MNIST-style direct class prediction
                predicted_class = np.argmax(output[0])
                confidence = np.max(softmax(output[0]))  # Still use softmax for confidence
                meter_reading = predicted_class
            elif output_processing == "softmax":
                # For models using softmax without scaling
                probabilities = softmax(output[0])
                predicted_class = np.argmax(probabilities)
                confidence = np.max(probabilities)
                meter_reading = predicted_class
            else:  # Default case: "softmax_scale10"
                # For original classification models with 10x scaling
                probabilities = softmax(output[0])
                predicted_class = np.argmax(probabilities)
                confidence = np.max(probabilities)
                meter_reading = predicted_class / self.model_config.get("scale_factor", 10)
                
            return meter_reading, confidence
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Error during prediction: {str(e)}")
        
        
def load_regions(regions_source):
    """Load regions from file or string representation."""
    try:
        if os.path.exists(regions_source):
            with open(regions_source, "r") as f:
                regions = json.load(f)
        else:
            regions_source = regions_source.strip().strip('"').strip("'")
            regions = ast.literal_eval(regions_source)
        
        return [tuple(region) for region in regions if len(region) == 4]
    except (FileNotFoundError, json.JSONDecodeError, SyntaxError, ValueError) as e:
        logging.error(f"Error loading regions: {e}")
        return []

def save_regions(file_path, regions):
    """Save regions to a JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(regions, f)
        flash("Regions saved successfully.", "success")
    except Exception as e:
        flash(f"Error saving regions: {e}", "error")

def load_image(image_source):
    """Load image from local file or remote URL."""
    if image_source.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_source)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image: {e}")
            return None
    else:
        if not os.path.exists(image_source):
            logging.error(f"Image file not found: {image_source}")
            return None
        return cv2.imread(image_source)

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for web interface."""
    
    # Always create template context with models
    template_context = {
        "models": MODELS,
        "default_model": DEFAULT_MODEL,
        "model_type": DEFAULT_MODEL,
        "final_reading": None,  # Initialize as None
        "result_image": None
    }

    if request.method == "POST":
        # Handle image upload/URL
        image_source = None
        filename = None
        
        if "image_file" in request.files:
            file = request.files["image_file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                file_path = os.path.join("static", filename)
                file.save(file_path)
                image_source = file_path

        if not image_source:
            image_source = request.form.get("image_url")
            if not image_source:
                flash("Please provide an image source", "error")
                return redirect(url_for("index"))
            try:
                response = requests.get(image_source)
                response.raise_for_status()
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    flash("Unable to load image from URL", "error")
                    return redirect(url_for("index"))
                filename = secure_filename(os.path.basename(image_source))
                file_path = os.path.join("static", filename)
                cv2.imwrite(file_path, image)
                image_source = file_path
            except requests.exceptions.RequestException as e:
                flash(f"Error downloading image: {e}", "error")
                return redirect(url_for("index"))

        image = load_image(image_source)
        if image is None:
            return redirect(url_for("index"))

        regions_source = request.form.get("regions_source", "regions.json")
        regions = load_regions(regions_source)
        if not regions:
            flash("No valid regions provided", "error")
            return redirect(url_for("index"))

        model_type = request.form.get("model_type", DEFAULT_MODEL)
        try:
            meter_reader = MeterReader(model_type)
        except Exception as e:
            flash(f"Error initializing model: {str(e)}", "error")
            return redirect(url_for("index"))

        raw_readings = []
        processed_readings = []
        confidence_scores = []
        
        for region in regions:
            x1, y1, x2, y2 = region
            try:
                region_image = image[y1:y2, x1:x2]
                if region_image.size == 0:
                    continue

                raw_reading, confidence = meter_reader.predict(region_image)
                raw_readings.append(raw_reading)
                confidence_scores.append(confidence)

                # Process reading based on model type
                if model_type == "mnist":
                    processed_reading = int(raw_reading)  # MNIST outputs direct class
                else:
                    processed_reading = round(raw_reading)
                    if processed_reading == 10:  # Handle wrap-around for original models
                        processed_reading = 0
                processed_readings.append(processed_reading)
            except Exception as e:
                logging.error(f"Error processing region {region}: {str(e)}")
                continue

        # Handle concatenation safely
        try:
            if not processed_readings:
                concatenated_readings = -1  # Error value
                flash("No valid readings were processed", "error")
            else:
                concatenated_readings = int(''.join(map(str, processed_readings)))
        except ValueError as e:
            logging.error(f"Error concatenating readings: {str(e)}")
            concatenated_readings = -1  # Error value
            flash("Error processing meter readings", "error")

        # Only generate result image if we have readings
        if processed_readings:
            result_image = image.copy()
            for i, (region, reading) in enumerate(zip(regions, raw_readings)):
                x1, y1, x2, y2 = region
                cv2.putText(result_image, f"{reading:.1f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if not request.form.get("no_confidence"):
                    confidence = confidence_scores[i]
                    cv2.putText(result_image, f"{int(round(confidence * 100))}%", 
                               (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                               (0, 165, 255), 2)

            result_filename = f"result_{os.path.basename(image_source)}"
            result_path = os.path.join("static", result_filename)
            cv2.imwrite(result_path, result_image)
            template_context["result_image"] = result_filename

        # Update template context
        template_context["model_type"] = request.form.get("model_type", DEFAULT_MODEL)
        
        # Add results to template context
        template_context.update({
            "result_image": result_filename,
            "raw_readings": raw_readings,
            "processed_readings": processed_readings,
            "confidence_scores": [int(round(score * 100)) for score in confidence_scores],
            "final_reading": concatenated_readings
        })
        
    return render_template("index.html", **template_context)

    # return render_template("index.html", 
                         # models=MODELS,
                         # default_model=DEFAULT_MODEL)

@app.route("/draw_regions", methods=["GET", "POST"])
def draw_regions():
    """Route for drawing regions on an image."""
    if request.method == "POST":
        image_source = None
        filename = None
        
        if "image_file" in request.files:
            file = request.files["image_file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                file_path = os.path.join("static", filename)
                file.save(file_path)
                image_source = file_path

        if not image_source:
            image_source = request.form.get("image_url")
            if not image_source:
                flash("Please provide an image source", "error")
                return redirect(url_for("draw_regions"))
            try:
                response = requests.get(image_source)
                response.raise_for_status()
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    flash("Unable to load image from URL", "error")
                    return redirect(url_for("draw_regions"))
                filename = secure_filename(os.path.basename(image_source))
                file_path = os.path.join("static", filename)
                cv2.imwrite(file_path, image)
                image_source = file_path
            except requests.exceptions.RequestException as e:
                flash(f"Error downloading image: {e}", "error")
                return redirect(url_for("draw_regions"))

        image = load_image(image_source)
        if image is None:
            return redirect(url_for("draw_regions"))

        return render_template("draw_regions.html", image_source=filename)

    return render_template("draw_regions.html")

@app.route("/save_regions", methods=["POST"])
def save_regions_route():
    """API endpoint for saving drawn regions."""
    regions = request.json.get("regions")
    if not regions:
        flash("No regions provided", "error")
        return redirect(url_for("draw_regions"))

    save_regions("regions.json", regions)
    return jsonify({"message": "Regions saved successfully"})

@app.route('/api/json_response', methods=["GET", "POST"])
def json_response():
    """API endpoint for programmatic meter reading."""
    try:
        # Parse request data
        if request.method == "GET":
            data = {
                "image_source": request.args.get("image_url"),
                "regions_source": request.args.get("regions_source"),
                "model_type": request.args.get("model_type", DEFAULT_MODEL)  # Use DEFAULT_MODEL if not specified
            }
        else:
            data = request.get_json()
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No JSON data provided",
                    "available_models": list(MODELS.keys()),
                    "default_model": DEFAULT_MODEL
                }), 400

        # Set default model if not specified or invalid
        model_type = data.get("model_type", DEFAULT_MODEL)
        if model_type not in MODELS:
            model_type = DEFAULT_MODEL  # Fall back to default if invalid model specified

        # Validate required parameters
        if not data.get("image_source") and not data.get("image_url"):
            return jsonify({
                "success": False,
                "error": "Missing required parameter: image_source or image_url",
                "available_models": list(MODELS.keys()),
                "default_model": DEFAULT_MODEL
            }), 400
        
        if not data.get("regions_source"):
            return jsonify({
                "success": False,
                "error": "Missing required parameter: regions_source",
                "available_models": list(MODELS.keys()),
                "default_model": DEFAULT_MODEL
            }), 400

        # Get and validate model type
        model_type = data.get("model_type", DEFAULT_MODEL)
        if model_type not in MODELS:
            return jsonify({
                "success": False,
                "error": f"Unknown model type: {model_type}",
                "available_models": list(MODELS.keys())
            }), 400

        # Load and validate image
        image_source = data.get("image_source") or data.get("image_url")
        image = load_image(image_source)
        if image is None:
            return jsonify({
                "success": False,
                "error": f"Unable to load image from: {image_source}",
                "model_used": model_type
            }), 400

        # Load and validate regions
        regions_source = data.get("regions_source")
        regions = load_regions(regions_source)
        if not regions:
            return jsonify({
                "success": False,
                "error": "No valid regions provided",
                "model_used": model_type
            }), 400

        # Initialize meter reader
        try:
            meter_reader = MeterReader(model_type)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Model initialization failed: {str(e)}",
                "model_used": model_type
            }), 500

        # Process regions
        raw_readings = []
        processed_readings = []
        confidence_scores = []
        valid_regions = []

        for i, (x1, y1, x2, y2) in enumerate(regions):
            try:
                # Validate region coordinates
                if (x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or 
                    x2 > image.shape[1] or y2 > image.shape[0]):
                    continue

                region_image = image[y1:y2, x1:x2]
                if region_image.size == 0:
                    continue

                raw_reading, confidence = meter_reader.predict(region_image)
                raw_readings.append(float(raw_reading))
                confidence_scores.append(float(confidence))

                processed_reading = round(raw_reading)
                if model_type != "mnist" and processed_reading == 10:
                    processed_reading = 0
                processed_readings.append(int(processed_reading))
                valid_regions.append([x1, y1, x2, y2])

            except Exception as e:
                logging.error(f"Error processing region {i}: {str(e)}")
                continue

        if not valid_regions:
            return jsonify({
                "success": False,
                "error": "No valid regions could be processed",
                "model_used": model_type
            }), 400

        # Calculate final reading
        try:
            final_reading = int(''.join(map(str, processed_readings)))
        except Exception as e:
            logging.error(f"Error concatenating readings: {str(e)}")
            final_reading = -1

        # Return successful response
        return jsonify({
            "success": True,
            "model_used": model_type,
            "model_description": MODELS[model_type]["description"],
            "regions_processed": len(valid_regions),
            "raw_readings": raw_readings,
            "processed_readings": processed_readings,
            "confidence_scores": [int(round(score * 100)) for score in confidence_scores],
            "final_reading": final_reading,
            "model_source_info": MODELS[model_type].get("source", {})
        }), 200

    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e),
            "available_models": list(MODELS.keys()),
            "default_model": DEFAULT_MODEL
        }), 500
        
@app.route("/api/models")
def list_models():
    """API endpoint to list available models."""
    return jsonify({
        "models": list(MODELS.keys()),
        "default_model": DEFAULT_MODEL,
        "model_details": MODELS
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)