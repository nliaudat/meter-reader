#!/usr/bin/env python3
"""
Meter Reading Detection Script

This script uses TensorFlow Lite models to detect and read meter values from images.
It supports multiple model types and can process images from files or URLs.
"""

import cv2
import numpy as np
import tensorflow as tf
import logging
import argparse
import requests
import os
import sys
import json
import ast
from typing import List, Tuple, Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
MODELS_DIR = "models"
DEFAULT_REGIONS_FILE = "regions.json"
DEFAULT_MODEL = "class100-0180"
DEFAULT_RESULT_IMAGE = "result.jpg"

# Model Configuration 
MODELS: Dict[str, Dict[str, Any]] = {
    "class100-0180": {
        "path": os.path.join(MODELS_DIR, "dig-class100-0180-s2-q.tflite"),
        "description": "dig-class100-0180",
        "output_processing": "softmax_scale10",
        "scale_factor": 10,
        "input_type": "float32"
    },
    "class100-0173": {
        "path": os.path.join(MODELS_DIR, "dig-class100-0173-s2-q.tflite"),
        "description": "dig-class100-0173",
        "output_processing": "softmax_scale10",
        "scale_factor": 10,
        "input_type": "float32"
    },
    "class10-0900": {
        "path": os.path.join(MODELS_DIR, "dig-cont_0900_s3_q.tflite"),
        "description": "dig-cont_0900",
        "output_processing": "softmax",
        "scale_factor": 1,
        "input_type": "float32"
    },
    "class10-0810": {
        "path": os.path.join(MODELS_DIR, "dig-cont_0810_s3_q.tflite"),
        "description": "dig-cont_0810",
        "output_processing": "softmax",
        "scale_factor": 1,
        "input_type": "float32"
    },
    "mnist": {
        "path": os.path.join(MODELS_DIR, "mnist.tflite"),
        "description": "MNIST Digit Classifier",
        "output_processing": "direct_class",
        "scale_factor": 1,
        "input_channels": 1,
        "input_size": (28, 28),
        "normalize": True,
        "invert": True
    }
}

class MeterReader:
    """Class for reading meter values using TensorFlow Lite models."""
    
    def __init__(self, model_type: str = DEFAULT_MODEL) -> None:
        """
        Initialize the MeterReader with a specific model type.
        
        Args:
            model_type: One of the keys from the MODELS dictionary
        """
        try:
            self.model_config = MODELS[model_type]
            self.model_type = model_type
        except KeyError:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODELS.keys())}")
            
        model_path = self.model_config["path"]
        logger.info(f"Loading {model_type} model from: {os.path.abspath(model_path)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
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
        if self.model_config.get('invert', False):
            image = cv2.bitwise_not(image)
        if self.model_config.get('normalize', False):
            image = image.astype('float32') / 255.0

        return image.astype(input_dtype)

    def predict(self, image: np.ndarray) -> Tuple[float, float]:
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
                probabilities = tf.nn.softmax(output[0]).numpy()
                predicted_class = np.argmax(probabilities)
                confidence = np.max(probabilities)
                meter_reading = predicted_class
            elif output_processing == "softmax":
                # For models using softmax without scaling
                probabilities = tf.nn.softmax(output[0]).numpy()
                predicted_class = np.argmax(probabilities)
                confidence = np.max(probabilities)
                meter_reading = predicted_class
            else:  # Default case: "softmax_scale10"
                # For original classification models with 10x scaling
                probabilities = tf.nn.softmax(output[0]).numpy()
                predicted_class = np.argmax(probabilities)
                confidence = np.max(probabilities)
                meter_reading = predicted_class / self.model_config.get("scale_factor", 10)
                
            return meter_reading, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Error during prediction: {str(e)}")


def load_regions(regions_source: str) -> List[Tuple[int, int, int, int]]:
    """Load regions from file or string representation."""
    try:
        if os.path.exists(regions_source):
            with open(regions_source, "r") as f:
                regions = json.load(f)
        else:
            regions_source = regions_source.strip().strip('"').strip("'")
            regions = ast.literal_eval(regions_source)
        
        # Validate regions format and convert to tuples
        valid_regions = []
        for region in regions:
            if len(region) == 4:
                try:
                    valid_regions.append(tuple(map(int, region)))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid region format: {region}")
            else:
                logger.warning(f"Region must have 4 coordinates, got: {region}")
        
        return valid_regions
        
    except (FileNotFoundError, json.JSONDecodeError, SyntaxError, ValueError) as e:
        logger.error(f"Error loading regions: {e}")
        return []


def load_image(image_source: str) -> Optional[np.ndarray]:
    """Load image from local file or remote URL."""
    try:
        if image_source.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # Load from file
            if not os.path.exists(image_source):
                logger.error(f"Image file not found: {image_source}")
                return None
            image = cv2.imread(image_source)
        
        if image is None:
            logger.error(f"Unable to decode image from: {image_source}")
            return None
        
        return image
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading image: {e}")
        return None


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    if args.model not in MODELS:
        logger.error(f"Invalid model type: {args.model}. Available models: {list(MODELS.keys())}")
        return False
    
    if not args.image_source:
        logger.error("Image source must be specified")
        return False
    
    # Check regions source
    if args.regions == DEFAULT_REGIONS_FILE and not os.path.exists(DEFAULT_REGIONS_FILE):
        logger.error(f"No regions specified and default file {DEFAULT_REGIONS_FILE} not found")
        return False
    
    return True


def print_help() -> None:
    """Print help information for the script."""
    print("\nMeter Reading Detection Script")
    print("=" * 50)
    print("\nUsage: python meter_reading.py [options]\n")
    print("Options:")
    print("  --help                     Show this help message and exit")
    print(f"  --model MODEL_TYPE        Model type to use (default: {DEFAULT_MODEL})")
    print("                             Available models:")
    for model_name, config in MODELS.items():
        print(f"                               {model_name}: {config['description']}")
    print(f"  --regions REGIONS_SOURCE  Path to JSON file or string representation of regions (default: {DEFAULT_REGIONS_FILE})")
    print("  --image_source IMAGE_SOURCE Path to local image file or URL of remote image (required)")
    print("  --no-gui                  Disable GUI (no image display)")
    print("  --no-output-image         Do not save the output image with annotations")
    print("  --no-confidence           Do not display confidence scores on output image")
    print("\nExamples:")
    print(f"  python meter_reading.py --model {DEFAULT_MODEL} --image_source sample.jpg")
    print(f"  python meter_reading.py --model {DEFAULT_MODEL} --regions custom_regions.json --image_source sample.jpg")
    print("  python meter_reading.py --model class10-0900 --regions \"[[10,10,50,50],[60,60,100,100]]\" --image_source http://example.com/image.jpg")
    print("\nNote: If --regions is not specified, the script will look for regions.json in the current directory.")


def process_image(
    meter_reader: MeterReader,
    image: np.ndarray,
    regions: List[Tuple[int, int, int, int]],
    no_confidence: bool = False
) -> Dict[str, Any]:
    """
    Process an image with the given regions and return results.
    
    Returns:
        Dictionary containing:
        - raw_readings: List of raw meter readings
        - processed_readings: List of processed readings
        - confidence_scores: List of confidence scores
        - final_reading: Concatenated final reading
        - result_image: Annotated result image (if generated)
    """
    raw_readings = []
    processed_readings = []
    confidence_scores = []
    valid_regions = []
    
    for region in regions:
        x1, y1, x2, y2 = region
        try:
            # Validate region coordinates
            if (x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or 
                x2 > image.shape[1] or y2 > image.shape[0]):
                logger.warning(f"Invalid region coordinates: {region}")
                continue

            region_image = image[y1:y2, x1:x2]
            if region_image.size == 0:
                logger.warning(f"Empty region: {region}")
                continue

            raw_reading, confidence = meter_reader.predict(region_image)
            raw_readings.append(raw_reading)
            confidence_scores.append(confidence)

            # Process reading based on model type
            processed_reading = round(raw_reading)
            if meter_reader.model_type != "mnist" and processed_reading == 10:
                processed_reading = 0
            processed_readings.append(processed_reading)
            valid_regions.append(region)
            
        except Exception as e:
            logger.error(f"Error processing region {region}: {str(e)}")
            continue

    if not processed_readings:
        raise ValueError("No valid readings were processed")

    # Calculate final reading
    try:
        final_reading = int(''.join(map(str, processed_readings)))
    except Exception as e:
        logger.error(f"Error concatenating readings: {str(e)}")
        final_reading = -1

    # Generate result image
    result_image = image.copy()
    for i, (region, reading) in enumerate(zip(valid_regions, raw_readings)):
        x1, y1, x2, y2 = region
        cv2.putText(result_image, f"{reading:.1f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not no_confidence and i < len(confidence_scores):
            confidence = confidence_scores[i]
            cv2.putText(result_image, f"{int(round(confidence * 100))}%", 
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 165, 255), 2)

    return {
        "raw_readings": raw_readings,
        "processed_readings": processed_readings,
        "confidence_scores": confidence_scores,
        "final_reading": final_reading,
        "result_image": result_image
    }


def main() -> None:
    """Main function to handle command line execution."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Meter Reader", add_help=False)
    parser.add_argument("--help", action="store_true", help="Show help message and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, 
                       help=f"Model type to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--regions", default=DEFAULT_REGIONS_FILE, 
                       help=f"Path to JSON file or string representation of regions (default: {DEFAULT_REGIONS_FILE})")
    parser.add_argument("--image_source", required=True, 
                       help="Path to local image file or URL of remote image")
    parser.add_argument("--no-gui", action="store_true", 
                       help="Disable GUI (no image display)")
    parser.add_argument("--no-output-image", action="store_true", 
                       help="Do not save output image with annotations")
    parser.add_argument("--no-confidence", action="store_true", 
                       help="Do not display confidence scores")

    # Parse the arguments
    args, _ = parser.parse_known_args()

    # Show help and exit if --help is specified
    if args.help:
        print_help()
        sys.exit(0)

    # Validate arguments
    if not validate_arguments(args):
        print_help()
        sys.exit(1)

    try:
        # Initialize the MeterReader
        meter_reader = MeterReader(args.model)

        # Load the image
        image = load_image(args.image_source)
        if image is None:
            logger.error(f"Unable to load image from: {args.image_source}")
            sys.exit(1)

        # Load regions
        regions = load_regions(args.regions)
        if not regions:
            logger.error(f"No valid regions found in {args.regions}")
            sys.exit(1)

        # Process the image
        results = process_image(
            meter_reader,
            image,
            regions,
            args.no_confidence
        )

        # Print results
        logger.info(f"\n{'='*50}")
        logger.info(f"Model used: {meter_reader.model_type}")
        logger.info(f"Raw Readings: {results['raw_readings']}")
        logger.info(f"Processed Readings: {results['processed_readings']}")
        logger.info(f"Confidence Scores: {[int(round(score * 100)) for score in results['confidence_scores']]}")
        logger.info(f"Final Reading: {results['final_reading']}")
        logger.info(f"{'='*50}\n")

        # Save result image if not disabled
        if not args.no_output_image:
            cv2.imwrite(DEFAULT_RESULT_IMAGE, results['result_image'])
            logger.info(f"Result image saved to: {DEFAULT_RESULT_IMAGE}")

            # Display the result if GUI is enabled
            if not args.no_gui:
                cv2.imshow("Meter Readings", results['result_image'])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Create required directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    main()