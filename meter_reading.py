"""
Enhanced Meter Reading Detection Script

This script uses TensorFlow Lite models to detect and read meter values from images.
It supports multiple model types, automatic region detection, and provides detailed output.
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
import csv
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('meter_reading.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration Constants
MODELS_DIR = Path("models")
DEFAULT_REGIONS_FILE = Path("regions.json")
DEFAULT_MODEL = "class100-0180"
DEFAULT_RESULT_IMAGE = Path("result.jpg")
MAX_IMAGE_SIZE = (1920, 1080)  # For memory safety

@dataclass
class ModelConfig:
    path: Path
    description: str
    output_processing: str
    scale_factor: float
    input_type: str  # 'float32', 'uint8', 'int8' - this determines quantization
    input_channels: int = 3
    input_size: Optional[Tuple[int, int]] = None
    normalize: bool = False
    invert: bool = False

    @property
    def quantized(self) -> bool:
        """Check if model is quantized based on input type"""
        return self.input_type in ['uint8', 'int8']
    
    @property
    def esp_dl_quantization(self) -> bool:
        """Check if model uses ESP-DL quantization (int8)"""
        return self.input_type == 'int8'

# Model Configuration
MODELS: Dict[str, ModelConfig] = {
    "class100-0180": ModelConfig(
        path=MODELS_DIR / "dig-class100-0180-s2-q.tflite",
        description="dig-class100-0180",
        output_processing="softmax_scale10",
        scale_factor=10.0,
        input_type="float32"
    ),
    "class100-0173": ModelConfig(
        path=MODELS_DIR / "dig-class100-0173-s2-q.tflite",
        description="dig-class100-0173",
        output_processing="softmax_scale10",
        scale_factor=10.0,
        input_type="float32"
    ),
    "class10-0900": ModelConfig(
        path=MODELS_DIR / "dig-cont_0900_s3_q.tflite",
        description="dig-cont_0900",
        output_processing="softmax",
        scale_factor=1.0,
        input_type="float32"
    ),
    "class10-0810": ModelConfig(
        path=MODELS_DIR / "dig-cont_0810_s3_q.tflite",
        description="dig-cont_0810",
        output_processing="softmax",
        scale_factor=1.0,
        input_type="float32"
    ),
    "mnist": ModelConfig(
        path=MODELS_DIR / "mnist.tflite",
        description="MNIST Digit Classifier",
        output_processing="direct_class",
        scale_factor=1.0,
        input_type="float32",
        input_channels=1,
        input_size=(28, 28),
        normalize=True, 
        invert=True
    ),
    "esp_quantization_ready": ModelConfig(
        path=MODELS_DIR / "esp_quantization_ready.tflite",
        description="esp_quantization_ready Digit Classifier",
        output_processing="softmax",
        scale_factor=1.0,
        input_type="uint8",  
        input_channels=1,
        input_size=(32, 20),
        normalize=True,
        invert=False
    ),
    "digit_recognizer_v4": ModelConfig(
        path=MODELS_DIR / "digit_recognizer_v4_10_GR_q.tflite",
        description="digit_recognizer_v4 Digit Classifier",
        output_processing="softmax",
        scale_factor=1.0,
        input_type="uint8",
        input_channels=1,
        input_size=(32, 20),
        normalize=True,
        invert=False
    ),
}

class MeterReader:
    """Class for reading meter values using TensorFlow Lite models."""
    
    def __init__(self, model_type: str = DEFAULT_MODEL) -> None:
        """Initialize the MeterReader with a specific model type."""
        if model_type not in MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODELS.keys())}")
            
        self.model_config = MODELS[model_type]
        self.model_type = model_type
        
        logger.info(f"Loading {model_type} model from: {self.model_config.path.absolute()}")
        
        if not self.model_config.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_config.path}")
            
        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_config.path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Store quantization parameters
            self.input_quantization = self.input_details[0].get('quantization', (1.0, 0))
            self.output_quantization = self.output_details[0].get('quantization', (1.0, 0))
            
            # Log quantization info
            self._log_quantization_info()
            
            # Validate model configuration matches actual model
            self._validate_model_config()
            
            # Debug logging
            self._debug_model_info()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _log_quantization_info(self) -> None:
        """Log detailed quantization information."""
        if self.model_config.quantized:
            input_scale, input_zero_point = self.input_quantization
            output_scale, output_zero_point = self.output_quantization
            
            logger.info(f"Quantization detected:")
            logger.info(f"  Input type: {self.model_config.input_type}")
            logger.info(f"  Input scale: {input_scale}, zero_point: {input_zero_point}")
            logger.info(f"  Output scale: {output_scale}, zero_point: {output_zero_point}")
            
            # Detect actual quantization scheme based on parameters
            if input_zero_point == -128 and abs(input_scale - 1/255.0) < 1e-6:
                logger.info("  Actual scheme: uint8 in int8 container [0,255] -> [-128,127]")
            elif input_zero_point == 0 and self.model_config.esp_dl_quantization:
                logger.info("  Actual scheme: True ESP-DL [-128,127]")
            else:
                logger.info("  Actual scheme: Standard quantization")
            
            # Additional output quantization info
            logger.info(f"  Output quantization - scale: {output_scale}, zero_point: {output_zero_point}")
            if output_zero_point != 0:
                logger.info(f"  Output requires dequantization: (output - {output_zero_point}) * {output_scale}")

    def _validate_model_config(self) -> None:
        """Validate that the model configuration matches the actual model."""
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 4:
            _, height, width, channels = input_shape
        elif len(input_shape) == 3:
            _, height, width = input_shape
            channels = 1
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        logger.info(f"Model actual input: {width}x{height}, channels: {channels}")

        if self.model_config.input_size:
            config_height, config_width = self.model_config.input_size
            if (height, width) != (config_height, config_width):
                logger.warning(f"Model config input_size {self.model_config.input_size} (HxW) doesn't match actual model input size {(height, width)} (HxW)")

        if channels != self.model_config.input_channels:
            logger.warning(f"Model config input_channels {self.model_config.input_channels} doesn't match actual model channels {channels}")
            
    def debug_output(self, output: np.ndarray) -> None:
        """Debug method to analyze output tensor."""
        logger.debug("=== OUTPUT DEBUG ===")
        logger.debug(f"Raw output shape: {output.shape}")
        logger.debug(f"Raw output dtype: {output.dtype}")
        logger.debug(f"Raw output range: [{output.min()}, {output.max()}]")
        logger.debug(f"Raw output values: {output[0]}")
        
        # Check if output is quantized
        if self.model_config.quantized:
            output_scale, output_zero_point = self.output_quantization
            logger.debug(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")
            
            # Dequantize manually
            dequantized = (output.astype(np.float32) - output_zero_point) * output_scale
            logger.debug(f"Dequantized range: [{dequantized.min():.6f}, {dequantized.max():.6f}]")
            logger.debug(f"Dequantized values: {dequantized[0]}")
            
            # Apply softmax to dequantized
            probs_dequant = tf.nn.softmax(dequantized[0]).numpy()
            logger.debug(f"Probabilities from dequant: {[f'{p:.4f}' for p in probs_dequant]}")
            
            # Apply softmax directly to raw output
            probs_raw = tf.nn.softmax(output[0].astype(np.float32)).numpy()
            logger.debug(f"Probabilities from raw: {[f'{p:.4f}' for p in probs_raw]}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image according to model input requirements."""
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

        # Use model config dimensions if specified, otherwise use model's actual dimensions
        if self.model_config.input_size:
            target_height, target_width = self.model_config.input_size
        else:
            target_height, target_width = height, width

        logger.debug(f"Target size: {target_width}x{target_height}, Model expects: {width}x{height}")

        # Resize to expected size - OpenCV uses (width, height)
        image = cv2.resize(image, (target_width, target_height))

        # Handle channel conversion based on model requirements
        if channels == 1 or self.model_config.input_channels == 1:
            # Model expects grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Ensure single channel for 4D input
            if len(image.shape) == 2 and len(input_shape) == 4:
                image = np.expand_dims(image, axis=-1)  # Add channel dimension (H, W, 1)
        elif channels == 3 or self.model_config.input_channels == 3:
            # Model expects color (RGB)
            if len(image.shape) == 2:
                # Convert grayscale to RGB by repeating channels
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Convert single channel to 3 channels
                image = np.repeat(image, 3, axis=2)
            # Ensure BGR to RGB conversion
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply preprocessing operations
        if self.model_config.invert:
            image = cv2.bitwise_not(image)
        
        # Handle quantization and normalization - SIMPLIFIED LOGIC
        if self.model_config.quantized:
            # Always normalize to [0, 1] first for consistency
            image = image.astype('float32') / 255.0
            
            # Apply quantization based on model requirements
            input_scale, input_zero_point = self.input_quantization
            
            if input_dtype == np.uint8:
                # Standard uint8 quantization [0, 255]
                image = (image * 255.0).astype(np.uint8)
            elif input_dtype == np.int8:
                # int8 quantization - handle both ESP-DL and uint8-in-int8 cases
                if input_zero_point == -128 and abs(input_scale - 1/255.0) < 1e-6:
                    # uint8 in int8 container
                    image = (image * 255.0).astype(np.uint8)
                    image = image.astype(np.float32)
                    image = (image / input_scale + input_zero_point).astype(np.int8)
                else:
                    # True ESP-DL quantization
                    image = (image * 255.0 - 128.0).astype(np.int8)
        else:
            # For non-quantized models
            if self.model_config.normalize:
                image = image.astype('float32') / 255.0
            else:
                image = image.astype('float32')
        
        # Add batch dimension
        if len(input_shape) == 4:
            image = np.expand_dims(image, axis=0)  # Add batch dimension (1, H, W, C)
        elif len(input_shape) == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dimension (1, H, W)

        logger.debug(f"Final preprocessed image shape: {image.shape}, dtype: {image.dtype}")
        logger.debug(f"Image value range: [{np.min(image)}, {np.max(image)}]")
        return image

    def _validate_input_image(self, image: np.ndarray) -> None:
        """Validate input image dimensions and type."""
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided")
        
        # Check if image dimensions match expected input size
        if self.model_config.input_size:
            expected_height, expected_width = self.model_config.input_size
            if image.shape[:2] != (expected_height, expected_width):
                logger.warning(f"Image size {image.shape[:2]} (HxW) doesn't match expected size {(expected_height, expected_width)} (HxW)")
                
    def _debug_model_info(self) -> None:
        """Log detailed model information for debugging."""
        logger.debug("=== Model Input Details ===")
        logger.debug(f"Input shape: {self.input_details[0]['shape']}")
        logger.debug(f"Input dtype: {self.input_details[0]['dtype']}")
        logger.debug(f"Input name: {self.input_details[0]['name']}")
        if 'quantization' in self.input_details[0]:
            logger.debug(f"Input quantization: {self.input_details[0]['quantization']}")
        
        logger.debug("=== Model Output Details ===")
        logger.debug(f"Output shape: {self.output_details[0]['shape']}")
        logger.debug(f"Output dtype: {self.output_details[0]['dtype']}")
        logger.debug(f"Output name: {self.output_details[0]['name']}")
        if 'quantization' in self.output_details[0]:
            logger.debug(f"Output quantization: {self.output_details[0]['quantization']}")
        
        logger.debug("=== Model Configuration ===")
        logger.debug(f"Model type: {self.model_type}")
        logger.debug(f"Input size: {self.model_config.input_size}")
        logger.debug(f"Input channels: {self.model_config.input_channels}")
        logger.debug(f"Input type: {self.model_config.input_type}")
        logger.debug(f"Quantized: {self.model_config.quantized}")
        logger.debug(f"ESP-DL Quantization: {self.model_config.esp_dl_quantization}")

    def _dequantize_output(self, output: np.ndarray) -> np.ndarray:
        """Dequantize output if the model is quantized."""
        if self.model_config.quantized and self.output_details[0]['dtype'] in [np.uint8, np.int8]:
            output_scale, output_zero_point = self.output_quantization
            # Convert to float32 first, then dequantize
            output_float = output.astype(np.float32)
            return (output_float - output_zero_point) * output_scale
        return output.astype(np.float32)

    def predict(self, image: np.ndarray) -> Tuple[float, float]:
        """Predict meter reading from an image - FIXED OUTPUT PROCESSING ONLY"""
        try:
            # Use the ORIGINAL working preprocessing
            input_image = self.preprocess_image(image)
            
            # Verify shape matches expected input shape
            expected_shape = self.input_details[0]['shape']
            if input_image.shape != tuple(expected_shape):
                logger.warning(f"Input shape {input_image.shape} doesn't match expected {tuple(expected_shape)}")
                if input_image.size == np.prod(expected_shape):
                    input_image = input_image.reshape(expected_shape)
                    logger.info(f"Reshaped input to: {input_image.shape}")
            
            # Set input tensor and run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            self.interpreter.invoke()
            
            # Get model output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # DEBUG: Check what we're getting
            logger.debug(f"Raw output: {output[0]}")
            logger.debug(f"Raw output dtype: {output.dtype}")
            
            # FIXED: Different output processing for quantized vs non-quantized models
            if self.model_config.quantized:
                # For quantized models, use gap-based confidence (to avoid softmax issues)
                output_scale, output_zero_point = self.output_quantization
                output_dequantized = output.astype(np.float32) * output_scale  # zero_point is 0
                
                output_values = output_dequantized[0]
                predicted_class = np.argmax(output_values)
                max_val = output_values[predicted_class]
                
                # Calculate confidence based on gap between max and second max
                        # the problem with softmax and quantized models
                        # Raw output: [255, 0, 0, 0, 0, 0, 0, 0, 0, 0] (very confident)
                        # After dequantization: [0.996, 0.000, 0.000, ...] (still very confident)
                        # After softmax: [0.231, 0.085, 0.085, ...] (flattened to ~23%)
                if len(output_values) > 1:
                    other_values = np.delete(output_values, predicted_class)
                    second_max = np.max(other_values)
                    confidence = (max_val - second_max) / max_val if max_val > 0 else 0.0
                else:
                    confidence = 1.0
                
                confidence = max(0.0, min(1.0, confidence))
                logger.debug(f"Quantized confidence - max: {max_val:.4f}, second: {second_max:.4f}, confidence: {confidence:.4f}")
                
            else:
                # For non-quantized models, use original softmax approach
                output_dequantized = self._dequantize_output(output)
                probabilities = tf.nn.softmax(output_dequantized[0]).numpy()
                predicted_class = np.argmax(probabilities)
                confidence = float(np.max(probabilities))
                logger.debug(f"Softmax confidence: {confidence:.4f}")
            
            # Apply scaling based on model type
            if self.model_config.output_processing == "direct_class":
                meter_reading = float(predicted_class)
            elif self.model_config.output_processing == "softmax":
                meter_reading = float(predicted_class)
            else:  # "softmax_scale10"
                meter_reading = float(predicted_class) / self.model_config.scale_factor
            
            return meter_reading, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Error during prediction: {str(e)}")

def load_regions(regions_source: Union[str, Path]) -> List[Tuple[int, int, int, int]]:
    """Load regions from file or string representation."""
    try:
        regions_path = Path(regions_source)
        
        if regions_path.exists():
            with regions_path.open('r') as f:
                regions = json.load(f)
        else:
            # Try to parse as string representation
            regions_str = str(regions_source).strip().strip('"').strip("'")
            regions = ast.literal_eval(regions_str)
        
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
        
        if not valid_regions:
            logger.error("No valid regions found in input")
        
        return valid_regions
        
    except (FileNotFoundError, json.JSONDecodeError, SyntaxError, ValueError) as e:
        logger.error(f"Error loading regions: {e}", exc_info=True)
        return []

def load_image(image_source: Union[str, Path], input_channels: int = 1) -> Optional[np.ndarray]:
    """Load image from local file or remote URL based on model's input requirements."""
    try:
        image_source_str = str(image_source)
        
        # Determine loading mode based on input_channels
        if input_channels == 1:
            # Load as grayscale
            load_mode = cv2.IMREAD_GRAYSCALE
        else:
            # Load as color (BGR)
            load_mode = cv2.IMREAD_COLOR
        
        if image_source_str.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image_source_str, timeout=10)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                logger.error(f"URL doesn't point to an image (Content-Type: {content_type})")
                return None
                
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, load_mode)
        else:
            # Load from file
            image_path = Path(image_source_str)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
                
            image = cv2.imread(str(image_path), load_mode)
        
        if image is None:
            logger.error(f"Unable to decode image from: {image_source_str}")
            return None
            
        # Convert BGR to RGB for color images if needed
        if input_channels == 3 and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Check image size
        if image.nbytes > 50 * 1024 * 1024:  # 50MB
            logger.warning("Large image detected, consider resizing")
            
        logger.info(f"Loaded image: {image_source_str}, shape: {image.shape}, channels: {1 if len(image.shape) == 2 else image.shape[2]}")
        return image
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading image: {e}", exc_info=True)
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
    regions_path = Path(args.regions)
    if args.regions == str(DEFAULT_REGIONS_FILE) and not regions_path.exists():
        logger.error(f"No regions specified and default file {DEFAULT_REGIONS_FILE} not found")
        return False
    
    return True

def print_help() -> None:
    """Print help information for the script."""
    help_text = f"""
Meter Reading Detection Script
{'=' * 50}

Usage: python meter_reading.py [options]

Options:
  --help                     Show this help message and exit
  --model MODEL_TYPE        Model type to use (default: {DEFAULT_MODEL})
                             Available models:"""
    
    for model_name, config in MODELS.items():
        help_text += f"\n                               {model_name}: {config.description}"
    
    help_text += f"""
  --regions REGIONS_SOURCE  Path to JSON file or string representation of regions (default: {DEFAULT_REGIONS_FILE})
  --image_source IMAGE_SOURCE Path to local image file or URL of remote image (required)
  --no-gui                  Disable GUI (no image display)
  --no-output-image         Do not save the output image with annotations
  --no-confidence           Do not display confidence scores on output image
  --test-all-models         Test all models with the same image and regions
  --expected_result EXPECTED_RESULT The real number read by human (for comparison with model results)

Examples:
  python meter_reading.py --model {DEFAULT_MODEL} --image_source sample.jpg
  python meter_reading.py --model {DEFAULT_MODEL} --regions custom_regions.json --image_source sample.jpg
  python meter_reading.py --model class10-0900 --regions "[[10,10,50,50],[60,60,100,100]]" --image_source http://example.com/image.jpg
  python meter_reading.py --test-all-models --image_source test.jpg --expected_result 12345

Note: If --regions is not specified, the script will look for {DEFAULT_REGIONS_FILE} in the current directory.
"""
    print(help_text)

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
    results = {
        "raw_readings": [],
        "processed_readings": [],
        "confidence_scores": [],
        "final_reading": -1,
        "result_image": None
    }
    
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
            results["raw_readings"].append(raw_reading)
            results["confidence_scores"].append(confidence)

            # Process reading based on model type
            processed_reading = round(raw_reading)
            if meter_reader.model_type != "mnist" and processed_reading == 10:
                processed_reading = 0
            results["processed_readings"].append(processed_reading)
            valid_regions.append(region)
            
        except Exception as e:
            logger.error(f"Error processing region {region}: {str(e)}", exc_info=True)
            continue

    if not results["processed_readings"]:
        raise ValueError("No valid readings were processed")

    # Calculate final reading
    try:
        results["final_reading"] = int(''.join(map(str, results["processed_readings"])))
    except Exception as e:
        logger.error(f"Error concatenating readings: {str(e)}")
        results["final_reading"] = -1

    # Generate result image (convert back to BGR for OpenCV display)
    results["result_image"] = image.copy()
    if len(results["result_image"].shape) == 3 and results["result_image"].shape[2] == 3:
        results["result_image"] = cv2.cvtColor(results["result_image"], cv2.COLOR_RGB2BGR)
        
    for i, (region, reading) in enumerate(zip(valid_regions, results["raw_readings"])):
        x1, y1, x2, y2 = region
        cv2.putText(results["result_image"], f"{reading:.1f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(results["result_image"], (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not no_confidence and i < len(results["confidence_scores"]):
            confidence = results["confidence_scores"][i]
            cv2.putText(results["result_image"], f"{int(round(confidence * 100))}%", 
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 165, 255), 2)

    return results
    
def test_all_models(
    image_source: Union[str, Path],
    regions_source: Union[str, Path] = DEFAULT_REGIONS_FILE,
    output_file: Union[str, Path] = "model_comparison.csv",
    expected_result: Optional[int] = None
) -> None:
    """
    Test all available models with the same image and regions, comparing their confidence scores.
    
    Args:
        image_source: Path or URL to the test image
        regions_source: Path to regions file or string representation
        output_file: Path to save the comparison results (CSV format)
        expected_result: The real number read by human (for comparison)
    """
    # Load image and regions once
    image = load_image(image_source, input_channels=1)  # Default to grayscale
    if image is None:
        logger.error(f"Failed to load image from {image_source}")
        return

    regions = load_regions(regions_source)
    if not regions:
        logger.error(f"No valid regions found in {regions_source}")
        return

    results = []
    headers = ["Model", "Description", "Final Reading", "Correct"]
    headers += [f"Digit {i+1} Confidence" for i in range(len(regions))]
    headers.append("Average Confidence")

    for model_name, model_config in MODELS.items():
        try:
            logger.info(f"\nTesting model: {model_name} - {model_config.description}")
            meter_reader = MeterReader(model_name)
            
            # Process the image
            result = process_image(meter_reader, image, regions, no_confidence=True)
            
            # Calculate average confidence
            avg_confidence = np.mean(result["confidence_scores"]) if result["confidence_scores"] else 0
            
            # Check if prediction matches expected result
            correct = "Yes" if expected_result is not None and result["final_reading"] == expected_result else "No"
            
            # Prepare row for CSV
            row = {
                "Model": model_name,
                "Description": model_config.description,
                "Final Reading": result["final_reading"],
                "Correct": correct,
                "Average Confidence": avg_confidence
            }
            
            # Add individual digit confidences
            for i, conf in enumerate(result["confidence_scores"]):
                row[f"Digit {i+1} Confidence"] = conf
            
            results.append(row)
            
            logger.info(f"Results for {model_name}:")
            logger.info(f"Final Reading: {result['final_reading']}")
            if expected_result is not None:
                logger.info(f"Expected Result: {expected_result}")
                logger.info(f"Match: {'Yes' if correct == 'Yes' else 'No'}")
            logger.info(f"Confidence Scores: {[f'{c:.2%}' for c in result['confidence_scores']]}")
            logger.info(f"Average Confidence: {avg_confidence:.2%}")
            
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {str(e)}", exc_info=True)
            continue

    # Save results to CSV
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
                
        logger.info(f"\nComparison results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save comparison results: {str(e)}", exc_info=True)  
    

def main() -> None:
    """Main function to handle command line execution."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Meter Reader", add_help=False)
    parser.add_argument("--help", action="store_true", help="Show help message and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, 
                       help=f"Model type to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--regions", default=str(DEFAULT_REGIONS_FILE), 
                       help=f"Path to JSON file or string representation of regions (default: {DEFAULT_REGIONS_FILE})")
    parser.add_argument("--image_source", 
                       help="Path to local image file or URL of remote image")
    parser.add_argument("--no-gui", action="store_true", 
                       help="Disable GUI (no image display)")
    parser.add_argument("--no-output-image", action="store_true", 
                       help="Do not save output image with annotations")
    parser.add_argument("--no-confidence", action="store_true", 
                       help="Do not display confidence scores")
    parser.add_argument("--test-all-models", action="store_true",
                       help="Test all models with the same image and regions")
    parser.add_argument("--expected_result", type=int,
                       help="The real number read by human (for comparison with model results)")

    try:
        # Parse the arguments
        args, _ = parser.parse_known_args()

        # Show help and exit if --help is specified
        if args.help:
            print_help()
            sys.exit(0)
            
        if args.test_all_models:
            if not args.image_source:
                print_help()
                logger.error("Error: --image_source is required for testing all models")
                sys.exit(1)
                
            test_all_models(args.image_source, args.regions, expected_result=args.expected_result)
            sys.exit(0)

        # Now check for required arguments
        if not args.image_source:
            print_help()
            logger.error("Error: --image_source is required")
            sys.exit(1)

        # Rest of the validation
        if not validate_arguments(args):
            print_help()
            sys.exit(1)

        # Initialize the MeterReader
        meter_reader = MeterReader(args.model)

        # Load the image with appropriate channels based on model
        input_channels = meter_reader.model_config.input_channels
        image = load_image(args.image_source, input_channels)
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
        if args.expected_result is not None:
            logger.info(f"Expected Result: {args.expected_result}")
            logger.info(f"Match: {'Yes' if results['final_reading'] == args.expected_result else 'No'}")
        logger.info(f"{'='*50}\n")

        # Save result image if not disabled
        if not args.no_output_image and results['result_image'] is not None:
            cv2.imwrite(str(DEFAULT_RESULT_IMAGE), results['result_image'])
            logger.info(f"Result image saved to: {DEFAULT_RESULT_IMAGE}")

            # Display the result if GUI is enabled
            if not args.no_gui:
                cv2.imshow("Meter Readings", results['result_image'])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Create required directories if they don't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)