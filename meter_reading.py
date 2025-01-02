import cv2
import numpy as np
import tensorflow as tf
import logging
import argparse
import requests
import os
import sys

# Set up logging for better output control
logging.basicConfig(level=logging.INFO)

class MeterReader:
    def __init__(self, model_path):
        """
        Initialize the MeterReader with a TensorFlow Lite model.
        
        Args:
            model_path (str): Path to the TensorFlow Lite model file.
        """
        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
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

    def visualize(self, image, regions, meter_readings, raw=True):
        """
        Visualize the meter readings on the image.
        
        Args:
            image (numpy.ndarray): Input image (BGR format).
            regions (list): List of tuples defining the regions (x1, y1, x2, y2).
            meter_readings (list): Predicted meter readings for each region.
            raw (bool): If True, display raw readings; otherwise, display processed readings.
        
        Returns:
            numpy.ndarray: Image with the meter readings displayed.
        """
        # Add the meter readings as text on each region
        for region, reading in zip(regions, meter_readings):
            x1, y1, x2, y2 = region
            if raw:
                # Display raw readings (floats)
                cv2.putText(image, f"{reading:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Display processed readings (integers)
                cv2.putText(image, f"{int(round(reading))}", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw a rectangle around the region
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image


def load_regions(file_path):
    """
    Load regions from a text file.
    
    Args:
        file_path (str): Path to the text file containing regions.
        
    Returns:
        list: List of tuples defining the regions (x1, y1, x2, y2).
    """
    regions = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                # Parse the coordinates from the file
                x1, y1, x2, y2 = map(int, line.strip().split(','))
                regions.append((x1, y1, x2, y2))
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
    except ValueError:
        logging.error(f"Invalid format in {file_path}. Expected 4 integers per line.")
    return regions


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


def validate_args(args):
    """
    Validate the command-line arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        bool: True if all arguments are valid, False otherwise.
    """
    if not os.path.exists(args.model):
        logging.error(f"Model file {args.model} not found.")
        return False

    if not os.path.exists(args.regions):
        logging.error(f"Regions file {args.regions} not found.")
        return False

    return True


def print_help():
    """
    Print help information for the script.
    """
    print("Usage: python tflite_meter_reading.py --model MODEL_PATH --regions REGIONS_FILE --image_source IMAGE_SOURCE")
    print("\nArguments:")
    print("  --model        Path to the TensorFlow Lite model file.")
    print("  --regions      Path to the regions file (text file with coordinates).")
    print("  --image_source Path to the local image file or URL of the remote image.")
    print("\nExample:")
    print("  python tflite_meter_reading.py --model model.tflite --regions regions.txt --image_source http://192.168.1.113/img_tmp/alg.jpg")
    print("  python tflite_meter_reading.py --model model.tflite --regions regions.txt --image_source /path/to/local/image.jpg")


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Meter Reader", add_help=False)
    parser.add_argument("--help", action="store_true", help="Show this help message and exit")
    parser.add_argument("--model", default="model.tflite", help="Path to the TensorFlow Lite model")
    parser.add_argument("--regions", default="regions.txt", help="Path to the regions file")
    parser.add_argument("--image_source", default="sample.jpg", help="Path to the local image file or URL of the remote image")

    # Parse the arguments
    args, _ = parser.parse_known_args()

    # Show help and exit if --help is specified
    if args.help:
        print_help()
        sys.exit(0)

    # Check if required arguments are missing
    if not args.model or not args.regions or not args.image_source:
        logging.error("Missing required arguments.")
        print_help()
        sys.exit(1)

    # Validate the arguments
    if not validate_args(args):
        print_help()
        sys.exit(1)

    # Initialize the MeterReader
    meter_reader = MeterReader(args.model)

    # Load the image (local or remote)
    image = load_image(args.image_source)
    if image is None:
        logging.error(f"Unable to load image from {args.image_source}.")
        sys.exit(1)

    # Load regions from the text file
    regions = load_regions(args.regions)

    # Extract the regions from the image
    image_regions = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in regions]

    # Predict the meter reading for each region
    raw_meter_readings = []  # Store raw readings
    processed_meter_readings = []  # Store processed readings
    for region in image_regions:
        raw_reading = meter_reader.predict(region)
        raw_meter_readings.append(raw_reading)

        # Preprocess the reading
        processed_reading = round(raw_reading)  # Round to the nearest integer
        if processed_reading == 10:  # Handle the special case
            processed_reading = 0
        processed_meter_readings.append(processed_reading)

    # Concatenate the processed meter readings into a single integer
    concatenated_readings = int(''.join(map(str, processed_meter_readings)))

    # Print the raw and final results
    logging.info(f"Raw Meter Readings: {raw_meter_readings}")
    logging.info(f"Processed Meter Readings: {processed_meter_readings}")
    logging.info(f"Final Meter Reading: {concatenated_readings}")

    # Visualize the results
    result_image = meter_reader.visualize(image, regions, raw_meter_readings, raw=True)

    # Display the result
    cv2.imwrite("result.jpg", result_image)
    cv2.imshow("Meter Readings", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()