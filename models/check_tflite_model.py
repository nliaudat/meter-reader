import argparse
import tensorflow as tf
from datetime import datetime

def inspect_tflite_model(model_path, verbose=False, output_file=None):
    """Inspect ops and details in a TFLite model."""
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Initialize output content
        output_content = []
        
        # --- Simple Op List (always printed to console) ---
        print(f"\nOperations in model '{model_path}':")
        for op in interpreter._get_ops_details():
            print(op['op_name'])
        
        # --- Detailed Report (for verbose/file output) ---
        header = f"\n=== TFLite Model Inspection Report ==="
        header += f"\nModel: {model_path}"
        header += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        header += f"\nTensorFlow Version: {tf.__version__}\n"
        output_content.append(header)

        # Use experimental Analyzer (TF 2.10+)
        if hasattr(tf.lite, 'experimental') and hasattr(tf.lite.experimental, 'Analyzer'):
            analyzer_output = tf.lite.experimental.Analyzer.analyze(model_path=model_path)
            output_content.append("\n[Analyzer Output]\n" + analyzer_output)
        else:
            output_content.append("\n[Warning] tf.lite.experimental.Analyzer not available (requires TF 2.10+).")

        # Verbose mode: Show detailed op info
        if verbose:
            output_content.append("\n[Detailed Operations]")
            for op in interpreter._get_ops_details():
                op_info = f"\nOp: {op['op_name']} (Index: {op['index']})"
                op_info += f"\n  Input Tensors: {op['inputs']}"
                op_info += f"\n  Output Tensors: {op['outputs']}"
                # Add quantization info if available
                for tensor_idx in op['inputs'] + op['outputs']:
                    tensor_details = interpreter.get_tensor_details()[tensor_idx]
                    if 'quantization' in tensor_details and tensor_details['quantization'] != (0, 0):
                        op_info += f"\n  Quantization (Tensor {tensor_idx}): Scale={tensor_details['quantization'][0]}, Zero-Point={tensor_details['quantization'][1]}"
                output_content.append(op_info)

        # Print detailed report to console (if verbose or output_file requested)
        if verbose or output_file:
            print('\n'.join(output_content))

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(output_content))
            print(f"\nReport saved to: {output_file}")

    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect operations and details in a TensorFlow Lite model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the TFLite model file (e.g., 'model.tflite')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed op info (inputs/outputs/quantization)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save report to a file (e.g., 'report.txt')"
    )
    args = parser.parse_args()
    inspect_tflite_model(args.model_path, args.verbose, args.output_file)