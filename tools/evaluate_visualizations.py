import os
import sys
from pathlib import Path
import argparse
import time

# Add project root to path to ensure we can import da3d
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from da3d.evaluation.vision_agent import VisionAgent
except ImportError as e:
    print(f"Failed to import VisionAgent: {e}")
    print("Please ensure you have installed the 'openai' package: pip install openai")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated 3D visualizations using a Vision Agent.")
    parser.add_argument("--output-dir", type=str, default="./test_outputs", help="Directory containing generated images")
    args = parser.parse_args()

    try:
        agent = VisionAgent()
    except ValueError as e:
        print(f"Error initializing VisionAgent: {e}")
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Define images to check and their specific prompts
    evaluations = [
        {
            "filename": "comparison_metric.png",
            "prompt": "This image shows three views (Top, Side, Front) of a 3D point cloud created from a depth map. \
                       1. Does the geometry look coherent and represent a 3D scene? \
                       2. Are there any obvious flying pixels or noise artifacts? \
                       3. Does the 'Side View' show a reasonable depth profile? \
                       Provide a brief assessment."
        },
        {
            "filename": "comparison_relative.png",
            "prompt": "This image shows a 'Relative Mode' visualization. \
                       Compare the depth scaling to a typical 3D scene. \
                       Does it look flattened or distorted compared to what you expect?"
        },
        {
            "filename": "comparison_metric_inverted.png",
            "prompt": "This image shows a 3D mesh where the depth might be inverted. \
                       Look at the 'Top View' and 'Side View'. \
                       Does the geometry look 'inside out' (e.g. background closer than foreground)? \
                       Confirm if it looks distorted or inverted."
        }
    ]

    print("Starting Vision Agent Evaluation...\n")

    output_dir = Path(args.output_dir)
    
    for item in evaluations:
        image_path = output_dir / item["filename"]
        if not image_path.exists():
            print(f"Skipping {item['filename']}: File not found.")
            continue

        print(f"--- Evaluating {item['filename']} ---")
        
        try:
            result = agent.evaluate_image(str(image_path), item["prompt"])
            print(f"\nAgent Assessment:\n{result}\n")
        except Exception as e:
            print(f"\nError evaluating {item['filename']}: {e}\n")
        
        print("-" * 50 + "\n")
        # Wait a bit to avoid rate limits
        time.sleep(2)

if __name__ == "__main__":
    main()
