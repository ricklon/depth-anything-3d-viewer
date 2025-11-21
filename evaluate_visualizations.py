import os
import sys
from pathlib import Path
import argparse
import time

# Add project root to path to ensure we can import da3d
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from da3d.evaluation.vision_agent import VisionAgent
except ImportError as e:
    print(f"Failed to import VisionAgent: {e}")
    print("Please ensure you have installed the 'openai' package: pip install openai")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated 3D visualizations using a Vision Agent.")
    parser.add_argument("--output-dir", type=str, default="./", help="Directory containing generated images")
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
            "prompt": "This image shows three views (Top, Side, Front) of a 3D mesh created from a depth map. \
                       1. Does the geometry look coherent? \
                       2. Are there any obvious flying pixels or noise artifacts? \
                       3. Does the 'Side View' show a reasonable depth profile for a typical scene? \
                       Provide a brief assessment."
        },
        {
            "filename": "comparison_metric_inverted.png",
            "prompt": "This image shows a 3D mesh where the depth might be inverted. \
                       Does the geometry look 'inside out' or incorrect compared to a normal 3D scene? \
                       Look at the 'Top View' and 'Side View'. \
                       Confirm if it looks distorted or inverted."
        },
        {
            "filename": "tuned_visualization.png",
            "prompt": "This is a point cloud visualization of a scene. \
                       Does the structure look like a recognizable 3D scene? \
                       Are the colors consistent with a natural image? \
                       Rate the visual quality from 1 to 10."
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
        print(f"Prompt: {item['prompt']}")
        
        result = agent.evaluate_image(str(image_path), item["prompt"])
        
        print(f"\nAgent Assessment:\n{result}\n")
        print("-" * 50 + "\n")
        
        # Add a small delay to avoid hitting rate limits (especially on free tiers)
        time.sleep(5)

if __name__ == "__main__":
    main()
