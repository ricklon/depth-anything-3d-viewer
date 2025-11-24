import os
import sys
from pathlib import Path
import argparse

# Add project root to path to ensure we can import da3d
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
    parser.add_argument("--test-dir", type=str, default="test_results/current", help="Directory containing generated test images")
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

    # Define images to check from the standard visual test suite
    evaluations = [
        {
            "filename": "view_front.png",
            "prompt": "This is the FRONT view of a 3D mesh generated from a depth map. \
                       Does the object appear to have correct proportions? \
                       Are there any obvious distortions?"
        },
        {
            "filename": "view_side.png",
            "prompt": "This is the SIDE view of the same 3D mesh. \
                       1. CRITICAL: Do you see distinct 'layers' or 'slices' (quantization artifacts)? \
                       2. Is the depth continuous or stepped? \
                       3. Does the object look flat or does it have volume? \
                       4. Are there 'flying pixels' (noise) floating in the air?"
        },
        {
            "filename": "view_top.png",
            "prompt": "This is the TOP view. \
                       Does the depth extrusion look consistent with the object's shape? \
                       Are there any artifacts?"
        }
    ]

    print(f"Starting Vision Agent Evaluation on {args.test_dir}...\n")

    output_dir = Path(args.test_dir)
    report_path = output_dir / "vision_agent_review.md"
    
    with open(report_path, "w") as f:
        f.write("# Vision Agent Review\n\n")
        
        for item in evaluations:
            image_path = output_dir / item["filename"]
            if not image_path.exists():
                print(f"Skipping {item['filename']}: File not found.")
                continue

            print(f"--- Evaluating {item['filename']} ---")
            print(f"Prompt: {item['prompt']}")
            
            try:
                result = agent.evaluate_image(str(image_path), item["prompt"])
                print(f"\nAgent Assessment:\n{result}\n")
                print("-" * 50 + "\n")
                
                f.write(f"## {item['filename']}\n")
                f.write(f"**Prompt:** {item['prompt']}\n\n")
                f.write(f"**Assessment:**\n{result}\n\n")
                f.write("---\n")
            except Exception as e:
                print(f"Failed to evaluate {item['filename']}: {e}")

    print(f"Review complete. Saved to {report_path}")

if __name__ == "__main__":
    main()
