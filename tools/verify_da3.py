#!/usr/bin/env python3
"""
Verification script for Depth-Anything-3 integration.
Runs the estimator on a sample image and generates an agent report.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from da3d.estimators.da3_estimator import DA3Estimator
from da3d.verification.visual_validator import VisualValidator
from da3d.verification.data_validator import DataValidator
from da3d.verification.agent_report import AgentReport

def create_dummy_image(width=518, height=518):
    """Create a dummy image for testing if no input is provided."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), -1)
    cv2.circle(img, (300, 300), 50, (0, 0, 255), -1)
    # Gradient background to simulate depth
    for i in range(width):
        img[:, i, 0] = i % 255
    return img

def main():
    parser = argparse.ArgumentParser(description="Verify DA3 Integration")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="test_verify", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing DA3 Estimator...")
    try:
        estimator = DA3Estimator()
        # We use a small model or default if possible, but DA3 might need download
        # Assuming 'da3-large' is the default we set in the class
        estimator.load_model({'encoder': 'da3-large'})
        print("[OK] Model loaded.")
    except Exception as e:
        print(f"[FAIL] Could not load DA3 Estimator: {e}")
        print("Make sure depth-anything-3 is installed and weights are available.")
        return

    # Load or create image
    if args.image and Path(args.image).exists():
        print(f"Loading image: {args.image}")
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Using dummy image.")
        img = create_dummy_image()

    print("Running Inference...")
    try:
        depth, confidence = estimator.infer_depth(img)
        print(f"[OK] Inference successful. Depth shape: {depth.shape}")
        if confidence is not None:
            print(f"[OK] Confidence map available. Shape: {confidence.shape}")
        else:
            print("[INFO] No confidence map returned.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Inference failed: {e}")
        return

    print("Running Verification Modules...")
    
    # 1. Visual Validator
    visual_validator = VisualValidator(output_dir)
    composite_path = visual_validator.generate_composite(img, depth, confidence)
    print(f"[OK] Visual composite saved to: {composite_path}")

    # 2. Data Validator
    data_validator = DataValidator()
    metrics = data_validator.compute_metrics(depth, confidence)
    print(f"[OK] Metrics computed: {metrics}")

    # 3. Agent Report
    agent_report = AgentReport(output_dir)
    report_path = agent_report.generate_report(metrics, composite_path)
    print(f"[OK] Agent report saved to: {report_path}")

    print("\nVerification Complete!")
    print(f"Please review the report at: {report_path}")

if __name__ == "__main__":
    main()
