
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import da3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from da3d.cli.commands import MODEL_CONFIGS
from video_depth_anything.video_depth import VideoDepthAnything

def check_model_output():
    print("Checking model output range...")
    
    # Use 'vits' (small) for speed
    encoder = 'vits'
    config = MODEL_CONFIGS[encoder]
    
    # Check for checkpoint
    checkpoint_path = Path(f'checkpoints/video_depth_anything_{encoder}.pth')
    if not checkpoint_path.exists():
        # Try metric checkpoint
        checkpoint_path = Path(f'checkpoints/metric_video_depth_anything_{encoder}.pth')
        
    if not checkpoint_path.exists():
        print(f"Error: No checkpoint found at {checkpoint_path}")
        print("Please download checkpoints first.")
        return

    print(f"Loading model from {checkpoint_path}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load standard model
    model = VideoDepthAnything(**config)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(device).eval()
    
    # Create dummy image (random noise)
    # 518x518 is default input size
    dummy_image = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
    
    print("Running inference on dummy image...")
    with torch.no_grad():
        # infer_video_depth takes list of frames
        # It returns (depths, fps)
        depths, _ = model.infer_video_depth(
            [dummy_image], 
            target_fps=30, 
            input_size=518, 
            device=device
        )
        
    depth = depths[0]
    print(f"Output shape: {depth.shape}")
    print(f"Min: {depth.min():.4f}")
    print(f"Max: {depth.max():.4f}")
    print(f"Mean: {depth.mean():.4f}")
    
    # Check if it looks like disparity (0-1 or similar) or depth (meters)
    # Video-Depth-Anything usually outputs inverse depth (disparity)
    
    # If we have metric checkpoint, check that too
    metric_checkpoint = Path(f'checkpoints/metric_video_depth_anything_{encoder}.pth')
    if metric_checkpoint.exists() and metric_checkpoint != checkpoint_path:
        print(f"\nLoading METRIC model from {metric_checkpoint}...")
        model_metric = VideoDepthAnything(**config, metric=True)
        model_metric.load_state_dict(torch.load(str(metric_checkpoint), map_location='cpu'), strict=True)
        model_metric = model_metric.to(device).eval()
        
        with torch.no_grad():
            depths_m, _ = model_metric.infer_video_depth(
                [dummy_image], 
                target_fps=30, 
                input_size=518, 
                device=device
            )
        
        depth_m = depths_m[0]
        print(f"Metric Output shape: {depth_m.shape}")
        print(f"Min: {depth_m.min():.4f}")
        print(f"Max: {depth_m.max():.4f}")
        print(f"Mean: {depth_m.mean():.4f}")

if __name__ == "__main__":
    check_model_output()
