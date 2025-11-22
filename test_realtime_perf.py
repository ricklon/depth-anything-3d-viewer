import time
import numpy as np
import open3d as o3d
from da3d.viewing.mesh import DepthMeshViewer

def test_performance():
    print("Initializing performance test...")
    
    # Create dummy data (640x480)
    width, height = 640, 480
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    depth = np.random.uniform(0.1, 10.0, (height, width)).astype(np.float32)
    
    # Initialize viewer with metric depth
    viewer = DepthMeshViewer(
        use_metric_depth=True,
        focal_length_x=470.4,
        focal_length_y=470.4,
        metric_depth_scale=1.0,
        display_mode='pointcloud'
    )
    
    print(f"Testing mesh generation for {width}x{height} input...")
    
    # Warmup
    viewer.create_mesh_from_depth(image, depth, subsample=2, invert_depth=False, smooth_mesh=False, use_sor=False)
    
    with open("perf_results.txt", "w") as f:
        # Test 1: SOR Enabled
        print("\nTesting with SOR ENABLED...")
        start_time = time.time()
        iterations = 5 # Reduced iterations for SOR
        for i in range(iterations):
            viewer.create_mesh_from_depth(image, depth, subsample=2, invert_depth=False, smooth_mesh=False, use_sor=True)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        fps = 1.0 / avg_time
        result_str = f"SOR ENABLED: Average time: {avg_time*1000:.2f}ms ({fps:.2f} FPS)\n"
        print(result_str)
        f.write(result_str)

        # Test 2: SOR Disabled
        print("\nTesting with SOR DISABLED...")
        start_time = time.time()
        iterations = 20
        for i in range(iterations):
            viewer.create_mesh_from_depth(image, depth, subsample=2, invert_depth=False, smooth_mesh=False, use_sor=False)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        fps = 1.0 / avg_time
        result_str = f"SOR DISABLED: Average time: {avg_time*1000:.2f}ms ({fps:.2f} FPS)\n"
        print(result_str)
        f.write(result_str)
        
        if fps < 30:
            f.write("\nNote: Real-time performance might be limited by mesh generation speed.\n")

if __name__ == "__main__":
    test_performance()
