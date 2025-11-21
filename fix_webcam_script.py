import re

# Read the file
with open('test_webcam_single_frame.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add no_interactive parameter to function signature (line 267)
content = content.replace(
    'def test_3d_variations(depth_raw, frame_rgb, focal_length_x):',
    'def test_3d_variations(depth_raw, frame_rgb, focal_length_x, no_interactive=False):'
)

# Fix 2: Wrap the viewer.view_mesh call in an if statement
old_display = '''            # Display
            print(f"    Opening 3D viewer...")
            viewer.view_mesh(
                mesh,
                window_name=f"Test {i+1}: {var['name']}",
                width=1024,
                height=768,
                show_wireframe=False
            )'''

new_display = '''            # Display
            if not no_interactive:
                print(f"    Opening 3D viewer...")
                viewer.view_mesh(
                    mesh,
                    window_name=f"Test {i+1}: {var['name']}",
                    width=1024,
                    height=768,
                    show_wireframe=False
                )
            else:
                print(f"    Skipping interactive viewer (headless mode)")'''

content = content.replace(old_display, new_display)

# Fix 3: Add argparse and update main function
old_config = '''    # Configuration
    CAMERA_ID = 0
    MAX_RES = 640
    ENCODER = 'vits'
    CHECKPOINTS_DIR = './checkpoints'
    OUTPUT_DIR = './test_outputs\''''

new_config = '''    import argparse
    parser = argparse.ArgumentParser(description="Single-frame webcam test")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--output-dir", type=str, default="./test_outputs", help="Output directory")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive 3D viewer")
    args = parser.parse_args()

    # Configuration
    CAMERA_ID = args.camera_id
    MAX_RES = 640
    ENCODER = 'vits'
    CHECKPOINTS_DIR = './checkpoints'
    OUTPUT_DIR = args.output_dir'''

content = content.replace(old_config, new_config)

# Fix 4: Update the function call
content = content.replace(
    'test_3d_variations(depth_raw, frame_rgb, focal_length_x)',
    'test_3d_variations(depth_raw, frame_rgb, focal_length_x, no_interactive=args.no_interactive)'
)

# Write the file back
with open('test_webcam_single_frame.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File updated successfully!")
