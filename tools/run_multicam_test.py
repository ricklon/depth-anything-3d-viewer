import subprocess
import os
from pathlib import Path
import time

# Configuration for the cameras
CAMERAS = [
    {"id": 0, "name": "Logi Cam C920e"},
    {"id": 1, "name": "HD Pro Webcam C920"}
]

def run_command(command, log_file=None):
    """Run a shell command and print output, optionally logging to a file."""
    print(f"Running: {command}")
    if log_file:
        log_file.write(f"\n{'='*60}\nRunning: {command}\n{'='*60}\n")
        log_file.flush()
    
    try:
        # Capture output to write to log file
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream output to console and log file
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='')
                if log_file:
                    log_file.write(line)
                    log_file.flush()
        
        return process.returncode == 0
    except Exception as e:
        msg = f"Error running command: {e}\n"
        print(msg)
        if log_file:
            log_file.write(msg)
        return False

def main():
    print("="*60)
    print(" MULTI-CAMERA TEST RUNNER")
    print("="*60)

    base_output_dir = Path("test_outputs_multicam")
    os.makedirs(base_output_dir, exist_ok=True)

    for cam in CAMERAS:
        cam_id = cam["id"]
        cam_name = cam["name"]
        cam_slug = cam_name.lower().replace(" ", "_")
        output_dir = base_output_dir / f"camera_{cam_id}_{cam_slug}"
        
        print(f"\n\n>>> TESTING CAMERA {cam_id}: {cam_name}")
        print(f"    Output Directory: {output_dir}")
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Open log file for this camera
        log_path = output_dir / "run.log"
        with open(log_path, "w") as log_file:
            log_file.write(f"Test Run for Camera {cam_id}: {cam_name}\n")
            log_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. Capture and Generate Depth
            print("\n[Step 1] Capturing and Generating Depth...")
            cmd_capture = f"uv run tests/test_webcam_single_frame.py --camera-id {cam_id} --output-dir {output_dir} --no-interactive"
            if not run_command(cmd_capture, log_file):
                print(f"Skipping remaining tests for Camera {cam_id} due to capture failure.")
                continue

            # 2. Generate Mode Comparison
            print("\n[Step 2] Generating Mode Comparison Plots...")
            cmd_compare = f"uv run tools/generate_mode_comparison.py --input-dir {output_dir} --output-dir {output_dir}"
            run_command(cmd_compare, log_file)

            # 3. Generate Tuned Visualization
            print("\n[Step 3] Generating Tuned Point Cloud...")
            cmd_tuned = f"uv run tools/generate_tuned_visualization.py --input-dir {output_dir} --output-dir {output_dir}"
            run_command(cmd_tuned, log_file)

            # 4. Run Vision Agent Evaluation
            print("\n[Step 4] Running Vision Agent Evaluation...")
            cmd_eval = f"uv run tools/evaluate_visualizations.py --output-dir {output_dir}"
            run_command(cmd_eval, log_file)

        print(f"\n>>> COMPLETED CAMERA {cam_id}")
        time.sleep(2) # Brief pause between cameras

    print("\n" + "="*60)
    print(" ALL TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
