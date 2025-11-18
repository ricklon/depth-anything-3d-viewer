#!/usr/bin/env python3
"""
Simple webcam calibration to find focal length in pixels.
Prints a checkerboard pattern - measure it and run calibration.
"""

import cv2
import numpy as np

def calibrate_webcam(camera_id=0, checkerboard_size=(9, 6)):
    """
    Calibrate webcam to find focal length in pixels.

    Args:
        camera_id: Camera device ID
        checkerboard_size: Inner corners (width, height) of checkerboard

    Instructions:
        1. Print a checkerboard pattern (9x6 inner corners recommended)
        2. Hold it in front of camera at different angles
        3. Press SPACE to capture calibration images
        4. Press ESC when done (need at least 10 images)
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None

    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {width}x{height}")
    print("\nInstructions:")
    print("1. Print a checkerboard pattern (Google 'opencv checkerboard')")
    print("2. Hold it in front of camera at different angles/distances")
    print("3. Press SPACE when checkerboard is detected (green corners)")
    print("4. Capture at least 10 good images")
    print("5. Press ESC when done\n")

    capture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        display = frame.copy()

        if ret_corners:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            # Draw corners
            cv2.drawChessboardCorners(display, checkerboard_size, corners2, ret_corners)

            cv2.putText(display, "Checkerboard detected! Press SPACE to capture",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            key = cv2.waitKey(1)
            if key == ord(' '):  # Space to capture
                objpoints.append(objp)
                imgpoints.append(corners2)
                capture_count += 1
                print(f"Captured {capture_count} calibration images")
        else:
            cv2.putText(display, "Looking for checkerboard...",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {capture_count} | ESC to finish",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Webcam Calibration', display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if capture_count < 10:
        print(f"\nNeed at least 10 images for calibration. Got {capture_count}.")
        return None

    print(f"\nCalibrating with {capture_count} images...")

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Focal length X: {fx:.1f} pixels")
        print(f"Focal length Y: {fy:.1f} pixels")
        print(f"Principal point X: {cx:.1f} pixels")
        print(f"Principal point Y: {cy:.1f} pixels")
        print("\nCamera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        print("="*60)

        print("\nUse these values with metric depth:")
        print(f"da3d webcam3d --metric --encoder vits \\")
        print(f"  --focal-length-x {fx:.1f} \\")
        print(f"  --focal-length-y {fy:.1f} \\")
        print(f"  --principal-point-x {cx:.1f} \\")
        print(f"  --principal-point-y {cy:.1f}")
        print("="*60)

        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    else:
        print("Calibration failed!")
        return None


if __name__ == "__main__":
    import sys

    camera_id = 0
    if len(sys.argv) > 1:
        camera_id = int(sys.argv[1])

    print("Starting webcam calibration...")
    print("Download checkerboard: https://github.com/opencv/opencv/blob/master/doc/pattern.png")
    print()

    result = calibrate_webcam(camera_id=camera_id)

    if result:
        # Save calibration
        np.savez('camera_calibration.npz',
                 camera_matrix=result['camera_matrix'],
                 dist_coeffs=result['dist_coeffs'])
        print("\nCalibration saved to: camera_calibration.npz")
