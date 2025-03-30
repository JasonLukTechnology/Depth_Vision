#!/usr/bin/env python3
"""
Example script for using stereo vision navigation system.

This script demonstrates how to use the stereo vision package
to detect obstacles and make navigation decisions.
"""
import os
import sys
import cv2
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from stereo_vision.stereo_depth_map import process_stereo_image, compute_stereo_depth_map, init_stereo_bm
from stereovision.calibration import StereoCalibration


def visualize_detection(image_path, distance_cm=30, downward=False):
    """Process a stereo image and visualize the results."""
    # Load the stereo image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Get image dimensions
    height, width = img.shape[:2]
    img_width = width // 2
    
    # Split into left and right images
    left_img = img[0:height, 0:img_width]
    right_img = img[0:height, img_width:width]
    
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Perform calibration and rectification
    print("Performing calibration and rectification...")
    calibration = StereoCalibration(input_folder='calib_result')
    rectified_pair = calibration.rectify((left_gray, right_gray))
    
    # Compute disparity map
    print("Computing disparity map...")
    sbm = init_stereo_bm()
    disparity_color, disparity_normalized, disparity = compute_stereo_depth_map(rectified_pair, sbm)
    
    # Process for navigation
    print(f"Processing for navigation with distance: {distance_cm}cm...")
    result, angle = process_stereo_image(image_path, distance_cm, downward)
    
    # Display results
    print(f"\nNavigation Result: {result}, Angle: {angle}\n")
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save visualization images
    base_name = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(f"output/{base_name}_disparity.jpg", disparity_color)
    cv2.imwrite(f"output/{base_name}_left.jpg", left_img)
    cv2.imwrite(f"output/{base_name}_right.jpg", right_img)
    
    print(f"Visualization images saved to output/{base_name}_*.jpg")
    
    # Show images if not running headless
    if os.environ.get('DISPLAY', ''):
        cv2.imshow("Left Image", left_img)
        cv2.imshow("Disparity Map", disparity_color)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize stereo obstacle detection")
    parser.add_argument("image", help="Path to stereo image file")
    parser.add_argument("--distance", type=int, default=30, help="Detection distance in cm")
    parser.add_argument("--downward", action="store_true", help="Look downward for obstacles")
    
    args = parser.parse_args()
    
    visualize_detection(args.image, args.distance, args.downward)
