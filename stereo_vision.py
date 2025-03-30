"""
Stereo Vision Depth Mapping and Obstacle Detection System

This module processes stereo image pairs to create depth maps and detect obstacles
for navigation assistance.
"""
import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
from stereovision.calibration import StereoCalibration

# Stereo Block Matching parameters
DEFAULT_PARAMS = {
    'sad_window_size': 23,      # SAD Windows Size (odd 5-255)
    'pre_filter_size': 255,     # Pre Filter Size (odd 5-255)
    'pre_filter_cap': 55,       # Pre Filter Cap (1-63)
    'min_disparity': 0,         # Min Disparity
    'num_disparities': 64,      # Number of disparities
    'texture_threshold': 0,     # Texture Threshold
    'uniqueness_ratio': 8,      # Uniqueness Ratio
    'speckle_range': 40,        # Speckle Range
    'speckle_window_size': 300  # Speckle Size
}

# Detection constants
MID_POINT = 315
IMG_HEIGHT = 480
IMG_WIDTH = 640
PHOTO_WIDTH = 1280

# Detection results
RESULT_PASS = 0
RESULT_LEFT_TURN = 1
RESULT_RIGHT_TURN = 2
RESULT_BLOCKED = 3


def init_stereo_bm(params=None):
    """Initialize the StereoBM object with given parameters."""
    if params is None:
        params = DEFAULT_PARAMS
    
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=params['sad_window_size'])
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(params['pre_filter_size'])
    sbm.setPreFilterCap(params['pre_filter_cap'])
    sbm.setMinDisparity(params['min_disparity'])
    sbm.setNumDisparities(params['num_disparities'])
    sbm.setTextureThreshold(params['texture_threshold'])
    sbm.setUniquenessRatio(params['uniqueness_ratio'])
    sbm.setSpeckleRange(params['speckle_range'])
    sbm.setSpeckleWindowSize(params['speckle_window_size'])
    
    return sbm


def compute_stereo_depth_map(rectified_pair, sbm):
    """Compute disparity map from stereo image pair."""
    dm_left = rectified_pair[0]
    dm_right = rectified_pair[1]
    
    disparity = sbm.compute(dm_left, dm_right)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    image = np.array(disparity_normalized, dtype=np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
    return disparity_color, disparity_normalized, disparity


def detect_obstacles_in_region(y_min, y_max, disparity, distance_limit, queue, x_bounds=(90, 630)):
    """Detect obstacles in a specific vertical region of the disparity map."""
    x_min, x_max = x_bounds
    block_list_x = []
    block_list_y = []
    block_list_distance = []
    
    for y in range(y_min, y_max):
        if max(disparity[y]) >= distance_limit:
            if y < 300:
                # Process the left and right regions separately for the upper part
                for x in range(x_min, 230):
                    distance = disparity[y][x]
                    if distance >= distance_limit:
                        block_list_x.append(x)
                        block_list_y.append(y)
                        block_list_distance.append(distance)
                        
                for x in range(470, x_max):
                    distance = disparity[y][x]
                    if distance >= distance_limit:
                        block_list_x.append(x)
                        block_list_y.append(y)
                        block_list_distance.append(distance)
            else:
                # Process the entire region for the lower part
                for x in range(x_min, x_max):
                    distance = disparity[y][x]
                    if distance >= distance_limit:
                        block_list_x.append(x)
                        block_list_y.append(y)
                        block_list_distance.append(distance)
    
    queue.put([block_list_x, block_list_y, block_list_distance])


def save_block_map(image_path, block_map, block_list_x, block_list_y):
    """Draw the detected blocks on an image and save it."""
    for i in range(len(block_list_x)):
        block_map = cv2.circle(
            block_map, 
            (block_list_x[i], block_list_y[i]), 
            radius=0, 
            color=(0, 0, 255), 
            thickness=-1
        )
    
    output_path = f"{image_path[:-4]}_BlockMap.jpg"
    cv2.imwrite(output_path, block_map)
    return output_path


def create_block_map(disparity_map, distance_cm, downward=False):
    """Create a block map from disparity map using parallel processing."""
    start_time = time.time()
    
    # Calculate distance limit based on the distance in cm
    if downward:
        distance_limit = int(-8.145 * (distance_cm-10) + 1455)
        y_min = 100
    else:
        distance_limit = int(-8.145 * distance_cm + 1455)
        y_min = 150
    
    print(f"Distance limit: {distance_limit}")
    
    # Split the processing into 4 threads for parallel execution
    y_max = 470
    y_range = y_max - y_min
    y_slice = y_range // 4
    
    queues = [Queue() for _ in range(4)]
    threads = []
    
    for i in range(4):
        start_y = y_min + (i * y_slice)
        end_y = y_min + ((i + 1) * y_slice) if i < 3 else y_max
        thread = Thread(
            target=detect_obstacles_in_region, 
            args=(start_y, end_y, disparity_map, distance_limit, queues[i])
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Gather results from all threads
    all_data = [queue.get() for queue in queues]
    
    # Combine results
    block_list_x = []
    block_list_y = []
    block_list_distance = []
    
    for data in all_data:
        block_list_x.extend(data[0])
        block_list_y.extend(data[1])
        block_list_distance.extend(data[2])
    
    print(f"Found {len(block_list_y)} blocks")
    print(f"Block map creation time: {time.time() - start_time:.4f} seconds")
    
    return block_list_x, block_list_y, block_list_distance


def analyze_block_list(block_list_x, block_list_y, block_list_distance):
    """Analyze detected blocks to determine navigation direction."""
    if not block_list_y:
        return RESULT_PASS, 0  # No obstacles detected
    
    # Calculate statistics for detected blocks
    x_min = min(block_list_x) if block_list_x else 0
    x_max = max(block_list_x) if block_list_x else 0
    y_min = min(block_list_y) if block_list_y else 0
    y_max = max(block_list_y) if block_list_y else 0
    distance_max = max(block_list_distance) if block_list_distance else 0
    
    print(f"X range: {x_min} to {x_max}")
    print(f"Y range: {y_min} to {y_max}")
    print(f"Max distance in cm: {int((distance_max-1455)/-8.125)}")
    
    # Determine navigation direction based on obstacle position
    if x_min > MID_POINT + 150 and distance_max > 1220:
        return RESULT_LEFT_TURN, 10  # Far right obstacle
    elif x_max < MID_POINT - 150 and distance_max > 1220:
        return RESULT_RIGHT_TURN, -10  # Far left obstacle
    elif x_min > MID_POINT and distance_max > 1220:
        return RESULT_LEFT_TURN, 20  # Right obstacle
    elif x_max < MID_POINT and distance_max > 1220:
        return RESULT_RIGHT_TURN, -20  # Left obstacle
    elif x_min > MID_POINT and x_min < MID_POINT+200 and distance_max > 950:
        return RESULT_LEFT_TURN, 10  # Near right obstacle
    elif x_max < MID_POINT and x_max > MID_POINT-200 and distance_max > 950:
        return RESULT_RIGHT_TURN, -10  # Near left obstacle
    elif x_min > MID_POINT:
        return RESULT_LEFT_TURN, 5  # Slight right obstacle
    elif x_max < MID_POINT:
        return RESULT_RIGHT_TURN, -5  # Slight left obstacle
    else:
        return RESULT_BLOCKED, 0  # Obstacle in center
        

def process_stereo_image(image_path, distance_cm, downward=False):
    """
    Process a stereo image to detect obstacles and determine navigation.
    
    Args:
        image_path: Path to the stereo image file
        distance_cm: Detection distance in centimeters
        downward: Whether to look downward (changes detection parameters)
        
    Returns:
        tuple: (result, angle) where result indicates navigation action
    """
    print(f"Processing with detection distance: {distance_cm} cm")
    
    # Load and split the stereo image
    pair_img = cv2.imread(image_path)
    if pair_img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    left_frame = pair_img[0:IMG_HEIGHT, 0:IMG_WIDTH]
    right_frame = pair_img[0:IMG_HEIGHT, IMG_WIDTH:PHOTO_WIDTH]
    
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Process through stereo vision pipeline
    start_time = time.time()
    calibration = StereoCalibration(input_folder='calib_result')
    rectified_pair = calibration.rectify((left_gray, right_gray))
    
    sbm = init_stereo_bm()
    disparity_color, disparity_normalized, disparity_map = compute_stereo_depth_map(rectified_pair, sbm)
    print(f"Calibration + rectification + stereo matching: {time.time() - start_time:.4f} seconds")
    
    # Detect obstacles
    block_list_x, block_list_y, block_list_distance = create_block_map(disparity_map, distance_cm, downward)
    
    # Analyze detected obstacles
    start_time = time.time()
    result, angle = analyze_block_list(block_list_x, block_list_y, block_list_distance)
    print(f"Analysis time: {time.time() - start_time:.4f} seconds")
    
    # Output navigation decision
    if result == RESULT_LEFT_TURN:
        print(f"Left turn: {angle} degrees")
    elif result == RESULT_RIGHT_TURN:
        print(f"Right turn: {angle} degrees")
    elif result == RESULT_PASS:
        print("No obstacles - continue")
    else:
        print("Blocked - stop")
    
    return result, angle


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stereo vision depth mapping for navigation")
    parser.add_argument("image", help="Path to stereo image file")
    parser.add_argument("--distance", type=int, default=30, help="Detection distance in cm")
    parser.add_argument("--downward", action="store_true", help="Look downward for obstacles")
    
    args = parser.parse_args()
    
    process_stereo_image(args.image, args.distance, args.downward)
