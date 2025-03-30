# Stereo Vision Navigation System

This project provides a stereo vision-based depth mapping and obstacle detection system for autonomous navigation. It processes stereo image pairs to create depth maps, detect obstacles, and determine navigation decisions.

## Features

- Depth map generation from stereo camera images
- Obstacle detection with configurable distance thresholds
- Navigation decision-making (left turn, right turn, stop, or continue)
- Multi-threaded processing for improved performance
- Visual output of depth maps and detected obstacles

## Installation

### Prerequisites

- Python 3.6+
- OpenCV
- NumPy
- [StereoVision](https://github.com/erget/StereoVision) library

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stereo_vision.git
   cd stereo_vision
   ```

2. Install the package and dependencies:
   ```
   pip install -e .
   ```

3. Ensure you have calibration data in the `calib_result` folder. If not, you'll need to calibrate your stereo camera first.

## Usage

### Basic Usage

Process a stereo image with default settings:

```bash
python -m stereo_vision.stereo_depth_map path/to/stereo_image.jpg
```

### Advanced Options

Specify detection distance and orientation:

```bash
python -m stereo_vision.stereo_depth_map path/to/stereo_image.jpg --distance 40 --downward
```

### Using as a Library

```python
from stereo_vision.stereo_depth_map import process_stereo_image

# Process an image with 30cm detection distance
result, angle = process_stereo_image("path/to/image.jpg", 30)

# Check the navigation decision
if result == 1:
    print(f"Turn left {angle} degrees")
elif result == 2:
    print(f"Turn right {angle} degrees")
elif result == 0:
    print("Continue straight")
else:
    print("Stop - obstacle ahead")
```

## Calibration

For accurate depth mapping, you need to calibrate your stereo camera setup:

1. Capture calibration images using a checkerboard pattern
2. Use the StereoVision library to calculate calibration parameters
3. Save the calibration files to the `calib_result` folder

## Project Structure

- `stereo_vision/` - Main package directory
  - `stereo_depth_map.py` - Core functionality
- `examples/` - Example usage and demonstrations
- `tests/` - Unit tests
- `calib_result/` - Calibration data directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenCV for computer vision algorithms
- StereoVision library for camera calibration utilities
