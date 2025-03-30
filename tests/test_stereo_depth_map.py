#!/usr/bin/env python3
"""
Unit tests for stereo depth mapping module.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from stereo_vision.stereo_depth_map import (
    init_stereo_bm,
    analyze_block_list,
    RESULT_PASS,
    RESULT_LEFT_TURN,
    RESULT_RIGHT_TURN,
    RESULT_BLOCKED
)


class TestStereoDepthmapFunctions(unittest.TestCase):
    """Test case for stereo depth map module functions."""
    
    def test_init_stereo_bm(self):
        """Test initialization of StereoBM object."""
        # Test with default parameters
        sbm = init_stereo_bm()
        self.assertEqual(sbm.getBlockSize(), 23)
        self.assertEqual(sbm.getPreFilterSize(), 255)
        self.assertEqual(sbm.getPreFilterCap(), 55)
        
        # Test with custom parameters
        custom_params = {
            'sad_window_size': 15,
            'pre_filter_size': 101,
            'pre_filter_cap': 31,
            'min_disparity': 0,
            'num_disparities': 32,
            'texture_threshold': 10,
            'uniqueness_ratio': 10,
            'speckle_range': 20,
            'speckle_window_size': 100
        }
        sbm = init_stereo_bm(custom_params)
        self.assertEqual(sbm.getBlockSize(), 15)
        self.assertEqual(sbm.getPreFilterSize(), 101)
        self.assertEqual(sbm.getPreFilterCap(), 31)
    
    def test_analyze_block_list_empty(self):
        """Test block list analysis with empty input."""
        result, angle = analyze_block_list([], [], [])
        self.assertEqual(result, RESULT_PASS)
        self.assertEqual(angle, 0)
    
    def test_analyze_block_list_right_obstacle(self):
        """Test block list analysis with obstacle on the right."""
        # Create a simulated obstacle on the right side
        # MID_POINT is 315
        block_list_x = [400, 410, 420]  # All points to the right of mid point
        block_list_y = [200, 210, 220]
        block_list_distance = [1300, 1350, 1400]  # Greater than 1220 threshold
        
        result, angle = analyze_block_list(block_list_x, block_list_y, block_list_distance)
        self.assertEqual(result, RESULT_LEFT_TURN)
        self.assertGreater(angle, 0)  # Positive angle for left turn
    
    def test_analyze_block_list_left_obstacle(self):
        """Test block list analysis with obstacle on the left."""
        # Create a simulated obstacle on the left side
        # MID_POINT is 315
        block_list_x = [200, 210, 220]  # All points to the left of mid point
        block_list_y = [200, 210, 220]
        block_list_distance = [1300, 1350, 1400]  # Greater than 1220 threshold
        
        result, angle = analyze_block_list(block_list_x, block_list_y, block_list_distance)
        self.assertEqual(result, RESULT_RIGHT_TURN)
        self.assertLess(angle, 0)  # Negative angle for right turn
    
    def test_analyze_block_list_center_obstacle(self):
        """Test block list analysis with obstacle in center."""
        # Create a simulated obstacle in the center
        block_list_x = [300, 315, 330]  # Points around mid point (315)
        block_list_y = [200, 210, 220]
        block_list_distance = [1300, 1350, 1400]
        
        result, angle = analyze_block_list(block_list_x, block_list_y, block_list_distance)
        self.assertEqual(result, RESULT_BLOCKED)
        self.assertEqual(angle, 0)


if __name__ == '__main__':
    unittest.main()
