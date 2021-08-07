# Creat By JasonLTechnology
# coding=utf-8

import cv2
import numpy as np
from stereovision.calibration import StereoCalibration
import json

# Depth map default preset
SWS = 19   # SAD Windows Size (SWS) (odd 5 to 255)
PFS = 255  # Pre Filter Size(odd 5 to 255)
PFC = 63  # Pre Filter Cap (1 to 63)
MDS = -5  # Min Disparity (-100 to 100)
NOD = 80  # Number of disparities (0 to 256 & Divisible by 16)
TTH = 0  # Texture Threshold (num < 0)
UR = 10  # Uniqueness Ratio (1-20)
SR = 40  # Speckle Range (0 to 40)
SPWS = 300  # Speckle Size (0 to 300)

# start up the StereoBM Class
sbm = cv2.StereoBM_create(numDisparities=16, blockSize=SWS)
sbm.setPreFilterType(1)
sbm.setPreFilterSize(PFS)
sbm.setPreFilterCap(PFC)
sbm.setMinDisparity(MDS)
sbm.setNumDisparities(NOD)
sbm.setTextureThreshold(TTH)
sbm.setUniquenessRatio(UR)
sbm.setSpeckleRange(SR)
sbm.setSpeckleWindowSize(SPWS)

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    #file = open("log.txt","r+")
    #file.writelines(str(disparity.tolist()))
    #file.close()
    print(disparity.max())
    print(disparity.min())
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    #disparity_normalized = disparity
    #print(disparity_normalized.max())
    #print(disparity_normalized.min())
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return disparity_color, disparity_normalized, disparity


def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        distance = distance
        print("Location: {},{} - Distance: {}".format(x,y,int(distance*100)/100))
        return distance


photo_width = 1280 
img_height = 480 
img_width = 640 

pair_img = cv2.imread("put your image path here")

left_frame = pair_img[0:img_height, 0:img_width]  # Y+H and X+W
right_frame = pair_img[0:img_height, img_width:photo_width]

 # Convert BGR to Grayscale
left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

# calling all calibration results
calibration = StereoCalibration(input_folder='calib_result/')
rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))
disparity_color, disparity_normalized, disparity_map = stereo_depth_map(rectified_pair)
cv2.namedWindow("DepthMap")
while True:
    # Show depth map and image frames
    output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
    cv2.imshow("DepthMap", output)
    # Mouse clicked function
    cv2.setMouseCallback("DepthMap", onMouse, disparity_map)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    else:
        continue
