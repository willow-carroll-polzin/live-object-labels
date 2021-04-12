## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

#IMPORTS:
from collections import deque
from imutils.video import VideoStream
import pyrealsense2 as rs
import numpy as np
import argparse
import cv2
import imutils
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

while True:
    ###########
    ###########
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)
    #key = cv2.waitKey(1) & 0xFF
    
    # for y in range(480):
        # for x in range(640):
            # dist = depth_frame.get_distance(x,y)
    #dist = depth_frame.get_distance(10,100)

    #print(depth_frame)
    
    ###########
    ###########
    # #Define the lower and upper boundaries of the tracking colour "yellow"
    colourLower = (22, 150, 55) #HSV colour
    colourUpper = (40, 255, 255) #HSV colour
    pts = deque(maxlen=64) #Initialize the list of tracked points
     
    # #Grab the reference to the camera
    # vs = VideoStream(src=1).start() #src=0 is camera 1 (D435i), src=1 is camera 2 (T265)
    
    # #Grab the current frame
    # frame = vs.read()
    # #frame = color_image

    #color_image = imutils.resize(color_image, width=600)       #Reduce size of frame
    blurred = cv2.GaussianBlur(color_image, (11, 11), 0) #Blur to reduce high freq noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #Convert to HSV

    #Construct a mask for the color "yellow", then perform
    #a series of dilations and erosions to remove any small
    #blobs left in the mask:
    mask = cv2.inRange(hsv, colourLower, colourUpper) #Supply range of colour
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # #Find contours in the mask and initialize the current (x, y) center of the object:
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None #Initilize the centre of the object to null
    
    #Only proceed if at least one contour was found
    if len(cnts) > 0:
        #Find the largest contour in the mask, then use
        #it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #Compute centroid of object
        
        #Determine the distance to the objects centroid
        if center == None:
            dist = 0.00
        else:
            dist = depth_frame.get_distance(center[0],center[1])
            
        #Only proceed if the radius meets a minimum size
        if radius > 1:
            cv2.circle(color_image, (int(x), int(y)), int(radius), (0, 255, 255), 2) #Draw the circle and centroid on the frame 
            cv2.circle(color_image, center, 5, (0, 0, 255), -1) #Update the list of tracked points

        print("Centre is: ", center)
        print("Distance to centre is: ", dist)
    
    #Add centroid to image by updating the points queue
    pts.appendleft(center)
    #print("Centre is: ", center)
    #print("Distance to centre is: ", dist)
    
    # Draw trail of object:
    for i in range(1, len(pts)): #Loop over the set of tracked points
        # If the current tracked point is None, ignore
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5) #Compute the thickness of the line
        cv2.line(color_image, pts[i - 1], pts[i], (0, 0, 255), thickness) #Draw the connecting lines
        
    #Display compressed frame with bound and tail
    cv2.imshow("Frame", color_image)

    ###########
    ###########

pipeline.stop()
