###############################################
#LIBS:
from collections import deque
from imutils.video import VideoStream
import pyrealsense2 as rs
import numpy as np
import argparse
import cv2
import imutils
import time

def colourTracking(depth,x,y, pipeline): 
    ###############################################
    #SETUP:

    #Configure depth and color streams
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # #Start streaming
    # pipeline.start(config)

    try:
        ###############################################
        #Wait for a coherent pair of frames: depth and color
        frames = pipeline.poll_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

    finally:
        print("Done tracking")
        #pipeline.stop()

    if depth_frame and color_frame:
        #Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #Stack both the orginal colour image and the depth-colour map horizontally
        images = np.hstack((color_image, depth_colormap))

        #Define the lower and upper boundaries of the tracking colour "yellow"
        colourLower = (22, 150, 55) #HSV colour bound
        colourUpper = (40, 255, 255) #HSV colour bound
        pts = deque(maxlen=64) #Initialize the list of tracked points
         
        color_image = imutils.resize(color_image, width=600)       #Reduce size of frame
        blurred = cv2.GaussianBlur(color_image, (11, 11), 0) #Blur to reduce high freq noise
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #Convert to HSV

        #Construct a mask for the color "yellow", then perform
        #a series of dilations and erosions to remove noise:
        mask = cv2.inRange(hsv, colourLower, colourUpper) #Supply range of colour
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        #Find contours in the mask and initialize the current (x, y) center of the object:
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
            #print("Centre is: ", center)
            #print("Distance to centre is: ", dist)
            depth = dist #Store the dist and return to caller
            return x,y,depth
        else:
            print("NO TARGET FOUND!")
            x,y,depth = -1,-1,-1
	    return x,y,depth
    return -1,-1,-1
