# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from aruco_tracker import *
import pyrealsense2 as rs
import numpy as np
import argparse
import cv2
import imutils
import time
import math as mt
#import statistics as st

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

def shapeTracker(depth, x, y, pipeline, mtx, dist, guiShow): 
    try:
        ###############################################
        #Wait for a coherent pair of frames: depth and color
        frames = pipeline.poll_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
    finally:
        #print("Done tracking")
        pass
        #pipeline.stop()
        
    if depth_frame and color_frame:
        #############################
        #TRACKING SETUP: 
        font = cv2.FONT_HERSHEY_COMPLEX 
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
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
        cnts_col = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_col = imutils.grab_contours(cnts_col)
        center = None #Initilize the centre of the object to null
        #############################

        
        #############################
        #COLOUR TRACKING:
        #Only proceed if at least one contour was found
        #Only proceed if at least one contour was found
        if len(cnts_col) > 0:
            #Find the largest contour in the mask, then use
            #it to compute the minimum enclosing circle and centroid
            c = max(cnts_col, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #Compute centroid of object
            
            #Determine the distance to the objects centroid
            #if center == None:
            #    dist = 0.00
            #else:
            #    dist = depth_frame.get_distance(center[0],center[1])
            
            #Display colour tracking
            #if (guiShow == 1):
                #Only proceed if the radius meets a minimum size
                #if radius > 1:
                    #print("YELLOW!")
                    #.circle(color_image, (int(x), int(y)), int(radius), (0, 255, 255), 2) #Draw the circle and centroid on the frame 
                    #cv2.circle(color_image, center, 5, (0, 0, 255), -1) #Update the list of tracked points
                #Display compressed frame with bound and tail
                    #cv2.imshow("COLOUR TRACKING", color_image)

            
            #print("Centre is: ", center)
            #print("Distance to centre is: ", dist)
            
            #depth = dist #Store the dist and return to caller
            
            #return x,y,depth
            
        # update the points queue
        #pts.appendleft(center)
        
        #else:
            #print("NO TARGET FOUND!")
            #x,y,depth = -1,-1,-1
            #return x,y,depth

        #############################
        #ARUCO TRACKER:
        corners, ret = arucoTracker(color_image, mtx, dist)
   
        #Find centroid of board
        dist_thresh = 1000
        centroids = []
        if (ret == True):
            for i in range(len(corners[0][0])):
                for j in range(i+1,len(corners[0][0])):
                    val = mt.sqrt((corners[0][0][i][0]-corners[0][0][j][0])**2+(corners[0][0][i][1]-corners[0][0][j][1])**2)
                    if val < dist_thresh: 
                        centroids.append(corners[0][0][i])
                        centroids.append(corners[0][0][j])
        
            #Remove duplicates from list:
            for i in range(len(centroids)):
                centroids[i]=tuple(centroids[i])
            centroids=list(set(centroids))  
            
            avg_centroid_X = []
            avg_centroid_Y = []
            centroid_X = [] 
            centroid_Y = []
            for k in range(len(centroids)):
                centroid_X.append(centroids[k][0])
                centroid_Y.append(centroids[k][1])
            
            #Only average if data recieved, else set centroid average to arb value
            if (len(centroid_X) == 0 or len(centroid_Y) == 0):
                avg_centroid_X = 0
                avg_centroid_Y = 0
            else: 
                avg_centroid_X = (centroid_X[0]+centroid_X[1])/2
                avg_centroid_Y = (centroid_Y[0]+centroid_Y[1])/2 
            avg_centroid = [avg_centroid_X, avg_centroid_Y]
            
            #Determine the distance to the shapes centroid
            if avg_centroid == None:
                dist = 0.00
            else:
                dist = depth_frame.get_distance(int(avg_centroid[0]),int(avg_centroid[1]))   
                #dist = 100  
            depth = dist #Store the dist and return to caller
            
            #Display aruco tracking
            if (guiShow == 1):
                cv2.circle(color_image, (int(avg_centroid_X), int(avg_centroid_Y)), 7, (45, 145, 50), -1)
                cv2.putText(color_image, "AVG", (int(avg_centroid_X) - 20, int(avg_centroid_Y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 145, 50), 2)
                cv2.imshow('ARUCO TRACKING',color_image)
                cv2.waitKey(1)
            #print ("Centre is: ", avg_centroid)
            return (avg_centroid_X,avg_centroid_Y,depth)
        ############################# 
    return -1,-1,-1
