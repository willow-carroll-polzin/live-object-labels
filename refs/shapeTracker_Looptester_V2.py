# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import pyrealsense2 as rs
import numpy as np
import argparse
import cv2
import imutils
import time
import math as mt
from ShapeDetector import ContourDetector
#from ShapeDetector3 import ContourDetector
from ShapeDetector import ShapeDetector
#from ShapeDetector3 import ShapeDetector
import statistics as st
from aruco_tracker import *

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

#Calibrate
mtx, dist = arucoTrackerCal()
guiShow = 1

# keep looping
while True:
    #############################
    #INPUT SETUP:
    
    # grab the current frame
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    print("FRAME: ", color_frame)
    
    if not depth_frame or not color_frame:
        continue
        
    font = cv2.FONT_HERSHEY_COMPLEX 
    
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    print("IMAGE: ", color_image)
    cv2.imshow("test", color_image)
    
    # #############################
    # #ARUCO TRACKER:
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
        avg_centroid_X = st.mean(centroid_X) ##TODO: FIX
        avg_centroid_Y = st.mean(centroid_Y) ##TODO: FIX
    avg_centroid = [avg_centroid_X, avg_centroid_Y]
    
    #Determine the distance to the shapes centroid
    if avg_centroid == None:
        dist = 0.00
    else:
        dist = depth_frame.get_distance(avg_centroid[0],avg_centroid[1])   
        #dist = 100  
    depth = dist #Store the dist and return to caller
    
    #Display aruco tracking
    if (guiShow == 1):
        cv2.circle(color_image, (int(avg_centroid_X), int(avg_centroid_Y)), 7, (45, 145, 50), -1)
        cv2.putText(color_image, "AVG", (int(avg_centroid_X) - 20, int(avg_centroid_Y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 145, 50), 2)
        cv2.imshow('ARUCO TRACKING',color_image)

    print("Centre is: ", avg_centroid)
    print(avg_centroid_X,avg_centroid_Y)
    
    cv2.waitKey(1)
    ############################ 
    
pipeline.stop()
