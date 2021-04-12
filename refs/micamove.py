import math as m
import pyrealsense2 as rs
import time
import numpy as np
import threading
import serial
import re

from threading import Thread
from shapeTracking import shapeTracker
from serialMonitor import SerialSetup, SerialDistances
#from colourTracker import colourTracking
#from colourTrackerWithGUI import colourTracking

NUM_SAMPLES = 50
FIR_COEFF =(1/NUM_SAMPLES,
	    1/NUM_SAMPLES,
	    1/NUM_SAMPLES,
	    1/NUM_SAMPLES,
        1/NUM_SAMPLES)
GUI_FLAG = 1 #Set = 0 to turn off GUI

class mica():
    
    #
    # Constructor
    #
    def __init__(self, vel):
        
        #
        # Create a NUM_SAMPLES by 2 array of previous samples.
        # Each sample contains (target_vector_x, target_vector_y, target_vector_theta) data.
        # Used for filtering algorithms.
        #
        self.target_history_vector = [(0, 0, 0)] * NUM_SAMPLES # (target_vector_x, target_vector_y, target_vector_theta)
        self.curr_sample_idx = 0

        # Velocity at which MiCA will move
        self.velocity_multiplier = vel

        # PIDs
        self.target_theta = 0
        self.mica_theta = 0
        self.pid_output_theta = 0
        pid_thread_theta = Thread(target = self.async_theta_pid)
        pid_thread_theta.setDaemon(True)
        pid_thread_theta.start()

#        self.target_x = 0
#        self.mica_x = 0
#        self.pid_output_x = 0
#        pid_thread_x = Thread(target = self.async_dist_x_pid)
#        pid_thread_x.setDaemon(True)
#        pid_thread_x.start()
#
#        self.target_y = 0
#        self.mica_y = 0
#        self.pid_output_y = 0
#        pid_thread_y = Thread(target = self.async_dist_y_pid)
#        pid_thread_y.setDaemon(True)
#        pid_thread_y.start()

        self.ult_left_dist = 0
        self.ult_right_dist = 0
        thread_get_ult_dist = Thread(target = self.async_get_ult_dist)
        thread_get_ult_dist.setDaemon(True)
        thread_get_ult_dist.start()


    def async_get_ult_dist(self):
        lock = threading.Lock()
        mcu_aux_serial = serial.Serial('/dev/mcu_aux', 9600, timeout=.1) 
        time.sleep(1) #give the connection a second to settle
        while True:
            try:
                data = mcu_aux_serial.readline()
                values = re.split(r'\t',data)
                values = [x.replace("\r\n","") for x in values]
                left_dist = float(values[0])
                right_dist= float(values[1])
            except:
                left_dist = -1
                right_dist = -1

            lock.acquire()
            try:
                self.ult_left_dist = left_dist
                self.ult_right_dist = right_dist
            finally:
                lock.release()


    #
    # Update the pose history vector and return the current pose
    #
    # Arguments:
    #   N/A
    #
    # Return Values:
    #
    # x     : The x position of the robot
    # y     : The y position of the robot
    # theta : The angle theta of the robot
    # t     : Timestamp of the data above
    def get_pose(self, pipeT265):
        time_in = time.time()
        data = rs.pose

	x = 0
	y = 0
	yaw = 0

        # Get the data from the T265 Tracking camera
        try:
            # Wait for the next set of frames from the camera
            frames = pipeT265.poll_for_frames()
            pose = frames.get_pose_frame()
            if pose:
                data = pose.get_pose_data()
                # Set output values
                x = data.translation.x
                y = data.translation.y
                quat_x = data.rotation.x
                quat_y = data.rotation.z # z and y are swapped in VR coords
                quat_z = data.rotation.y # 
                quat_w = data.rotation.w

	        yaw = m.atan2(2.0*(quat_x*quat_y), quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
	        yaw *= 180.0 / m.pi
	        if yaw < 0: yaw += 360
	        if yaw > 180: yaw = yaw - 360

	        # Quat to euleur
                yaw = m.atan2(2.0*(quat_x*quat_y + quat_w*quat_z), quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
                yaw *= 180.0 / m.pi
                if yaw < 0.0: yaw += 360.0
                if yaw > 180.0: yaw = yaw - 360.0

        except:
            print("No 	 :(")

        t = int(round(time.time()*1000))

        return(x, y, yaw, t)

    #
    # Get the target pose
    #
    # Arguments:
    #   N/A
    #
    # Return Values:
    #
    # x : Target x position
    # y : Target y position
    def get_target(self,pipeD435, mtx, dist):

        frame_depth, frame_x, frame_y = 0, 0, 0
        # Get the target position on the frame
        #x,y,depth=colourTracking(frame_depth, frame_x, frame_y, pipeD435)
        x,y,depth=shapeTracker(frame_depth, frame_x, frame_y, pipeD435, mtx, dist, GUI_FLAG)
        theta = 0
        theta = (x /600*87-87.0/2)
	if (theta < 0): theta += 360
        if (theta > 180): theta = theta - 360
        
        return(depth, x, theta) # x, y, theta

    #
    # Set the speed at which MiCA should move
    # idfk what the units for this number is
    #
    def set_velocity(self, vel):
        self.velocity_multiplier = vel

    def async_theta_pid(self):
        lock = threading.Lock()
        KP = 0.75
        KD = 0.015
        KI = 0
        SAMPLE_TIME = 0.1

        prev_error = 0
        sum_error = 0
        t2 = 0
        delta_t = 0

        while True:
            t1 = time.time()*1000
            theta_error = self.target_theta
            
            output = (theta_error * KP) \
                    + (prev_error * KD) \
                    + (sum_error * KI)

            if delta_t == 0:
                prev_error = 0
            else:
                prev_error = theta_error / delta_t
	        sum_error += theta_error

            lock.acquire()
            try:
                self.pid_output_theta = output
            finally:
                lock.release()

            delta_t = 0
            while (delta_t < SAMPLE_TIME):
                t2 = time.time()*1000
                delta_t = t2 - t1

#    def async_dist_x_pid(self):
#        lock = threading.Lock()
#        KP = 0.75
#        KD = 0.015
#        KI = 0
#        SAMPLE_TIME = 0.1
#
#        prev_error = 0
#        sum_error = 0
#        t2 = 0
#        delta_t = 0
#
#        while True:
#            t1 = time.time()*1000
#            error = self.target_x + HOLD_OFF_DIST
#            
#            output = (error * KP) \
#                    + (prev_error * KD) \
#                    + (sum_error * KI)
#
#            if delta_t == 0:
#                prev_error = 0
#            else:
#                prev_error = error / delta_t
#	        sum_error += error
#
#            lock.acquire()
#            try:
#                self.pid_output_x = output
#            finally:
#                lock.release()
#
#            delta_t = 0
#            while (delta_t < SAMPLE_TIME):
#                t2 = time.time()*1000
#                delta_t = t2 - t1
#
#    def async_dist_y_pid(self):
#        lock = threading.Lock()
#        KP = 0.75
#        KD = 0.015
#        KI = 0
#        SAMPLE_TIME = 0.1
#
#        prev_error = 0
#        sum_error = 0
#        t2 = 0
#        delta_t = 0
#
#        while True:
#            t1 = time.time()*1000
#            error = self.target_y
#            
#            output = (error * KP) \
#                    + (prev_error * KD) \
#                    + (sum_error * KI)
#
#            if delta_t == 0:
#                prev_error = 0
#            else:
#                prev_error = error / delta_t
#	        sum_error += error
#
#            lock.acquire()
#            try:
#                self.pid_output_y = output
#            finally:
#                lock.release()
#
#            delta_t = 0
#            while (delta_t < SAMPLE_TIME):
#                t2 = time.time()*1000
#                delta_t = t2 - t1

    # Performs navigation tasks
    #
    # Arguments:
    #   N/A
    #
    # Return Values:
    #   vx: The velocity in the x direction
    #   vy: The velocity in the y direction
    #
    def navigate(self, pipeD435, pipeT265, mtx, dist):

        # Get our pose and the target pose then create a vector between them
        (mica_x, mica_y, mica_theta, t) = self.get_pose(pipeT265)
        (target_x, target_y, target_theta) = self.get_target(pipeD435,mtx,dist)

	if (target_x == -1):
	    target_x = mica_x
	    target_y = mica_y

        target_vector_x = target_x - mica_x
        target_vector_y = target_y - mica_y
        theta_error = target_theta - mica_theta

        # Update the target history vector
        self.target_history_vector[self.curr_sample_idx] = (target_vector_x, target_vector_y, theta_error)

        # Increment the current sample index by 1 or warp to 0 if we've reached NUM_SAMPLES
	if (self.curr_sample_idx >= (NUM_SAMPLES - 1)):
	    self.curr_sample_idx = 0
	else:
	    self.curr_sample_idx = self.curr_sample_idx + 1

        # Average out the history of the target vectors
        target_vector_x_filtered = 0
        target_vector_y_filtered = 0

        for sample in self.target_history_vector:
            target_vector_x_filtered = target_vector_x_filtered + sample[0] / NUM_SAMPLES
            target_vector_y_filtered = target_vector_y_filtered + sample[1] / NUM_SAMPLES

        # Normalize the filtered values
        normalizer = max(abs(target_vector_x_filtered), abs(target_vector_y_filtered))
	target_vector_x_norm = 0
	target_vector_y_norm = 0
	if (normalizer != 0):
            target_vector_x_norm = target_vector_x_filtered / normalizer
            target_vector_y_norm = target_vector_y_filtered / normalizer

        # Convert the vector into a velocity vector
	target_vector_x_vel = max(min(self.velocity_multiplier, target_vector_x_norm * self.velocity_multiplier), -self.velocity_multiplier);
	target_vector_y_vel = max(min(self.velocity_multiplier, target_vector_y_norm * self.velocity_multiplier), -self.velocity_multiplier);

	theta_vel_bounded = max(min(20, self.pid_output_theta), -20);

        # If angle is small, sidestep instead of turn
        if(abs(theta_vel_bounded) > 6):
            theta_vel_bounded = 0
            if(theta_vel_bounded > 0):
                target_vector_x_vel = 10
            else:
                target_vecotr_x_vel = -10

        print(str(self.ult_left_dist) + " " + str(self.ult_right_dist))

        if(self.ult_left_dist < 30 and target_vector_y_vel < 0):
            target_vector_x_vel = 0

        if(self.ult_right_dist < 30 and target_vector_y_vel > 0):
            target_vector_x_vel = 0

        return(target_vector_y_vel, target_vector_x_vel, theta_vel_bounded)
        #return(0, 0, theta_vel_bounded)
