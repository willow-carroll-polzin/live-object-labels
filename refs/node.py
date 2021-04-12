#***********************
#LIBS:
from __future__ import absolute_import, division, print_function

import logging
import time
#Ros
import rospy
from std_msgs.msg import String

#Motor drivers and gamepad:
from .omnidrive import Omnidrive
from .gamepad import initialize_gamepad

#Target tracking and velocity:
import pyrealsense2 as rs 
from .micamove import *
from .aruco_tracker import arucoTrackerCal

#***********************
# logging.basicConfig(format="[%(asctime)s][%(name).10s][%(levelname).1s] %(message)s",
#                     datefmt="%Y-%m-%d %H:%M:%S",
#                     level=logging.DEBUG)

#***********************
#CONSTANTS:
NODE_NAME = "itadomni"
PUBLISHING_TOPIC_NAME = "omnidrive"
SUBSCRIBING_TOPIC_NAME = "omnidrive_feedback"
#***********************
#***********************
#ItadOmni base class:
class ItadOmni(object):
  @classmethod
  def main(cls):
    print("Hello, ITAD!")
    node = cls(NODE_NAME, SUBSCRIBING_TOPIC_NAME, PUBLISHING_TOPIC_NAME)
    node.run()
    
  READY_MSG = "!!ready!!"

  def subscribing_callback(self, data):
    if data.data == self.READY_MSG:
      self.ready = True
      self.logger.info("feedback topic reports mcu is ready")
    else:
      self.logger.debug(data.data)

  def __init__(self, node_name, subscribing_topic_name, publishing_topic_name):
    self.ready = False
    self.emergency = False
    rospy.init_node(node_name)
    self.subscriber = rospy.Subscriber(subscribing_topic_name, String, self.subscribing_callback)

    self.logger = logging.getLogger(node_name)
    self.drivetrain = Omnidrive(publishing_topic_name)
    
    #This will block until the gamepad exists
    self.gamepad = initialize_gamepad()

    #Maximum is 500
    self.speed_multiplier = 100
    
    #MiCA object
    self.mica = mica(self.speed_multiplier)
    
    self.mtx, self.dist = arucoTrackerCal()

  def run(self):
    self.logger.info("starting node... waiting for mcu to become ready")
    r = rospy.Rate(30)
    auto_mode = False
    #Init D435i camera pipeline:
    pipeD435 = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeD435.start(config)
    
    #Init T265 camera pipeline:
    pipeT265 = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)
    pipeT265.start(cfg)
    

    while not rospy.is_shutdown():
      r.sleep()
      if self.emergency:
        self.drivetrain.emergency_stop()
        continue

      # Gamepad input and controls:
      gamepad_values = self.gamepad.get()
      vx, vy, omega, emergency_button_pressed, start_button_pressed, auto_mode_pressed, x_btn= self.gamepad.interpret_data(*gamepad_values)
      
      # Check if autonomous or manual mode engaged:
      if auto_mode_pressed:
        print("Changing control master")
        auto_mode = not auto_mode
        
      # Intel RealSense input:
      if auto_mode:
        vx, vy , omega = self.mica.navigate(pipeD435, pipeT265, self.mtx, self.dist)
	self.speed_multiplier = 7
	self.mica.set_velocity(self.speed_multiplier)
      else:
	self.speed_multiplier = 250

      print("hoho " + str(vx) + " " + str(vy) + " " + str(omega)) 

      # Check for button presses:
      if emergency_button_pressed:
          self.emergency = True
          self.logger.info("Triggered emergency stop")
          self.drivetrain.emergency_stop() #Publish emergency stop to drivetrain ros topic

      # Cycle through speed settings
      if start_button_pressed:
        if self.speed_multiplier == 250:
            self.speed_multiplier = 500
        elif self.speed_multiplier == 500:
            self.speed_multiplier = 125
        elif self.speed_multiplier == 125:
          self.speed_multiplier = 250
      ###########
      #Apply multipliers: 
      # The drivetrain is mapped backwards as positive x is designed to be
      # backward
      vx = -int(round(vx * self.speed_multiplier))
      vy = int(round(vy * self.speed_multiplier))

      # 0.7 multiplier is needed to reduce the speed of the turn, which is much
      # faster than other directions of motion as all motors are in use.
      omega = int(round(omega * self.speed_multiplier * 0.7))
      ###########

      #Publish velocities to drivetrain ros topic
      self.drivetrain.set_velocity(vx, vy, omega)
      #####
      #Check if target has been reached
      #self.checkTarg = path.targReached()
      #if checkTarg ==0 {
        #print("Target has been reached. Stop")
        #self.drivetrain.set_velocity(0,0,0)
      #####
    self.logger.info("shutdown")
    
    pipeD435.stop()
    pipeT265.stop()
#***********************

#itad = ItadOmni("itadomni", "itadomni", "itadomni")
