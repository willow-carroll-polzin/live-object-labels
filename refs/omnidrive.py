from __future__ import absolute_import, division, print_function

import logging

import rospy
from std_msgs.msg import Int16MultiArray


ESTOP_INT = -32768


class Omnidrive(object):
  def __init__(self, topic_name):
    self.logger = logging.getLogger("omnidrive")
    self.logger.info("starting drivetrain publisher")
    self.publisher = rospy.Publisher(topic_name, Int16MultiArray, queue_size=10)

  def set_velocity(self, vx, vy, omega):
    self._publish(vx, vy, omega)

  def emergency_stop(self):
    self.logger.warn("EMERGENCY STOP SENT!")
    self._publish(ESTOP_INT, ESTOP_INT, ESTOP_INT)

  def _publish(self, v1, v2, v3):
    msg = Int16MultiArray()
    msg.data = [int(round(v1)), int(round(v2)), int(round(v3))]
    self.publisher.publish(msg)
