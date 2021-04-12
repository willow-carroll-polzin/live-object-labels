from __future__ import absolute_import, division, print_function

import logging
import os
import time

import pygame


class Gamepad(object):
  def __init__(self, j):
    if not getattr(self, "logger", None):
      self.logger = logging.getLogger("gamepad")

    self.j = j
    self.num_axes, self.num_btns, self.num_hats = self.j.get_numaxes(), self.j.get_numbuttons(), self.j.get_numhats()
    self.logger.info("gamepad detected: {} with {} axis, {} hats, and {} buttons".format(self.j.get_name(), self.num_axes, self.num_hats, self.num_btns))

  def get(self):
    pygame.event.pump()

    axis_data = [0 for i in range(self.num_axes)]
    button_data = [0 for _ in range(self.num_btns)]
    hat_data = [0 for _ in range(self.num_hats)]

    for i in range(self.num_axes):
      axis_data[i] = self.j.get_axis(i)

    for i in range(self.num_btns):
      button_data[i] = self.j.get_button(i)

    for i in range(self.num_hats):
      hat_data[i] = self.j.get_hat(i)

    return self.fix_data(axis_data, button_data, hat_data)

  def fix_data(self, axis_data, button_data, hat_data):
    return axis_data, button_data, hat_data


class F310XPad(Gamepad):
  AXIS_LEFT_X = 0
  AXIS_LEFT_Y = 1
  AXIS_RIGHT_X = 3
  AXIS_RIGHT_Y = 4
  AXIS_LT = 2
  AXIS_RT = 5

  BTN_A = 0
  BTN_B = 1
  BTN_Y = 3
  BTN_X = 2
  BTN_LB = 4
  BTN_RB = 5
  BTN_L_STICK = 9
  BTN_R_STICK = 10
  BTN_START = 7
  BTN_BACK = 6
  BTN_CENTER = 8

  HAT_DPAD_X = 0
  HAT_DPAD_Y = 1

  def __init__(self, *args, **kwargs):
    self.logger = logging.getLogger("f310xpad")
    super(F310XPad, self).__init__(*args, **kwargs)
    self._seen_lt_move = False
    self._seen_rt_move = False

    self._start_button_pressed_last = False

  def fix_data(self, axis_data, button_data, hat_data):
    # Because they start at 0 for some god damn reason
    if not self._seen_lt_move:
      if axis_data[self.AXIS_LT] != 0:
        self._seen_lt_move = True
      else:
        axis_data[self.AXIS_LT] = -1.0

    if not self._seen_rt_move:
      if axis_data[self.AXIS_RT] != 0:
        self._seen_rt_move = True
      else:
        axis_data[self.AXIS_RT] = -1.0

    axis_data[self.AXIS_LEFT_Y] *= -1
    axis_data[self.AXIS_RIGHT_Y] *= -1

    return axis_data, button_data, hat_data

  def interpret_data(self, axis_data, button_data, hat_data):
    """Return vx, vy, omega, emergency_button_pressed, auto_mode_pressed

    +x is the forward direction
    +y is towards the right
    +omega is clockwise

    Values between -1 and 1
    """
    x_axis = axis_data[self.AXIS_LEFT_X]
    y_axis = axis_data[self.AXIS_LEFT_Y]
    right_x_axis = axis_data[self.AXIS_RIGHT_X]
    # rt = (axis_data[F310XPad.AXIS_RT] + 1.0) / 2.0
    # lt = (axis_data[F310XPad.AXIS_LT] + 1.0) / 2.0

    vx = y_axis
    vy = x_axis
    # omega = int(round((rt - lt) * self.SPEED_MULTIPLIER))
    omega = right_x_axis
    emergency_button_pressed = button_data[self.BTN_B]
    auto_mode_pressed = button_data[self.BTN_A]
    x_button_pressed = button_data[self.BTN_X]

    start_button_pressed = button_data[self.BTN_START]
    start_button_released = (not start_button_pressed and self._start_button_pressed_last)
    self._start_button_pressed_last = start_button_pressed

    return vx, vy, omega, emergency_button_pressed, start_button_released, auto_mode_pressed, x_button_pressed


class ThrustMasterTFlight(Gamepad):
  "Not used anymore"
  AXIS_LEFT_Y = 1
  AXIS_LEFT_X = 0
  AXIS_RIGHT_X = 3
  BTN_B = 8

  def fix_data(self, axis_data, button_data, hat_data):
    axis_data[self.AXIS_LEFT_Y] *= -1
    return axis_data, button_data, hat_data

  def interpret_data(self, axis_data, button_data, hat_data):
    x_axis = axis_data[self.AXIS_LEFT_X]
    y_axis = axis_data[self.AXIS_LEFT_Y]
    right_x_axis = axis_data[self.AXIS_RIGHT_X]
    # rt = (axis_data[F310XPad.AXIS_RT] + 1.0) / 2.0
    # lt = (axis_data[F310XPad.AXIS_LT] + 1.0) / 2.0

    vx = y_axis
    vy = x_axis
    # omega = int(round((rt - lt) * self.SPEED_MULTIPLIER))
    omega = right_x_axis
    emergency_button_pressed = button_data[self.BTN_B]
    auto_mode_pressed = button_data[self.BTN_A]

    return vx, vy, omega, emergency_button_pressed, auto_mode_pressed


GAMEPAD_MAP = {
  "Logitech Gamepad F710": F310XPad,
  "Logitech Gamepad F310": F310XPad,
  "Thrustmaster T.Flight Hotas X": ThrustMasterTFlight,
}


def initialize_gamepad():
  logger = logging.getLogger("gamepad")

  os.environ["SDL_VIDEODRIVER"] = "dummy"

  logger.info("attempting to get joysticks")
  pygame.init()

  count = pygame.joystick.get_count()

  while True:
    while count == 0:
      time.sleep(0.5)
      pygame.joystick.quit()
      pygame.joystick.init()
      count = pygame.joystick.get_count()
      logger.info("joystick not found, waiting for it to be plugged in...")

    logger.info("joystick found! initializing")

    # 0 harded coded is bad.
    # TODO: get rid of the hardcode here
    j = pygame.joystick.Joystick(0)
    j.init()

    name = j.get_name()
    if name not in GAMEPAD_MAP:
      logger.warn("{} is not a valid gamepad to be used".format(name))
      time.sleep(0.5)
    else:
      break

  return GAMEPAD_MAP[name](j)


if __name__ == "__main__":
  logging.basicConfig(format="[%(asctime)s][%(name).10s][%(levelname).1s] %(message)s",
                      datefmt="%Y-%m-%d %H:%M:%S",
                      level=logging.DEBUG)
  gamepad = initialize_gamepad()
  while True:
    axis_data, button_data, hat_data = gamepad.get()
    gamepad.logger.info("axis: {} | button: {} | hat: {}".format(axis_data, button_data, hat_data))
    vx, vy, omega, emergency = gamepad.interpret_data(axis_data, button_data, hat_data)
    gamepad.logger.info("vx = {}, vy = {}, omega = {}, emergency = {}".format(vx, vy, omega, emergency))
    time.sleep(0.3)
