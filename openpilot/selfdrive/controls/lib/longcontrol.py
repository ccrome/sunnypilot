import numpy as np
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants
from opendbc.car.honda.values import CAR

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState

# CAN-input identification, not per-speed controller tuning. GAS_COMMAND
# full scale represents 2.2 in the fitted plant but the CR-V encoder maps an
# effort command of 2.0 to full scale. Keep that unit conversion explicit.
CRV_DRIVE_EFFORT_GAIN = 1.6040 * 2.2 / 2.0
CRV_BRAKE_EFFORT_GAIN = 0.9250
CRV_BRAKE_RESPONSE_TIME = 0.5270
CRV_ROLLING_ACCEL = 0.1888
CRV_DRAG_COEFFICIENT = 0.0003693


def long_control_state_trans(CP_SP, active, long_control_state,
                             should_stop, brake_pressed, cruise_standstill):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if should_stop:
        long_control_state = LongCtrlState.stopping

  return long_control_state

class LongControl:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController(0.0,
                             0.4 if CP.carFingerprint == CAR.HONDA_CRV_5G else (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    crv = CP.carFingerprint == CAR.HONDA_CRV_5G
    self.drive_gain = CRV_DRIVE_EFFORT_GAIN if crv else 1.0
    self.brake_gain = CRV_BRAKE_EFFORT_GAIN if crv else 1.0
    self.rolling = CRV_ROLLING_ACCEL if crv else 0.0
    self.drag = CRV_DRAG_COEFFICIENT if crv else 0.0
    self.last_output_accel = 0.0

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0] * self.brake_gain
    self.pid.pos_limit = accel_limits[1] * self.drive_gain

    self.long_control_state = long_control_state_trans(self.CP_SP, active, self.long_control_state,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        # TODO: can we just go straight to stopAccel?
        output_accel -= 1.0 * DT_CTRL  # m/s^2/s while trying to stop
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      # Integrating acceleration error is velocity-error feedback. It does
      # not differentiate noisy wheel speeds again or switch pedals for a
      # single acceleration sample. Feedforward supplies the identified road
      # load and converts net acceleration to actuator effort; the integral
      # learns the remaining load/grade error continuously.
      effort = self.pid.update(error, speed=CS.vEgo,
                               feedforward=a_target + self.rolling + self.drag * CS.vEgo ** 2)
      output_accel = max(effort, 0.0) / self.drive_gain + min(effort, 0.0) / self.brake_gain

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
