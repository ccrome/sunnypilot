import numpy as np
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants
from opendbc.car.honda.values import CAR
from opendbc.car.honda.hondacan import CRV_BRAKE_RELEASE_ACCEL
from openpilot.selfdrive.controls.lib.terminal_stop import CRV_WHEEL_SPEED_CUTOFF

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState

# CAN-input identification, not per-speed controller tuning. GAS_COMMAND
# full scale represents 2.2 in the fitted plant but the CR-V encoder maps an
# effort command of 2.0 to full scale. Keep that unit conversion explicit.
CRV_DRIVE_EFFORT_GAIN = 1.6040 * 2.2 / 2.0
CRV_BRAKE_EFFORT_GAIN = 0.9250
CRV_BRAKE_RESPONSE_TIME = 0.5270
# Low-speed CAN replay identifies negative acceleration as net deceleration.
# Keep ordinary acceleration regulation unchanged; terminal tracking uses
# this independently identified request-to-net-acceleration conversion.
CRV_BRAKE_NET_GAIN = 1.0341
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
    self.debug = {}

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, stop_phase=0, v_target=0., j_target=0., safety_pressure=0.):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    terminal_stop = self.CP.carFingerprint == CAR.HONDA_CRV_5G and stop_phase != 0
    brake_gain = CRV_BRAKE_NET_GAIN if terminal_stop else self.brake_gain
    self.pid.neg_limit = accel_limits[0] * brake_gain
    # A committed stop retains brake authority even while inverse-dynamics
    # feedforward releases pressure. Zero request would release the brake
    # entirely and expose CVT creep. This is a pedal constraint, not a reset.
    safety_pressure = float(np.clip(safety_pressure, 0., 1.)) * float(active)
    # The Honda brake request can remain asserted for a small positive net
    # acceleration command. Permit that inverse-dynamics easing while keeping
    # the terminal command below the brake-to-gas release boundary.
    normal_pos_limit = CRV_BRAKE_RELEASE_ACCEL - .01 if terminal_stop else accel_limits[1] * self.drive_gain
    self.pid.pos_limit = min(normal_pos_limit, normal_pos_limit + safety_pressure *
                            (min(a_target, 0.) * brake_gain - normal_pos_limit))
    self.debug = {'active': bool(active), 'terminalStop': bool(terminal_stop), 'accelerationError': 0.,
                  'speedConfidence': 1., 'targetAcceleration': float(a_target), 'targetJerk': float(j_target),
                  'safetyPressure': float(safety_pressure), 'brakeGain': float(brake_gain),
                  'positiveEffortLimit': float(self.pid.pos_limit), 'negativeEffortLimit': float(self.pid.neg_limit),
                  'pidUpdated': False}

    self.long_control_state = long_control_state_trans(self.CP_SP, active, self.long_control_state,
                                                       CS.standstill if terminal_stop else should_stop, CS.brakePressed,
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
      if self.CP.carFingerprint != CAR.HONDA_CRV_5G:
        self.reset()

    else:  # LongCtrlState.pid
      # The Honda wheel channels are censored below ~0.58 m/s; fade velocity
      # feedback rather than interpreting their disappearance as deceleration.
      speed_confidence = max(0., CS.vEgo ** 2 - CRV_WHEEL_SPEED_CUTOFF ** 2) / (CS.vEgo ** 2 + CRV_WHEEL_SPEED_CUTOFF ** 2)
      error = (speed_confidence if terminal_stop else 1.) * (a_target - CS.aEgo)
      self.debug.update(accelerationError=float(error), speedConfidence=float(speed_confidence if terminal_stop else 1.),
                        pidUpdated=True)
      # Integrating acceleration error is velocity-error feedback. It does
      # not differentiate noisy wheel speeds again or switch pedals for a
      # single acceleration sample. Feedforward supplies the identified road
      # load and converts net acceleration to actuator effort; the integral
      # learns the remaining load/grade error continuously.
      effort = self.pid.update(error, speed=CS.vEgo,
                               feedforward=a_target + float(not terminal_stop) * (self.rolling + self.drag * CS.vEgo ** 2)
                               + float(terminal_stop) * (1. - safety_pressure) * CRV_BRAKE_RESPONSE_TIME * j_target)
      output_accel = max(effort, 0.0) / self.drive_gain + min(effort, 0.0) / brake_gain

    # Safety constrains the applied pedal, not just a planner reference that
    # feedforward/integral or the stopping ramp can subsequently undo.
    self.debug['outputBeforeSafety'] = float(output_accel)
    output_accel += safety_pressure * (min(output_accel, a_target) - output_accel)
    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    self.debug.update(outputAfterSafety=float(output_accel), outputAcceleration=float(self.last_output_accel),
                      feedforward=float(self.pid.f), integral=float(self.pid.i), proportional=float(self.pid.p),
                      pidEffort=float(self.pid.control))
    return self.last_output_accel

  def write_debug(self, debug):
    for name, value in self.debug.items():
      setattr(debug, name, value)
