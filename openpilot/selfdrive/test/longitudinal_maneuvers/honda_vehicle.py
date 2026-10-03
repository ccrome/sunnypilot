"""Causal Honda actuator plant driven by production CAN, not planner targets.

Dynamics are a transparent grey-box approximation, not a Honda ECU emulator.
Nominal response gains/time constants are based on the archived plant fit;
transport delay is explicit and configurable because that fit did not identify it.
"""
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
import math

from opendbc.can import CANParser
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.values import CAR, DBC
from openpilot.selfdrive.controls.lib.longcontrol import LongControl


@dataclass(frozen=True)
class HondaDynamics:
  gas_gain: float = 1.3003
  gas_tau: float = 1.1018
  brake_gain: float = 1.0179
  brake_tau: float = 0.5189
  delay: float = 0.3
  rolling: float = 0.012
  drag: float = 0.000046064


class HondaVehicle:
  def __init__(self, CP, CP_SP, speed, parameters=None):
    self.parameters = parameters or HondaDynamics()
    self.speed = speed
    self.acceleration = 0.0
    self.effort = 0.0
    self.longitudinal = LongControl(CP, CP_SP)
    self.controller = CarController(DBC[CAR.HONDA_CRV_5G], CP, CP_SP)
    self.parser = CANParser(DBC[CAR.HONDA_CRV_5G][Bus.pt], [('ACC_CONTROL', 0)], 1)
    self.command = (0.0, 0.0, False)
    self.queue = deque([self.command] * round(self.parameters.delay / DT_CTRL))
    self.frame = 0
    self.output = 0.0

  def step(self, target, should_stop, enabled, pitch, duration, cruise):
    ticks = round(duration / DT_CTRL)
    if ticks < 1 or abs(ticks * DT_CTRL - duration) > 1e-8:
      raise ValueError('Plant period must be an integer multiple of the 100 Hz controller period')
    for _ in range(ticks):
      state = structs.CarState.new_message()
      state.vEgo = float(self.speed)
      state.aEgo = float(self.acceleration)
      state.standstill = self.speed < 0.01
      state.cruiseState.available = True
      self.output = float(self.longitudinal.update(enabled, state, target, should_stop, (-3.5, 2.0)))
      cc = structs.CarControl.new_message()
      cc.enabled = enabled
      cc.longActive = enabled
      cc.actuators.accel = self.output
      cc.actuators.longControlState = self.longitudinal.long_control_state
      cc.hudControl.setSpeed = float(cruise)
      cs = SimpleNamespace(out=state, v_cruise_factor=1.0, is_metric=True,
                           acc_hud={'FCM_OFF': 0, 'FCM_OFF_2': 0, 'FCM_PROBLEM': 0, 'ICONS': 0},
                           lkas_hud={'LKAS_PROBLEM': 0, 'LKAS_OFF': 0})
      _, frames = self.controller.update(cc.as_reader(), structs.CarControlSP(), cs, self.frame * 10_000_000)
      if 0x1df in self.parser.update([(self.frame * 10_000_000, frames)]):
        values = self.parser.vl['ACC_CONTROL']
        self.command = (float(values['ACCEL_COMMAND']), max(0.0, float(values['GAS_COMMAND'])),
                        bool(values['BRAKE_REQUEST']))
      self.queue.append(self.command)
      accel, gas, braking = self.queue.popleft()
      p = self.parameters
      # Gas has its own effort channel; never infer its effect from accel sign.
      drive = p.brake_gain * min(accel, 0.0) if braking else p.gas_gain * gas / 1600.0 * 2.2
      tau = p.brake_tau if braking else p.gas_tau
      self.effort += (1.0 - math.exp(-DT_CTRL / tau)) * (drive - self.effort)
      road_load = p.rolling + p.drag * self.speed ** 2
      self.acceleration = self.effort - 9.81 * math.sin(pitch) - road_load
      next_speed = max(0.0, self.speed + self.acceleration * DT_CTRL)
      if self.speed == 0.0 and next_speed == 0.0:
        self.acceleration = 0.0
      self.speed = next_speed
      self.frame += 1
    return self.speed, self.acceleration
