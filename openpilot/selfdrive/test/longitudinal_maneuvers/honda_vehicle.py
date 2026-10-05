"""Causal Honda actuator plant driven by production CAN, not planner targets.

Dynamics are a transparent grey-box approximation, not a Honda ECU emulator.
Nominal gains, rise/release time constants, and transport delay are fitted to
recorded CAN input. The actual production speed observer closes the loop.
"""
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
import math

import numpy as np

from opendbc.can import CANParser
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.values import CAR, DBC
from opendbc.car.honda.carstate import CarState
from openpilot.selfdrive.controls.lib.longcontrol import LongControl


@dataclass(frozen=True)
class HondaDynamics:
  # Whole-route CAN-input fit: 3d + 3a training, 3b held out. See
  # calibrate_honda_vehicle.py and the calibration provenance JSON.
  gas_gain: float = 1.6040
  gas_tau: float = 0.5579
  gas_release_tau: float = 0.1222
  brake_gain: float = 1.0341
  brake_tau: float = 0.5270
  delay: float = 0.05
  rolling: float = 0.1888
  drag: float = 0.0003693
  speed_sample_period: float = 0.02
  speed_quantum: float = 0.01 / 3.6 / 4
  # Conservative training-route high-frequency raw-speed scale. This is a
  # disturbance proxy, not proof all wheel-speed variation is sensor noise.
  speed_noise_std: float = 0.006
  seed: int = 0
  # Low-speed identification: routes 41/44 + stock 3a, held out 47/3b.
  # Negative CAN acceleration is a net-deceleration request, not raw brake
  # force. Creep, minimum brake authority and pressure buildup are grey-box
  # approximations of the receiving ECU/drivetrain, not planner corrections.
  creep_accel: float = 0.5851
  creep_speed: float = 1.2660
  creep_release_tau: float = 0.0477
  brake_buildup_tau: float = 0.2614
  brake_preload: float = 0.3477
  hold_decel: float = 0.5
  # Each wheel disappears below 2.02 km/h in all four inspected drives.
  # Express the cutoff in the harness's calibrated/control-speed units.
  wheel_speed_cutoff: float = 0.5773
  wheel_speed_spread: float = 0.012


class HondaVehicle:
  def __init__(self, CP, CP_SP, speed, parameters=None):
    self.parameters = parameters or HondaDynamics()
    self.speed = speed
    self.acceleration = 0.0
    self.effort = 0.0
    self.drive_effort = 0.0
    self.brake_effort = 0.0
    self.creep_effort = 0.0
    self.brake_demand = 0.0
    self.sensor = CarState(CP, CP_SP)
    self.sensor.v_ego_kf.set_x([[float(speed)], [0.0]])
    self.measured_speed = float(speed)
    self.measured_acceleration = 0.0
    self.measured_speed_std = 0.0
    self.measured_acceleration_std = 0.0
    self.wheel_count = 4
    self.speed_dropout_time = 0.0
    self.speed_measurement_valid = True
    self.vehicle_sensors_invalid = False
    self.raw_speed = float(speed)
    self.wheel_speeds = np.full(4, float(speed))
    self.wheel_speed_error = lambda: 0.0
    self.random = np.random.default_rng(self.parameters.seed)
    self.sensor_ticks = round(self.parameters.speed_sample_period / DT_CTRL)
    if self.sensor_ticks < 1 or abs(self.sensor_ticks * DT_CTRL - self.parameters.speed_sample_period) > 1e-8:
      raise ValueError('Speed sampling must be an integer multiple of the controller period')
    self.longitudinal = LongControl(CP, CP_SP)
    self.controller = CarController(DBC[CAR.HONDA_CRV_5G], CP, CP_SP)
    self.parser = CANParser(DBC[CAR.HONDA_CRV_5G][Bus.pt], [('ACC_CONTROL', 0)], 1)
    self.command = (0.0, 0.0, False)
    self.standstill_command = False
    self.queue = deque([(self.command, False)] * round(self.parameters.delay / DT_CTRL))
    self.frame = 0
    self.output = 0.0

  def step(self, target, should_stop, enabled, pitch, duration, cruise, stop_phase=0, stop_reference=(0., 0., 0.), safety_pressure=0.):
    ticks = round(duration / DT_CTRL)
    if ticks < 1 or abs(ticks * DT_CTRL - duration) > 1e-8:
      raise ValueError('Plant period must be an integer multiple of the 100 Hz controller period')
    for _ in range(ticks):
      state = structs.CarState.new_message()
      state.vEgo = float(self.measured_speed)
      state.aEgo = float(self.measured_acceleration)
      state.standstill = self.speed < 0.01
      state.cruiseState.available = True
      self.output = float(self.longitudinal.update(enabled, state, target, should_stop, (-3.5, 2.0),
                                                 stop_phase, stop_reference[0], stop_reference[2], safety_pressure))
      cc = structs.CarControl.new_message()
      cc.enabled = enabled
      cc.longActive = enabled
      cc.actuators.accel = self.output
      cc.actuators.longControlState = self.longitudinal.long_control_state
      cc.actuators.crvStopActive = stop_phase != 0
      cc.orientationNED = [0.0, float(pitch), 0.0]
      cc.hudControl.setSpeed = float(cruise)
      cs = SimpleNamespace(out=state, v_cruise_factor=1.0, is_metric=True,
                           acc_hud={'FCM_OFF': 0, 'FCM_OFF_2': 0, 'FCM_PROBLEM': 0, 'ICONS': 0},
                           lkas_hud={'LKAS_PROBLEM': 0, 'LKAS_OFF': 0})
      _, frames = self.controller.update(cc.as_reader(), structs.CarControlSP(), cs, self.frame * 10_000_000)
      if 0x1df in self.parser.update([(self.frame * 10_000_000, frames)]):
        values = self.parser.vl['ACC_CONTROL']
        self.command = (float(values['ACCEL_COMMAND']), max(0.0, float(values['GAS_COMMAND'])),
                        bool(values['BRAKE_REQUEST']))
        self.standstill_command = bool(values['STANDSTILL'])
      self.queue.append((self.command, self.standstill_command))
      command, hold = self.queue.popleft()
      self.advance(command, pitch, hold)
      self.frame += 1
    return self.speed, self.acceleration

  def advance(self, command, pitch, standstill=False):
    """Advance physical dynamics and the real observer; also usable for replay."""
    accel, gas, braking = command
    p = self.parameters
    drive_target = p.gas_gain * gas / 1600.0 * 2.2 * (not braking)
    tau = p.gas_tau if drive_target > self.drive_effort else p.gas_release_tau
    self.drive_effort += (1.0 - math.exp(-DT_CTRL / tau)) * (drive_target - self.drive_effort)
    low_weight = math.exp(-(self.speed / p.creep_speed) ** 2)
    creep_target = p.creep_accel * low_weight * (not braking)
    creep_tau = p.gas_release_tau if creep_target > self.creep_effort else p.creep_release_tau
    self.creep_effort += (1.0 - math.exp(-DT_CTRL / creep_tau)) * (creep_target - self.creep_effort)
    road_load = p.rolling + p.drag * self.speed ** 2
    resistance = road_load + 9.81 * math.sin(pitch)
    requested_decel = p.brake_gain * accel - p.brake_preload * low_weight
    if standstill:
      requested_decel = min(requested_decel, -p.hold_decel)
    brake_target = min(0.0, requested_decel + resistance - self.drive_effort - self.creep_effort) * braking
    self.brake_demand += (1.0 - math.exp(-DT_CTRL / max(p.brake_buildup_tau * low_weight, 1e-6))) * (brake_target - self.brake_demand)
    self.brake_effort += (1.0 - math.exp(-DT_CTRL / p.brake_tau)) * (self.brake_demand - self.brake_effort)
    self.effort = self.drive_effort + self.creep_effort + self.brake_effort
    self.acceleration = self.effort - resistance
    next_speed = max(0.0, self.speed + self.acceleration * DT_CTRL)
    if self.speed == 0.0 and next_speed == 0.0:
      self.acceleration = 0.0
    self.speed = next_speed
    # DBC wheel speeds have a 0.01 km/h quantum; averaging four wheels gives
    # the smaller quantum below. Sample-and-hold precedes the production KF.
    if self.frame % self.sensor_ticks == 0:
      noise = self.random.normal(0., p.speed_noise_std) * (self.speed > 0.)
      wheel_speeds = np.maximum(0., self.speed + noise + p.wheel_speed_spread * np.array([-1.5, -.5, .5, 1.5]))
      wheel_speeds[0] = max(0.0, wheel_speeds[0] + self.wheel_speed_error())
      self.wheel_speeds = np.where(wheel_speeds >= p.wheel_speed_cutoff,
                                   np.round(wheel_speeds / (4. * p.speed_quantum)) * (4. * p.speed_quantum), 0.)
      self.raw_speed = float(np.mean(self.wheel_speeds))
    # Replay the production CR-V observer with sample-and-held wheel CAN
    # channels. In particular, a zero wheel channel is censored motion below
    # its observed threshold, not a zero vehicle speed.
    (self.measured_speed, self.measured_acceleration, _, self.measured_speed_std,
     self.measured_acceleration_std) = self.sensor.update_crv_speed_observer(self.wheel_speeds, 0.0)
    self.wheel_count = self.sensor.crv_wheel_count
    self.speed_dropout_time = self.sensor.crv_speed_dropout_time
    self.speed_measurement_valid = self.sensor.crv_speed_measurement_valid
    self.vehicle_sensors_invalid = self.sensor.crv_speed_dropout_timeout
