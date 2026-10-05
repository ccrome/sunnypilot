from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics, HondaVehicle

import math
import numpy as np
import json
from dataclasses import asdict
from pathlib import Path


def vehicle(parameters=None):
  cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
  cp.openpilotLongitudinalControl = True
  return HondaVehicle(cp, sp, 10.0, parameters)


def test_transport_delay_and_response_are_causal():
  v = vehicle(HondaDynamics(delay=0.3))
  v.step(1.0, False, True, 0.0, 0.2, 15.0)
  assert v.command[1] > 0
  assert v.acceleration < 0  # Command has not yet reached the physical actuator.
  v.step(1.0, False, True, 0.0, 1.0, 15.0)
  assert v.acceleration > 0
  assert v.acceleration != 1.0


def test_braking_reaches_vehicle_through_encoded_can():
  v = vehicle()
  v.step(-1.0, False, True, 0.0, 1.0, 5.0)
  assert v.command[2]
  assert v.command[1] == 0
  assert v.speed < 10.0
  assert v.acceleration < -0.5


def test_downhill_can_hold_brake_with_a_positive_acceleration_request():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.step(0.18, False, True, math.atan(-.15), 1.0, 10.0)
  assert v.command[0] > 0.0
  assert v.command[1] == 0.0
  assert v.command[2]
  assert v.brake_effort < 0.0


def test_committed_stop_keeps_brake_requested_while_easing_pressure():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.step(0.04, False, True, 0.0, 0.1, 10.0, stop_phase=1)
  assert v.command[0] > 0.0
  assert v.command[1] == 0.0
  assert v.command[2]


def test_downhill_brake_request_can_ease_pressure_without_releasing_mode():
  level = vehicle(HondaDynamics(speed_noise_std=0.0))
  eased = vehicle(HondaDynamics(speed_noise_std=0.0))
  for n in range(200):
    for v, command in ((level, 0.0), (eased, 0.18)):
      v.frame = n
      v.advance((command, 0.0, True), math.atan(-.15))
  assert level.brake_effort < 0.0
  assert eased.brake_effort < 0.0
  assert 0.10 < eased.acceleration - level.acceleration < 0.25


def test_response_parameters_change_physical_response():
  fast = vehicle(HondaDynamics(gas_tau=0.2, delay=0.0))
  slow = vehicle(HondaDynamics(gas_tau=1.5, delay=0.4))
  for v in (fast, slow):
    v.step(1.0, False, True, 0.0, 0.5, 15.0)
  assert fast.speed > slow.speed


def test_disabled_controller_cannot_send_gas_or_brake():
  v = vehicle()
  v.step(2.0, False, False, 0.0, 0.5, 15.0)
  assert v.command == (0.0, 0.0, False)


def test_controller_receives_observed_not_perfect_acceleration():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.advance((1.0, 800., False), 0.)
  assert v.measured_acceleration != v.acceleration
  raw = v.raw_speed
  v.frame += 1
  v.advance((1.0, 800., False), 0.)
  assert v.raw_speed == raw  # 50 Hz sensor held through 100 Hz controller tick.
  assert v.speed != raw


def test_sensor_disturbance_is_deterministic_and_seeded():
  a, b = vehicle(), vehicle()
  c = vehicle(HondaDynamics(seed=1))
  for v in (a, b, c):
    v.step(.3, False, True, 0., 1., 15.)
  assert a.speed == b.speed
  assert a.measured_acceleration == b.measured_acceleration
  assert a.measured_acceleration != c.measured_acceleration


def test_drive_and_brake_efforts_do_not_reset_at_crossover():
  v = vehicle()
  for n in range(50):
    v.frame = n
    v.advance((1., 800., False), 0.)
  before = v.drive_effort
  v.advance((-1., 0., True), 0.)
  assert 0 < v.drive_effort < before
  assert v.brake_effort < 0
  assert v.effort == v.drive_effort + v.creep_effort + v.brake_effort


def test_observer_matches_production_for_identical_wheel_samples():
  from opendbc.car.honda.carstate import CarState
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  reference = CarState(v.longitudinal.CP, v.longitudinal.CP_SP)
  reference.v_ego_kf.set_x([[10.], [0.]])
  for n in range(100):
    v.frame = n
    v.advance((1., 800., False), 0.)
    estimate = reference.update_crv_speed_observer(v.wheel_speeds, 0.)
    assert estimate[:2] == (v.measured_speed, v.measured_acceleration)


def test_default_parameters_have_independent_route_validation():
  report = json.loads(Path(__file__).with_name('honda_low_speed_calibration.json').read_text())
  assert not set(report['train_routes']) & set(report['validation_routes'])
  assert asdict(HondaDynamics()) == report['simulation_dynamics']
  for route in report['validation_routes']:
    result = report['routes'][route]
    assert result['new']['stops'] >= 3
    assert result['new']['speed_rmse_mps'] < result['old']['speed_rmse_mps']
    for metric in ('speed_rmse_mps', 'acceleration_rmse_mps2'):
      assert result['higher_speed']['new'][metric] < result['higher_speed']['old'][metric]


def test_legacy_calibration_provenance_is_preserved():
  report = json.loads(Path(__file__).with_name('honda_calibration.json').read_text())
  assert not set(report['train_routes']) & set(report['validation_routes'])
  for route in report['validation_routes']:
    new = report['models']['gas_effort']['routes'][route]
    old = report['previous_model'][route]
    assert new['windows'] >= 100
    assert new['acceleration_rmse_mps2'] < old['acceleration_rmse_mps2'] * .6
    assert new['speed_rmse_mps'] < old['speed_rmse_mps'] * .6


def test_plant_calibration_does_not_retarget_production_feedforward():
  from openpilot.selfdrive.controls.lib.longcontrol import (
    CRV_BRAKE_EFFORT_GAIN, CRV_BRAKE_RESPONSE_TIME, CRV_DRIVE_EFFORT_GAIN, CRV_ROLLING_ACCEL, CRV_DRAG_COEFFICIENT)
  p = HondaDynamics()
  assert CRV_DRIVE_EFFORT_GAIN == p.gas_gain * 2.2 / 2.
  legacy = json.loads(Path(__file__).with_name('honda_calibration.json').read_text())['simulation_dynamics']
  assert CRV_BRAKE_EFFORT_GAIN == legacy['brake_gain']
  assert CRV_BRAKE_EFFORT_GAIN != p.brake_gain  # Independent identification, not controller tuning.
  assert CRV_BRAKE_RESPONSE_TIME == p.brake_tau
  assert CRV_ROLLING_ACCEL == p.rolling
  assert CRV_DRAG_COEFFICIENT == p.drag


def test_wheel_zero_is_not_a_physical_stop():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.speed = .5
  v.advance((0., 0., False), 0.)
  assert v.raw_speed == 0.
  assert v.speed > .49


def test_wheel_channels_disappear_separately():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.speed = v.parameters.wheel_speed_cutoff
  v.advance((0., 0., False), 0.)
  assert 0. < v.raw_speed < v.speed * .8


def test_small_recorded_gas_command_can_restart_creep():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.speed = .7
  initial = v.speed
  for n in range(50):
    v.frame = n
    v.advance((0., 13., False), 0.)  # Latest-drive final gas was only 13 CAN units.
  assert v.speed > initial
  assert v.acceleration > .1
  assert v.creep_effort > v.drive_effort


def test_low_speed_brake_pressure_is_causal_and_releases_without_reset():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.speed = .7
  v.advance((-1., 0., True), 0.)
  assert -1. < v.brake_demand < v.brake_effort < 0.
  for n in range(1, 50):
    v.frame = n
    v.advance((-1., 0., True), 0.)
  before = v.brake_effort
  demand_before = v.brake_demand
  v.advance((0., 13., False), 0.)
  assert demand_before < v.brake_demand < 0.
  assert v.brake_effort < 0. and abs(v.brake_effort - before) < .02
  # Force can continue building briefly after demand starts releasing.
  for n in range(50, 250):
    v.frame = n
    v.advance((0., 13., False), 0.)
  assert before < v.brake_effort < 0.


def test_hold_command_can_stop_on_a_downhill_without_teleporting_speed():
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  v.speed = .3
  v.advance((0., 0., True), -.03, standstill=True)
  assert v.speed > .29  # Receiving ECU has finite pressure/force response.
  for n in range(1, 500):
    v.frame = n
    v.advance((0., 0., True), -.03, standstill=True)
  assert v.speed == 0.
  assert v.raw_speed == 0.


def test_standstill_request_is_decoded_from_production_can():
  v = vehicle()
  v.speed = v.measured_speed = 0.
  v.sensor.v_ego_kf.set_x([[0.], [0.]])
  v.step(-1., True, True, 0., 1., 0.)
  assert v.command[2]
  assert v.standstill_command
  assert v.speed == 0.


def test_terminal_feedforward_cannot_override_emergency_can_braking():
  v = vehicle()
  v.longitudinal.pid.i = .5
  v.step(-3.5, False, True, 0., .1, 0., stop_phase=1,
         stop_reference=(1., -1., 20.), safety_pressure=1.)
  assert v.command == (-3.5, 0., True)


def test_negative_can_request_is_net_deceleration_on_moderate_grades():
  for pitch in (-.03, 0., .03):
    v = vehicle(HondaDynamics(speed_noise_std=0.0))
    v.speed = 20.
    for n in range(500):
      v.frame = n
      v.advance((-.8, 0., True), pitch)
    assert abs(v.acceleration - (-.8 * v.parameters.brake_gain)) < .02


def test_command_measurement_resolves_drift_without_counting_one_bit_noise():
  from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import (
    COMMAND_DERIVATIVE_P95_MAX, _p95_command_derivative)
  quantum = 1 / 1600
  one_bit = np.tile([0.01, 0.01 + quantum], 600)
  assert _p95_command_derivative(one_bit) == 0
  assert _p95_command_derivative(np.arange(1200) * quantum) > COMMAND_DERIVATIVE_P95_MAX
  assert _p95_command_derivative(np.tile([0.01, 0.02], 600)) > COMMAND_DERIVATIVE_P95_MAX
