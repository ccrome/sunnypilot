from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics, HondaVehicle

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
  assert v.effort == v.drive_effort + v.brake_effort


def test_estimator_matches_production_for_identical_raw_samples():
  from opendbc.car.honda.carstate import CarState
  v = vehicle(HondaDynamics(speed_noise_std=0.0))
  reference = CarState(v.longitudinal.CP, v.longitudinal.CP_SP)
  reference.v_ego_kf.set_x([[10.], [0.]])
  for n in range(100):
    v.frame = n
    v.advance((1., 800., False), 0.)
    assert reference.update_speed_kf(v.raw_speed) == (v.measured_speed, v.measured_acceleration)


def test_default_parameters_have_independent_route_validation():
  report = json.loads(Path(__file__).with_name('honda_calibration.json').read_text())
  assert not set(report['train_routes']) & set(report['validation_routes'])
  assert asdict(HondaDynamics()) == report['simulation_dynamics']
  for route in report['validation_routes']:
    new = report['models']['gas_effort']['routes'][route]
    old = report['previous_model'][route]
    assert new['windows'] >= 100
    assert new['acceleration_rmse_mps2'] < old['acceleration_rmse_mps2'] * .6
    assert new['speed_rmse_mps'] < old['speed_rmse_mps'] * .6


def test_feedforward_matches_physical_force_units():
  from openpilot.selfdrive.controls.lib.longcontrol import (
    CRV_BRAKE_EFFORT_GAIN, CRV_BRAKE_RESPONSE_TIME, CRV_DRIVE_EFFORT_GAIN, CRV_ROLLING_ACCEL, CRV_DRAG_COEFFICIENT)
  p = HondaDynamics()
  assert CRV_DRIVE_EFFORT_GAIN == p.gas_gain * 2.2 / 2.
  assert CRV_BRAKE_EFFORT_GAIN == p.brake_gain
  assert CRV_BRAKE_RESPONSE_TIME == p.brake_tau
  assert CRV_ROLLING_ACCEL == p.rolling
  assert CRV_DRAG_COEFFICIENT == p.drag


def test_command_measurement_resolves_drift_without_counting_one_bit_noise():
  from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import (
    COMMAND_DERIVATIVE_P95_MAX, _p95_command_derivative)
  quantum = 1 / 1600
  one_bit = np.tile([0.01, 0.01 + quantum], 600)
  assert _p95_command_derivative(one_bit) == 0
  assert _p95_command_derivative(np.arange(1200) * quantum) > COMMAND_DERIVATIVE_P95_MAX
  assert _p95_command_derivative(np.tile([0.01, 0.02], 600)) > COMMAND_DERIVATIVE_P95_MAX
