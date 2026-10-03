from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics, HondaVehicle


def vehicle(parameters=None):
  cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
  cp.openpilotLongitudinalControl = True
  return HondaVehicle(cp, sp, 10.0, parameters)


def test_transport_delay_and_response_are_causal():
  v = vehicle()
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


def test_command_measurement_resolves_drift_without_counting_one_bit_noise():
  import numpy as np
  from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import (
    COMMAND_DERIVATIVE_P95_MAX, _p95_command_derivative)
  quantum = 1 / 1600
  one_bit = np.tile([0.01, 0.01 + quantum], 600)
  assert _p95_command_derivative(one_bit) == 0
  assert _p95_command_derivative(np.arange(1200) * quantum) > COMMAND_DERIVATIVE_P95_MAX
  assert _p95_command_derivative(np.tile([0.01, 0.02], 600)) > COMMAND_DERIVATIVE_P95_MAX
