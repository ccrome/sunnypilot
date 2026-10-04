"""Explicit uncertainty probes; these are not calibrated Honda parameters.

Keep the same one-second moving-gap floor, even when normal nominal cases
pass. Maximum-command counterfactuals distinguish unavailable authority from
avoidable delay in the controller trajectory. No production tuning is changed.
"""
import math
import json

import numpy as np

from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics
from openpilot.selfdrive.test.longitudinal_maneuvers import test_crv_longitudinal_quality_regression as q


PROBES = (
  ("slow_moving", 45, 40, False, HondaDynamics(delay=.8, gas_tau=1.5, brake_tau=.7)),
  ("slow_stopped", 25, 25, True, HondaDynamics(delay=.8, gas_tau=1.5, brake_tau=.7)),
  ("weak_moving", 45, 40, False, HondaDynamics(gas_gain=1.3, brake_gain=.8, gas_tau=.8,
                                             gas_release_tau=.2, delay=.2, speed_noise_std=.01, seed=1)),
)


def immediate_maximum_brake_gap(speed, lead_speed, gap, acceleration, parameters):
  """Optimistic physical reference, not a control law or pass-gate exemption.

  Independent 100 Hz integration commands maximum brake immediately, respects
  transport and buildup, and stops at speed match. Starting net acceleration
  initializes drive effort. No jerk/comfort constraint or perception delay is
  imposed: this is deliberately more aggressive than an occupant controller.
  """
  dt = .01
  drive = max(0., acceleration + parameters.rolling + parameters.drag * speed ** 2)
  brake = 0.
  minimum = gap / max(speed, .1)
  for tick in range(6000):
    committed = tick * dt >= parameters.delay
    drive_target = 0. if committed else drive
    drive += (1. - math.exp(-dt / parameters.gas_release_tau)) * (drive_target - drive)
    brake += (1. - math.exp(-dt / parameters.brake_tau)) * (
      (-3.5 * parameters.brake_gain if committed else 0.) - brake)
    previous = speed
    speed = max(0., speed + dt * (drive + brake - parameters.rolling - parameters.drag * speed ** 2))
    gap += (lead_speed - .5 * (previous + speed)) * dt
    minimum = min(minimum, gap / max(speed, .1))
    if speed <= lead_speed:
      return minimum
  raise AssertionError("Maximum braking did not match lead speed")


def _run_probe(label, ego, closing, stopped, parameters):
  return label, q._run_lead_case(ego, closing, stopped, actuator_parameters=parameters)


def test_full_system_preserves_gap_with_actuator_uncertainty():
  failures = []
  for probe, result in zip(PROBES, q._parallel_runs(_run_probe, PROBES), strict=True):
    label, rows = result
    parameters = probe[-1]
    t, speed, gap = (rows[:, i].astype(float) for i in (0, 1, 2))
    moving = (t >= q.LEAD_IN_S) & (speed * q.MPH > 1.)
    minimum = float(np.min(rows[moving, 3].astype(float)))
    onset = int(np.flatnonzero(t >= q.LEAD_IN_S)[0])
    # The last lead-in row precedes lead acquisition; use the requested lead.
    lead_speed = 0. if probe[3] else max(0., probe[1] - probe[2]) * q.MPH
    optimistic = immediate_maximum_brake_gap(speed[onset] * q.MPH, lead_speed,
                                           gap[onset], float(rows[onset, 4]), parameters)
    if minimum < 1. or np.min(gap) < 0.:
      failures.append({"case": label, "minimum_time_gap_s": minimum,
                       "minimum_clearance_m": float(np.min(gap)),
                       "immediate_maximum_brake_time_gap_s": optimistic,
                       "transport_s": parameters.delay, "brake_response_s": parameters.brake_tau})
  assert not failures, json.dumps(failures, indent=2)


def test_maximum_brake_reference_reports_transport_penalty():
  nominal = immediate_maximum_brake_gap(20., 2., 80., 0., HondaDynamics())
  delayed = immediate_maximum_brake_gap(20., 2., 80., 0., HondaDynamics(delay=.8, brake_tau=.7))
  assert nominal > delayed > 1.
