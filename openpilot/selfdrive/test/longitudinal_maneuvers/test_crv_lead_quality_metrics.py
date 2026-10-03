"""Keep physical velocity matching distinct from Honda pedal handoff."""
import math
from types import SimpleNamespace

import numpy as np

from opendbc.car.honda.values import CAR
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel

from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import (
  _moving_lead_maneuver_quality,
)


def lead_rows(speeds, modes):
  rows = np.zeros((len(speeds), 17), dtype=object)
  rows[:, 0] = np.arange(len(speeds)) + 16.0
  rows[:, 1] = speeds
  rows[:, 3] = [2.5, 2.3, 2.05, 2.0, 2.0][:len(speeds)]
  rows[:, 5] = 15.0
  rows[:, 10] = modes
  return rows


def test_gas_handoff_does_not_end_physical_deceleration():
  rows = lead_rows([20.0, 18.0, 15.9, 15.0, 14.5], ["brake", "gas", "gas", "gas", "gas"])
  phases, undershoot, match_gap = _moving_lead_maneuver_quality(rows)
  assert phases == ["brake", "gas"]
  assert undershoot == 0.5
  assert match_gap == 2.05


def test_missing_velocity_match_is_not_a_successful_handoff():
  rows = lead_rows([20.0, 18.0], ["brake", "gas"])
  _, _, match_gap = _moving_lead_maneuver_quality(rows)
  assert math.isinf(match_gap)


def test_velocity_undershoot_is_measured_even_without_brake_command():
  rows = lead_rows([15.0, 13.0], ["gas", "gas"])
  _, undershoot, _ = _moving_lead_maneuver_quality(rows)
  assert undershoot == 2.0


def test_crv_acceleration_change_does_not_retune_other_cars():
  cp = SimpleNamespace(carFingerprint="other", steerRatio=15.0, wheelbase=2.7)
  assert get_cruise_accel(False, 11.0, 10.0, 0.0, 0.0, cp, 1.0, -0.3, 1.0) == 0.12
  cp.carFingerprint = CAR.HONDA_CRV_5G
  assert get_cruise_accel(False, 11.0, 10.0, 0.0, 0.0, cp, 1.0, -0.3, 1.0) == 0.35
