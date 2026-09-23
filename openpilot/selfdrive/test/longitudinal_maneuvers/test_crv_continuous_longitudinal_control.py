"""Regression tests for continuous longitudinal candidate arbitration."""

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  get_throttle_authority, smooth_min,
)


class TestCrvContinuousLongitudinalControl(OpenpilotTestCase):
  def test_smooth_min_is_conservative_and_continuous(self):
    values = np.linspace(-0.2, 0.2, 41)
    outputs = np.array([smooth_min([value, -value]) for value in values])

    assert np.all(outputs <= np.minimum(values, -values))
    assert np.max(np.abs(np.diff(outputs))) < 0.03
    assert np.all(np.diff(outputs) <= 0.03)

  def test_throttle_authority_changes_continuously_with_probability(self):
    probabilities = np.linspace(0.0, 1.0, 101)
    authority = np.array([get_throttle_authority(probability, 15.0) for probability in probabilities])

    assert np.all(np.diff(authority) >= 0.0)
    assert np.max(np.diff(authority)) <= 0.051
    assert authority[0] == 0.0
    assert authority[-1] == 1.0

  def test_low_speed_has_full_throttle_authority(self):
    assert get_throttle_authority(0.0, 0.0) == 1.0
    assert get_throttle_authority(0.0, 2.5) == 1.0
    assert 0.0 < get_throttle_authority(0.0, 3.75) < 1.0
    assert get_throttle_authority(0.0, 5.0) == 0.0
