"""Separate tracker acceleration uncertainty from physical lead motion."""
import math

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


@pytest.mark.parametrize("amplitude", (0.5, 1.0))
def test_stationary_follow_reference_rejects_acceleration_only_tracker_disturbance(amplitude):
  # A derivative-state disturbance must not masquerade as physical lead motion.
  # Both leads describe the same constant-speed car; only aLeadK is disturbed.
  speed = 20.0
  plant = Plant(lead_relevancy=True, speed=speed, distance_lead=2.05 * speed,
                physics=True, realtime=False, car_fingerprint=CAR.HONDA_CRV_5G,
                personality=log.LongitudinalPersonality.relaxed)
  rows = []
  while plant.current_time < 195.0:
    lead = {"a_lead": amplitude * math.sin(2.0 * math.pi * 0.2 * plant.current_time)}
    plant.step(v_lead=speed, v_cruise=speed, lead_one=lead, lead_two=lead)
    rows.append((plant.current_time, plant.speed, plant.gas_command, plant.brake_intensity))
  rows = np.asarray(rows)
  tail = rows[rows[:, 0] >= rows[-1, 0] - 60.0]
  assert np.max(np.abs(tail[:, 1] - speed)) < 0.44704
  assert np.ptp(tail[:, 2]) < 0.05, f"settled gas span={np.ptp(tail[:, 2])}"
  assert np.ptp(tail[:, 3]) < 0.05, f"settled brake span={np.ptp(tail[:, 3])}"
