"""Regression coverage for CR-V tracked-lead surging.

The lead remains continuously tracked throughout these maneuvers.  The test
measures the simulated vehicle response during settled following, rather than
using gas/brake mode transitions as the definition of a surge.
"""

from __future__ import annotations

import math

import numpy as np

from opendbc.car.honda.values import CAR
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


SIMULATION_DURATION = 70.0
INITIAL_GAP = 80.0
LEAD_PROBABILITY = 1.0
GRADE_3_PERCENT = math.atan(0.03)

# A settled acceleration excursion larger than this is the test's definition
# of a surge.  This is deliberately independent of command-mode transitions.
MAX_SETTLED_ACCEL_PEAK_TO_TROUGH = 0.5
# The plant's lead-following response can retain a bounded speed offset while
# settling.  This is a sanity bound; the acceleration excursion is the L11
# surge gate.
MAX_SETTLED_SPEED_ERROR_PEAK_TO_TROUGH = 3.5
MIN_SETTLED_GAP = 5.0

# The lead-speed schedule has two gentle slowdown/recovery cycles.  The
# intervals below exclude the intentional lead transitions and keep only the
# settled portions for the acceptance metrics.
SETTLED_LEAD_INTERVALS = ((7.0, 10.0), (20.0, 25.0), (35.0, 40.0),
                          (50.0, 55.0), (65.0, 70.0))


def lead_speed_schedule(base_speed: float, time_s: float) -> float:
  if time_s < 10.0:
    return base_speed
  if time_s < 15.0:
    return base_speed - 1.5 * (time_s - 10.0) / 5.0
  if time_s < 25.0:
    return base_speed - 1.5
  if time_s < 30.0:
    return base_speed - 1.5 + 1.5 * (time_s - 25.0) / 5.0
  if time_s < 40.0:
    return base_speed
  if time_s < 45.0:
    return base_speed - 1.5 * (time_s - 40.0) / 5.0
  if time_s < 55.0:
    return base_speed - 1.5
  if time_s < 60.0:
    return base_speed - 1.5 + 1.5 * (time_s - 55.0) / 5.0
  return base_speed


def rolling_grade(time_s: float) -> float:
  if time_s < 15.0 or time_s >= 45.0:
    return 0.0
  if time_s < 25.0:
    return GRADE_3_PERCENT
  if time_s < 35.0:
    return 0.0
  return -GRADE_3_PERCENT


def settled_samples(rows: np.ndarray, grade_profile: str) -> np.ndarray:
  mask = np.zeros(len(rows), dtype=bool)
  for start, end in SETTLED_LEAD_INTERVALS:
    mask |= (rows[:, 0] >= start) & (rows[:, 0] < end)

  # Grade transitions are excluded from the acceptance windows.  Constant
  # grade cases have no transition to exclude.
  if grade_profile == "rolling":
    for transition in (15.0, 25.0, 35.0, 45.0):
      mask &= np.abs(rows[:, 0] - transition) > 5.0
  return rows[mask]


def run_maneuver(initial_speed: float, grade_profile: str) -> np.ndarray:
  plant = Plant(
    lead_relevancy=True,
    speed=initial_speed,
    distance_lead=INITIAL_GAP,
    e2e=False,
    car_fingerprint=CAR.HONDA_CRV_5G,
  )

  rows = []
  while plant.current_time < SIMULATION_DURATION:
    time_s = plant.current_time
    if grade_profile == "uphill":
      pitch = GRADE_3_PERCENT
    elif grade_profile == "downhill":
      pitch = -GRADE_3_PERCENT
    elif grade_profile == "rolling":
      pitch = rolling_grade(time_s)
    else:
      pitch = 0.0

    v_lead = lead_speed_schedule(initial_speed, time_s)
    plant.step(
      v_lead=v_lead,
      prob_lead=LEAD_PROBABILITY,
      v_cruise=initial_speed * 3.6,
      pitch=pitch,
    )
    rows.append((
      time_s,
      plant.speed,
      plant.acceleration,
      v_lead,
      plant.distance_lead - plant.distance,
      plant.planner.output_a_target,
    ))

  return np.asarray(rows)


class TestCrvTrackedLeadSurgingRegression(OpenpilotTestCase):
  def test_tracked_lead_following_settles_without_surging(self):
    """Tracked-lead following must settle across speed and grade conditions."""
    failures = []
    for initial_speed in (18.0, 24.0, 29.0, 33.0):
      for grade_profile in ("flat", "uphill", "downhill", "rolling"):
        rows = run_maneuver(initial_speed, grade_profile)
        settled = settled_samples(rows, grade_profile)
        speed_error = settled[:, 1] - settled[:, 3]
        settled_jerk = np.gradient(settled[:, 2], settled[:, 0])
        accel_peak_to_trough = float(np.ptp(settled[:, 2]))
        speed_error_peak_to_trough = float(np.ptp(speed_error))
        minimum_gap = float(np.min(rows[:, 4]))

        if minimum_gap <= MIN_SETTLED_GAP \
            or accel_peak_to_trough > MAX_SETTLED_ACCEL_PEAK_TO_TROUGH \
            or speed_error_peak_to_trough > MAX_SETTLED_SPEED_ERROR_PEAK_TO_TROUGH:
          failures.append({
            "speed_mps": initial_speed,
            "grade": grade_profile,
            "minimum_gap_m": minimum_gap,
            "accel_peak_to_trough": accel_peak_to_trough,
            "speed_error_peak_to_trough": speed_error_peak_to_trough,
            "jerk_p95_abs": float(np.percentile(np.abs(settled_jerk), 95)),
          })

    assert not failures, f"tracked-lead surging cases: {failures}"
