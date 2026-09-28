"""Closed-loop CR-V longitudinal quality gates.

These tests deliberately exercise the planner through the Plant rather than
testing isolated tuning constants.  The optional Plant physics model keeps the
existing maneuver tests deterministic and backward compatible while making
these checks sensitive to road load and Honda actuator handoff behavior.
"""

from __future__ import annotations

import math

import numpy as np

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from opendbc.car.common.conversions import Conversions as CV
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


MPH = CV.MPH_TO_MS
SIM_RATE = 20.0
DT = 1.0 / SIM_RATE
TARGET_GAP = 2.05


def _run_speed_transition(start_mph: float, target_mph: float) -> np.ndarray:
  plant = Plant(speed=start_mph * MPH, physics=True, realtime=False,
                car_fingerprint=CAR.HONDA_CRV_5G, sim_rate=SIM_RATE)
  duration = max(30.0, 15.0 + abs(target_mph - start_mph) * 0.40)
  rows = []
  while plant.current_time < duration:
    plant.step(v_cruise=target_mph * MPH)
    rows.append((plant.current_time, plant.speed / MPH, plant.acceleration,
                 plant.gas_command, float(plant.brake_request), plant.brake_intensity,
                 plant.actuator_mode, plant.mode_transitions, plant.planner_acceleration))
  return np.asarray(rows, dtype=object)


def _crossing_time(rows: np.ndarray, target_mph: float, start_mph: float) -> float:
  direction = np.sign(target_mph - start_mph)
  crossed = np.flatnonzero(direction * (rows[:, 1].astype(float) - target_mph) >= 0.0)
  return float(rows[crossed[0], 0]) if len(crossed) else math.inf


def _run_cruise(target_mph: float, grade_percent: int) -> np.ndarray:
  plant = Plant(speed=0.0, physics=True, realtime=False,
                car_fingerprint=CAR.HONDA_CRV_5G, sim_rate=SIM_RATE)
  duration = 75.0
  rows = []
  pitch = math.atan(grade_percent / 100.0)
  while plant.current_time < duration:
    plant.step(v_cruise=target_mph * MPH, pitch=pitch)
    rows.append((plant.current_time, plant.speed / MPH, grade_percent,
                 plant.gas_command, plant.brake_intensity, plant.brake_request,
                 plant.actuator_mode, plant.mode_transitions, plant.acceleration,
                 plant.planner_acceleration))
  return np.asarray(rows, dtype=object)


def _p95_jerk(rows: np.ndarray) -> float:
  return float(np.percentile(np.abs(np.gradient(rows[:, 8].astype(float), DT)), 95))


def _run_lead_case(ego_mph: float, closing_mph: float, stopped: bool) -> np.ndarray:
  ego = ego_mph * MPH
  closing = closing_mph * MPH
  plant = Plant(lead_relevancy=True, speed=ego,
                distance_lead=max(2.5 * ego, 2.0), physics=True, realtime=False,
                personality=log.LongitudinalPersonality.relaxed,
                car_fingerprint=CAR.HONDA_CRV_5G, sim_rate=SIM_RATE)
  rows = []
  while plant.current_time < 38.0:
    approaching = plant.current_time >= 3.0
    lead_speed = 0.0 if stopped and approaching else max(0.0, ego - closing) if approaching else ego
    plant.step(v_lead=lead_speed, prob_lead=1.0, v_cruise=ego)
    gap = max(0.0, plant.distance_lead - plant.distance)
    time_gap = gap / max(plant.speed, 0.1)
    rows.append((plant.current_time, plant.speed / MPH, gap, time_gap,
                 plant.acceleration, lead_speed / MPH, plant.planner_acceleration,
                 plant.gas_command, plant.brake_intensity, plant.brake_request,
                 plant.actuator_mode, plant.mode_transitions))
  return np.asarray(rows, dtype=object)


class TestCrvLongitudinalQualityRegression(OpenpilotTestCase):
  def test_speed_transitions_have_bounded_crossing_error(self):
    failures = []
    for target in (15, 25, 35, 45, 55, 65, 75, 85, 90):
      rows = _run_speed_transition(0, target)
      speed = rows[:, 1].astype(float)
      crossing = _crossing_time(rows, target, 0)
      peak = float(np.max(speed))
      if not math.isfinite(crossing) or peak - target > 1.0:
        failures.append({"direction": "up", "target_mph": target, "peak_mph": peak,
                         "crossing_s": crossing, "reached_target": math.isfinite(crossing),
                         "mode": rows[-1, 6],
                         "transitions": int(rows[-1, 7])})

    for target in (65, 55, 45, 35, 25, 15, 5, 0):
      rows = _run_speed_transition(90, target)
      speed = rows[:, 1].astype(float)
      crossing = _crossing_time(rows, target, 90)
      trough = float(np.min(speed))
      if not math.isfinite(crossing) or target - trough > 1.0:
        failures.append({"direction": "down", "target_mph": target, "trough_mph": trough,
                         "crossing_s": crossing, "reached_target": math.isfinite(crossing),
                         "mode": rows[-1, 6],
                         "transitions": int(rows[-1, 7])})
    assert not failures, f"CR-V speed-transition failures: {failures}"

  def test_steady_cruise_grade_matrix_is_settled_and_stable(self):
    failures = []
    for target in range(10, 91, 10):
      for grade in range(-15, 16, 3):
        rows = _run_cruise(target, grade)
        settled = rows[rows[:, 0].astype(float) >= 25.0]
        speed = settled[:, 1].astype(float)
        gas = settled[:, 3].astype(float)
        brake = settled[:, 4].astype(float)
        modes = settled[:, 6]
        speed_error = float(np.max(np.abs(speed - target)))
        gas_span = float(np.ptp(gas))
        brake_span = float(np.ptp(brake))
        mode_transitions = int(settled[-1, 7]) - int(settled[0, 7])
        # A mode can change once while a grade/load transition is absorbed;
        # repeated gas/brake alternation is the quality failure.
        if (speed_error > 1.0 or gas_span > 0.05 or brake_span > 0.05
            or mode_transitions > 2):
          failures.append({"speed_mph": target, "grade_percent": grade,
                           "settled_mode": str(modes[-1]), "speed_error_mph": speed_error,
                           "gas_span": gas_span, "brake_intensity_span": brake_span,
                           "transition_count": mode_transitions, "jerk_p95": _p95_jerk(settled)})
    assert not failures, f"CR-V steady-cruise failures: {failures}"

  def test_fast_closing_lead_recovers_without_late_gap_loss(self):
    failures = []
    for ego in (25, 45, 65, 85):
      for closing in (1, 3, 5, 10, 15, 20, 30, 40):
        for stopped in (False, True):
          rows = _run_lead_case(ego, closing, stopped)
          t = rows[:, 0].astype(float)
          gaps = rows[:, 3].astype(float)
          closest = int(np.argmin(gaps))
          recovery = np.flatnonzero((np.arange(len(rows)) > closest) &
                                    (gaps >= TARGET_GAP - 0.25))
          recovery_i = int(recovery[0]) if len(recovery) else len(rows)
          recovery_gap = float(gaps[recovery_i]) if recovery_i < len(rows) else math.inf
          later_min = float(np.min(gaps[recovery_i + 1:])) if recovery_i + 1 < len(rows) else recovery_gap
          recover_time = float(t[recovery_i] - t[closest]) if recovery_i < len(rows) else math.inf
          min_gap = float(gaps[closest])
          if min_gap < 1.0 or recover_time > 15.0 or later_min < recovery_gap - 1e-3:
            failures.append({"ego_mph": ego, "closing_mph": closing, "stopped": stopped,
                             "minimum_time_gap_s": min_gap, "recovery_s": recover_time,
                             "recovery_gap_s": recovery_gap, "later_min_gap_s": later_min,
                             "settled_target_s": TARGET_GAP})
    assert not failures, f"CR-V fast-closing lead failures: {failures}"
