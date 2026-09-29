"""Closed-loop CR-V longitudinal quality gates.

These tests deliberately exercise the planner through the Plant rather than
testing isolated tuning constants.  The optional Plant physics model keeps the
existing maneuver tests deterministic and backward compatible while making
these checks sensitive to road load and Honda actuator handoff behavior.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from opendbc.car.honda.values import CAR
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.cereal import log
from opendbc.car.common.conversions import Conversions as CV
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MIN, get_max_accel
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


MPH = CV.MPH_TO_MS
SIM_RATE = 20.0
DT = 1.0 / SIM_RATE
TARGET_GAP = 2.05
POST_TARGET_SETTLE_S = 60.0
RUN_DURATION_S = 180.0
LEAD_IN_S = 15.0
TOTAL_RUN_DURATION_S = LEAD_IN_S + RUN_DURATION_S
COMMAND_DERIVATIVE_P95_MAX = 0.001
GRADE_FEASIBILITY_MARGIN = 0.1


def _run_speed_transition(start_mph: float, target_mph: float) -> np.ndarray:
  plant = Plant(speed=start_mph * MPH, physics=True, realtime=False,
                car_fingerprint=CAR.HONDA_CRV_5G, sim_rate=SIM_RATE)
  # Keep the trace alive well beyond the crossing.  The quality gate must see
  # the complete post-transition behavior, not just the instant the target is
  # first reached.
  duration = LEAD_IN_S + max(RUN_DURATION_S, abs(target_mph - start_mph) * 0.8 + POST_TARGET_SETTLE_S)
  rows = []
  while plant.current_time < duration:
    v_cruise = start_mph * MPH if plant.current_time < LEAD_IN_S else target_mph * MPH
    plant.step(v_cruise=v_cruise)
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
  # The cruise matrix also needs a full post-target stability window.  The
  # longest acceleration in this matrix reaches target before this budget.
  duration = TOTAL_RUN_DURATION_S
  rows = []
  pitch = math.atan(grade_percent / 100.0)
  while plant.current_time < duration:
    v_cruise = 0.0 if plant.current_time < LEAD_IN_S else target_mph * MPH
    plant.step(v_cruise=v_cruise, pitch=pitch)
    rows.append((plant.current_time, plant.speed / MPH, grade_percent,
                 plant.gas_command, plant.brake_intensity, plant.brake_request,
                 plant.actuator_mode, plant.mode_transitions, plant.acceleration,
                 plant.planner_acceleration))
  return np.asarray(rows, dtype=object)


def _p95_jerk(rows: np.ndarray) -> float:
  return float(np.percentile(np.abs(np.gradient(rows[:, 8].astype(float), DT)), 95))


def _p95_command_derivative(values: np.ndarray) -> float:
  return float(np.percentile(np.abs(np.gradient(values.astype(float), DT)), 95))


def _run_lead_case(ego_mph: float, closing_mph: float, stopped: bool) -> np.ndarray:
  ego = ego_mph * MPH
  closing = closing_mph * MPH
  lead_speed_at_maneuver = max(0.0, ego - closing)
  target_gap_at_maneuver = max(2.0, lead_speed_at_maneuver * TARGET_GAP)
  braking_distance = closing * closing / (2.0 * abs(ACCEL_MIN))
  feasible_start_gap = braking_distance + target_gap_at_maneuver + 1.0
  initial_gap = max(2.5 * ego, feasible_start_gap) if not stopped and closing < ego else max(2.5 * ego, 2.0)
  plant = Plant(lead_relevancy=True, speed=ego,
                distance_lead=initial_gap, physics=True, realtime=False,
                personality=log.LongitudinalPersonality.relaxed,
                car_fingerprint=CAR.HONDA_CRV_5G, sim_rate=SIM_RATE)
  rows = []
  maneuver_lead_speed = None
  while plant.current_time < TOTAL_RUN_DURATION_S:
    approaching = plant.current_time >= LEAD_IN_S
    stopped_lead = stopped or closing >= ego
    if approaching and maneuver_lead_speed is None:
      maneuver_lead_speed = 0.0 if stopped_lead else max(0.0, plant.speed - closing)
    lead_speed = maneuver_lead_speed if approaching else plant.speed
    plant.step(v_lead=lead_speed, prob_lead=1.0 if approaching else 0.0, v_cruise=ego)
    gap = max(0.0, plant.distance_lead - plant.distance)
    time_gap = gap / max(plant.speed, 0.1)
    rows.append((plant.current_time, plant.speed / MPH, gap, time_gap,
                 plant.acceleration, lead_speed / MPH, plant.planner_acceleration,
                 plant.gas_command, plant.brake_intensity, plant.brake_request,
                 plant.actuator_mode, plant.mode_transitions,
                 plant.predictive_brake, plant.safety_override))
  return np.asarray(rows, dtype=object)


def _parallel_runs(function, cases):
  with ProcessPoolExecutor(max_workers=min(32, len(cases))) as executor:
    return list(executor.map(function, *zip(*cases, strict=True)))


def _run_speed_case(start_mph, target_mph):
  return _run_speed_transition(start_mph, target_mph)


def _run_cruise_case(target_mph, grade_percent):
  return _run_cruise(target_mph, grade_percent)


def _run_lead_case_spec(ego_mph, closing_mph, stopped):
  return _run_lead_case(ego_mph, closing_mph, stopped)


def _lead_case_physically_feasible(ego_mph: float, closing_mph: float, stopped: bool) -> bool:
  ego_speed = ego_mph * MPH
  lead_speed = 0.0 if stopped or closing_mph >= ego_mph else (ego_mph - closing_mph) * MPH
  relative_speed = max(ego_speed - lead_speed, 0.0)
  braking_distance = relative_speed ** 2 / (2.0 * abs(ACCEL_MIN))
  target_gap = max(2.0, lead_speed * TARGET_GAP)
  initial_gap = max(2.5 * ego_speed, braking_distance + target_gap + 1.0) \
    if not stopped and closing_mph < ego_mph else max(2.5 * ego_speed, 2.0)
  required_gap = braking_distance + target_gap + 1.0
  return initial_gap >= required_gap


class TestCrvLongitudinalQualityRegression(OpenpilotTestCase):
  def test_speed_transitions_have_bounded_crossing_error(self):
    failures = []
    up_cases = [(0, target) for target in (15, 25, 35, 45, 55, 65, 75, 85, 90)]
    down_cases = [(90, target) for target in (65, 55, 45, 35, 25, 15, 5, 0)]
    for (_start, target), rows in zip(up_cases, _parallel_runs(_run_speed_case, up_cases), strict=True):
      speed = rows[:, 1].astype(float)
      crossing = _crossing_time(rows, target, 0)
      peak = float(np.max(speed))
      settled = rows[rows[:, 0].astype(float) >= crossing] if math.isfinite(crossing) else rows[:0]
      settled_duration = float(settled[-1, 0] - settled[0, 0]) if len(settled) else 0.0
      settled_error = float(np.max(np.abs(settled[:, 1].astype(float) - target))) if len(settled) else math.inf
      stable = rows[rows[:, 0].astype(float) >= float(rows[-1, 0]) - POST_TARGET_SETTLE_S]
      gas_d_p95 = _p95_command_derivative(stable[:, 3])
      brake_d_p95 = _p95_command_derivative(stable[:, 5])
      if (not math.isfinite(crossing) or peak - target > 1.0
          or settled_duration < POST_TARGET_SETTLE_S or settled_error > 1.0
          or gas_d_p95 > COMMAND_DERIVATIVE_P95_MAX or brake_d_p95 > COMMAND_DERIVATIVE_P95_MAX):
        failures.append({"direction": "up", "target_mph": target, "peak_mph": peak,
                         "crossing_s": crossing, "reached_target": math.isfinite(crossing),
                         "settled_duration_s": settled_duration, "settled_error_mph": settled_error,
                         "gas_derivative_p95": gas_d_p95, "brake_derivative_p95": brake_d_p95,
                         "mode": rows[-1, 6],
                         "transitions": int(rows[-1, 7])})

    for (_start, target), rows in zip(down_cases, _parallel_runs(_run_speed_case, down_cases), strict=True):
      speed = rows[:, 1].astype(float)
      crossing = _crossing_time(rows, target, 90)
      trough = float(np.min(speed))
      settled = rows[rows[:, 0].astype(float) >= crossing] if math.isfinite(crossing) else rows[:0]
      settled_duration = float(settled[-1, 0] - settled[0, 0]) if len(settled) else 0.0
      settled_error = float(np.max(np.abs(settled[:, 1].astype(float) - target))) if len(settled) else math.inf
      stable = rows[rows[:, 0].astype(float) >= float(rows[-1, 0]) - POST_TARGET_SETTLE_S]
      gas_d_p95 = _p95_command_derivative(stable[:, 3])
      brake_d_p95 = _p95_command_derivative(stable[:, 5])
      if (not math.isfinite(crossing) or target - trough > 1.0
          or settled_duration < POST_TARGET_SETTLE_S or settled_error > 1.0
          or gas_d_p95 > COMMAND_DERIVATIVE_P95_MAX or brake_d_p95 > COMMAND_DERIVATIVE_P95_MAX):
        failures.append({"direction": "down", "target_mph": target, "trough_mph": trough,
                         "crossing_s": crossing, "reached_target": math.isfinite(crossing),
                         "settled_duration_s": settled_duration, "settled_error_mph": settled_error,
                         "gas_derivative_p95": gas_d_p95, "brake_derivative_p95": brake_d_p95,
                         "mode": rows[-1, 6],
                         "transitions": int(rows[-1, 7])})
    assert not failures, f"CR-V speed-transition failures: {failures}"

  def test_steady_cruise_grade_matrix_is_settled_and_stable(self):
    failures = []
    cases = [(target, grade) for target in range(10, 91, 10) for grade in range(-15, 16, 3)]
    for (target, grade), rows in zip(cases, _parallel_runs(_run_cruise_case, cases), strict=True):
        settled = rows[rows[:, 0].astype(float) >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
        speed = settled[:, 1].astype(float)
        gas = settled[:, 3].astype(float)
        brake = settled[:, 4].astype(float)
        modes = settled[:, 6]
        speed_error = float(np.max(np.abs(speed - target)))
        required_accel = 9.81 * math.sin(math.atan(grade / 100.0)) + 0.012
        feasible = (A_CRUISE_MIN + GRADE_FEASIBILITY_MARGIN <= required_accel
                    <= get_max_accel(target * MPH) - GRADE_FEASIBILITY_MARGIN)
        gas_span = float(np.ptp(gas))
        brake_span = float(np.ptp(brake))
        gas_d_p95 = _p95_command_derivative(settled[:, 3])
        brake_d_p95 = _p95_command_derivative(settled[:, 4])
        mode_transitions = int(settled[-1, 7]) - int(settled[0, 7])
        # A mode can change once while a grade/load transition is absorbed;
        # repeated gas/brake alternation is the quality failure.
        if ((feasible and (speed_error > 1.0 or gas_span > 0.05 or brake_span > 0.05
                           or gas_d_p95 > COMMAND_DERIVATIVE_P95_MAX
                           or brake_d_p95 > COMMAND_DERIVATIVE_P95_MAX))
            or (not feasible and (gas_span > 0.10 or brake_span > 0.10))
            or mode_transitions > 2):
          failures.append({"speed_mph": target, "grade_percent": grade,
                           "settled_mode": str(modes[-1]), "speed_error_mph": speed_error,
                           "feasible": feasible, "required_accel_mps2": required_accel,
                           "gas_span": gas_span, "brake_intensity_span": brake_span,
                           "gas_derivative_p95": gas_d_p95, "brake_derivative_p95": brake_d_p95,
                           "transition_count": mode_transitions, "jerk_p95": _p95_jerk(settled)})
    assert not failures, f"CR-V steady-cruise failures: {failures}"

  def test_fast_closing_lead_recovers_without_late_gap_loss(self):
    failures = []
    cases = [(ego, closing, stopped)
             for ego in (25, 45, 65, 85)
             for closing in (1, 3, 5, 10, 15, 20, 30, 40)
             for stopped in (False, True)]
    for (ego, closing, stopped), rows in zip(cases, _parallel_runs(_run_lead_case_spec, cases), strict=True):
          stopped_case = stopped or closing >= ego
          t = rows[:, 0].astype(float)
          gaps = rows[:, 2 if stopped_case else 3].astype(float)
          closest = int(np.argmin(gaps))
          recovery = (np.flatnonzero((np.arange(len(rows)) > closest)
                                     & (gaps >= TARGET_GAP - 0.25))
                      if not stopped_case and float(np.min(gaps)) < TARGET_GAP - 0.25
                      else np.empty(0, dtype=int))
          recovery_i = int(recovery[0]) if len(recovery) else len(rows)
          recovery_gap = float(gaps[recovery_i]) if recovery_i < len(rows) else math.inf
          later_min = float(np.min(gaps[recovery_i + 1:])) if recovery_i + 1 < len(rows) else recovery_gap
          recover_time = float(t[recovery_i] - t[closest]) if recovery_i < len(rows) else math.inf
          min_gap = float(gaps[closest])
          tail_start = TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S
          tail = rows[t >= tail_start]
          tail_gaps = tail[:, 2 if stopped_case else 3].astype(float)
          tail_gap_span = float(np.ptp(tail_gaps)) if len(tail) else math.inf
          tail_gas_d_p95 = _p95_command_derivative(tail[:, 7]) if len(tail) else math.inf
          tail_brake_d_p95 = _p95_command_derivative(tail[:, 8]) if len(tail) else math.inf
          tail_mode_transitions = (int(tail[-1, 11]) - int(tail[0, 11])) if len(tail) else math.inf
          tail_target_error = (float(np.max(np.abs(tail_gaps - TARGET_GAP)))
                               if len(tail) and not stopped_case else 0.0 if len(tail) else math.inf)
          recovery_started = recovery_i < len(rows)
          physical_feasible = _lead_case_physically_feasible(ego, closing, stopped_case)
          safety_engaged = bool(np.any(rows[:, 13].astype(bool)))
          stopped_case_failure = (not safety_engaged or float(rows[-1, 1]) > 0.5
                                     or len(tail) == 0 or tail_gap_span > 0.05
                                     or tail_mode_transitions > 2)
          feasible_case_failure = (min_gap < 1.0
                                   or (recovery_started and (recover_time > 15.0
                                       or later_min < recovery_gap - 1e-3))
                                   or len(tail) == 0
                                   or tail[-1, 0] - tail[0, 0] < POST_TARGET_SETTLE_S
                                   or tail_target_error > 0.25 or tail_mode_transitions > 2
                                   or tail_gas_d_p95 > COMMAND_DERIVATIVE_P95_MAX
                                   or tail_brake_d_p95 > COMMAND_DERIVATIVE_P95_MAX)
          if stopped_case_failure if stopped_case else (feasible_case_failure if physical_feasible else stopped_case_failure):
            failures.append({"ego_mph": ego, "closing_mph": closing, "stopped": stopped,
                             "physical_feasible": physical_feasible, "safety_engaged": safety_engaged,
                             "minimum_time_gap_s": min_gap, "recovery_s": recover_time,
                             "recovery_gap_s": recovery_gap, "later_min_gap_s": later_min,
                             "settled_target_s": TARGET_GAP, "settled_gap_span_s": tail_gap_span,
                             "settled_target_error_s": tail_target_error,
                             "gas_derivative_p95": tail_gas_d_p95,
                             "brake_derivative_p95": tail_brake_d_p95,
                             "settled_mode_transitions": tail_mode_transitions})
    assert not failures, f"CR-V fast-closing lead failures: {failures}"
