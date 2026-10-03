"""Closed-loop CR-V longitudinal quality gates.

These tests exercise the planner, production 100 Hz LongControl and Honda
CarController, encoded CAN commands, and causal vehicle dynamics through Plant.
The vehicle response is a configurable grey-box approximation; passing these
tests does not establish that every hardware response has been modeled.
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
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  A_CRUISE_MIN, CRV_LEAD_FOLLOW_EMERGENCY_BRAKE_JERK, get_max_accel)
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
LEAD_SET_SPEED_OVERSHOOT_MPH = 1.5
FAR_LEAD_GAP_TIME_S = 6.0
LEAD_SPEED_DEVIATION_STD_MPH = 1.5
LEAD_SPEED_DEVIATION_KNOT_S = 2.0
ROLLING_LEAD_CRAWL_MPH = 1.5
ROLLING_LEAD_MAX_JERK_MPS3 = 8.0
ROLLING_LEAD_P95_JERK_MPS3 = 3.0
LEAD_INITIAL_GAP_MARGIN_M = 3.0
LEAD_INITIAL_TIME_MARGIN_S = 1.0


def _jerk_limited_braking_distance(relative_speed: float) -> float:
  """Distance to remove a relative speed with the CR-V safety jerk envelope."""
  if relative_speed <= 0.0:
    return 0.0
  max_decel = abs(ACCEL_MIN)
  brake_jerk = CRV_LEAD_FOLLOW_EMERGENCY_BRAKE_JERK
  ramp_time = max_decel / brake_jerk
  ramp_speed_loss = 0.5 * max_decel * ramp_time
  if relative_speed <= ramp_speed_loss:
    stop_time = math.sqrt(2.0 * relative_speed / brake_jerk)
    return relative_speed * stop_time - brake_jerk * stop_time ** 3 / 6.0
  remaining_speed = relative_speed - ramp_speed_loss
  ramp_distance = (relative_speed * ramp_time
                   - brake_jerk * ramp_time ** 3 / 6.0)
  return ramp_distance + remaining_speed ** 2 / (2.0 * max_decel)


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
                 plant.actuator_mode, plant.mode_transitions, plant.planner_acceleration, plant.vehicle.output))
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
                 plant.planner_acceleration, plant.vehicle.output))
  return np.asarray(rows, dtype=object)


def _p95_jerk(rows: np.ndarray) -> float:
  return float(np.percentile(np.abs(np.gradient(rows[:, 8].astype(float), DT)), 95))


def _p95_command_derivative(values: np.ndarray, quantum: float = 1.0 / 1600.0) -> float:
  """Resolved command motion beyond one CAN count, at short and long scales.

  One-count toggling cannot resolve a physical derivative. The one-second
  comparison also catches a persistent ramp made of single-count steps.
  This is measurement uncertainty, not filtering in the control loop.
  """
  values = values.astype(float)
  rates = [np.percentile(np.maximum(0.0, np.abs(np.diff(values)) - quantum - 1e-12) / DT, 95)]
  lag = round(1.0 / DT)
  if len(values) > lag:
    rates.append(np.percentile(np.maximum(0.0, np.abs(values[lag:] - values[:-lag]) - quantum - 1e-12), 95))
  return float(max(rates))


def _direct_mode_reversals(modes: np.ndarray) -> int:
  return sum({str(previous), str(current)} == {"gas", "brake"}
             for previous, current in zip(modes, modes[1:], strict=False))


def _moving_lead_maneuver_quality(rows: np.ndarray) -> tuple[list[str], float, float]:
  maneuver = rows[rows[:, 0].astype(float) >= LEAD_IN_S]
  modes = maneuver[:, 10]
  changes = np.flatnonzero(modes[1:] != modes[:-1]) + 1
  phases = [str(modes[0]), *(str(modes[i]) for i in changes)]
  releases = [i for i in changes if modes[i - 1] == "brake" and modes[i] == "gas"]
  if not releases:
    return phases, 0.0, TARGET_GAP
  release = maneuver[releases[0]]
  return phases, abs(float(release[1]) - float(release[5])), float(release[3])


def _rolling_lead_speed(ego_mph: float, elapsed_s: float) -> float:
  # The lead decelerates to a crawl, holds that crawl long enough to be
  # observable, then completes the stop.  The interpolation keeps the test
  # input continuous instead of injecting an artificial speed step. Scale the
  # deceleration time by speed so every case has the same lead deceleration
  # envelope instead of making the high-speed case artificially severe.
  decel_duration = max(8.0, (ego_mph - ROLLING_LEAD_CRAWL_MPH) * MPH / 1.5)
  crawl_start = 3.0 + decel_duration
  crawl_end = crawl_start + 12.0
  stop_end = crawl_end + 3.0
  times = (0.0, 3.0, crawl_start, crawl_end, stop_end, RUN_DURATION_S)
  speeds = (ego_mph, ego_mph, ROLLING_LEAD_CRAWL_MPH, ROLLING_LEAD_CRAWL_MPH, 0.0, 0.0)
  return float(np.interp(elapsed_s, times, speeds))


def _rolling_lead_stop_time(ego_mph: float) -> float:
  decel_duration = max(8.0, (ego_mph - ROLLING_LEAD_CRAWL_MPH) * MPH / 1.5)
  return LEAD_IN_S + 3.0 + decel_duration + 12.0 + 3.0


def _lead_speed_deviation_profile(base_lead_mph: float, seed: int):
  rng = np.random.default_rng(seed)
  knot_times = np.arange(0.0, RUN_DURATION_S + LEAD_SPEED_DEVIATION_KNOT_S,
                         LEAD_SPEED_DEVIATION_KNOT_S)
  deviations = rng.normal(0.0, LEAD_SPEED_DEVIATION_STD_MPH, len(knot_times))
  # Start at the requested nominal speed, then apply a fixed, zero-mean
  # distribution of meaningful lead-speed changes. Return to nominal before
  # the required settled window so the command-stability gate measures a true
  # settled state rather than a continuously moving target.
  deviations[0] = 0.0
  disturbance_end = RUN_DURATION_S - POST_TARGET_SETTLE_S - 30.0
  disturbance_knots = knot_times < disturbance_end
  deviations[disturbance_knots] -= np.mean(deviations[disturbance_knots])
  deviations[~disturbance_knots] = 0.0
  deviations = np.clip(deviations, -3.0 * LEAD_SPEED_DEVIATION_STD_MPH,
                       3.0 * LEAD_SPEED_DEVIATION_STD_MPH)
  return knot_times, np.maximum(0.0, base_lead_mph + deviations)


def _run_lead_profile(ego_mph: float, profile: str, lead_speed_mph: float = 0.0,
                      initial_gap: float | None = None, seed: int = 0) -> np.ndarray:
  ego = ego_mph * MPH
  if initial_gap is None:
    relative_speed = max(ego - lead_speed_mph * MPH, 0.0)
    target_gap_at_maneuver = max(2.0, lead_speed_mph * MPH * TARGET_GAP)
    braking_distance = _jerk_limited_braking_distance(relative_speed)
    initial_gap = max(2.5 * ego,
                      braking_distance + target_gap_at_maneuver
                      + LEAD_INITIAL_GAP_MARGIN_M + ego * LEAD_INITIAL_TIME_MARGIN_S)
  plant = Plant(lead_relevancy=True, speed=ego,
                distance_lead=initial_gap, physics=True, realtime=False,
                personality=log.LongitudinalPersonality.relaxed,
                car_fingerprint=CAR.HONDA_CRV_5G, sim_rate=SIM_RATE)
  deviation_times = deviation_speeds = None
  if profile == "deviation":
    deviation_times, deviation_speeds = _lead_speed_deviation_profile(lead_speed_mph, seed)
  rows = []
  while plant.current_time < TOTAL_RUN_DURATION_S:
    approaching = plant.current_time >= LEAD_IN_S
    elapsed = max(0.0, plant.current_time - LEAD_IN_S)
    if not approaching:
      lead_speed = plant.speed
    elif profile == "fixed":
      lead_speed = lead_speed_mph * MPH
    elif profile == "rolling_stop":
      lead_speed = _rolling_lead_speed(ego_mph, elapsed) * MPH
    elif profile == "deviation":
      lead_speed = float(np.interp(elapsed, deviation_times, deviation_speeds)) * MPH
    elif profile == "stopped":
      lead_speed = 0.0
    else:
      raise ValueError(f"unknown lead profile: {profile}")
    plant.step(v_lead=lead_speed, prob_lead=1.0 if approaching else 0.0, v_cruise=ego)
    gap = max(0.0, plant.distance_lead - plant.distance)
    time_gap = gap / max(plant.speed, 0.1)
    rows.append((plant.current_time, plant.speed / MPH, gap, time_gap,
                 plant.acceleration, lead_speed / MPH, plant.planner_acceleration,
                 plant.gas_command, plant.brake_intensity, plant.brake_request,
                 plant.actuator_mode, plant.mode_transitions,
                 plant.predictive_brake, plant.safety_override, plant.vehicle.output))
  return np.asarray(rows, dtype=object)


def _run_lead_case(ego_mph: float, closing_mph: float, stopped: bool) -> np.ndarray:
  # Keep this helper in mph at its boundary.  Passing the converted m/s value
  # into _run_lead_profile would apply MPH a second time and make the labeled
  # closing-speed cases harsher than requested.
  lead_speed_at_maneuver = max(0.0, ego_mph - closing_mph)
  stopped_lead = stopped or closing_mph >= ego_mph
  profile = "stopped" if stopped_lead else "fixed"
  return _run_lead_profile(ego_mph, profile, lead_speed_at_maneuver)


def _run_rolling_lead_stop_case(ego_mph: float) -> np.ndarray:
  ego = ego_mph * MPH
  braking_distance = _jerk_limited_braking_distance(ego)
  initial_gap = max(2.5 * ego, braking_distance + 2.0)
  return _run_lead_profile(ego_mph, "rolling_stop", initial_gap=initial_gap)


def _run_lead_speed_deviation_case(ego_mph: float, lead_mph: float, seed: int) -> np.ndarray:
  ego = ego_mph * MPH
  worst_relative_speed = max(ego - (lead_mph - 3.0 * LEAD_SPEED_DEVIATION_STD_MPH) * MPH, 0.0)
  braking_distance = _jerk_limited_braking_distance(worst_relative_speed)
  target_gap = max(2.0, lead_mph * MPH * TARGET_GAP)
  initial_gap = max(2.5 * ego,
                    braking_distance + target_gap + LEAD_INITIAL_GAP_MARGIN_M
                    + ego * LEAD_INITIAL_TIME_MARGIN_S)
  return _run_lead_profile(ego_mph, "deviation", lead_mph, initial_gap, seed)


def _run_far_lead_case(target_lead_mph: float) -> np.ndarray:
  set_speed_mph = target_lead_mph + 10.0
  ego = set_speed_mph * MPH
  initial_gap = FAR_LEAD_GAP_TIME_S * ego
  return _run_lead_profile(set_speed_mph, "fixed", target_lead_mph, initial_gap)


def _parallel_runs(function, cases):
  with ProcessPoolExecutor(max_workers=min(32, len(cases))) as executor:
    return list(executor.map(function, *zip(*cases, strict=True)))


def _run_speed_case(start_mph, target_mph):
  return _run_speed_transition(start_mph, target_mph)


def _run_cruise_case(target_mph, grade_percent):
  return _run_cruise(target_mph, grade_percent)


def _run_lead_case_spec(ego_mph, closing_mph, stopped):
  return _run_lead_case(ego_mph, closing_mph, stopped)


def _run_rolling_lead_stop_case_spec(ego_mph):
  return _run_rolling_lead_stop_case(ego_mph)


def _run_lead_speed_deviation_case_spec(ego_mph, lead_mph, seed):
  return _run_lead_speed_deviation_case(ego_mph, lead_mph, seed)


def _run_far_lead_case_spec(ego_mph):
  return _run_far_lead_case(ego_mph)


def _lead_case_physically_feasible(ego_mph: float, closing_mph: float, stopped: bool) -> bool:
  ego_speed = ego_mph * MPH
  lead_speed = 0.0 if stopped or closing_mph >= ego_mph else (ego_mph - closing_mph) * MPH
  relative_speed = max(ego_speed - lead_speed, 0.0)
  braking_distance = _jerk_limited_braking_distance(relative_speed)
  target_gap = max(2.0, lead_speed * TARGET_GAP)
  initial_gap = max(2.5 * ego_speed,
                    braking_distance + target_gap + LEAD_INITIAL_GAP_MARGIN_M
                    + ego_speed * LEAD_INITIAL_TIME_MARGIN_S) \
    if not stopped and closing_mph < ego_mph else max(2.5 * ego_speed, 2.0)
  required_gap = braking_distance + target_gap + LEAD_INITIAL_GAP_MARGIN_M
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
      brake_d_p95 = _p95_command_derivative(stable[:, 5], 0.01 / 3.5)
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
      brake_d_p95 = _p95_command_derivative(stable[:, 5], 0.01 / 3.5)
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
      brake_d_p95 = _p95_command_derivative(settled[:, 4], 0.01 / 3.5)
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
      tail_brake_d_p95 = _p95_command_derivative(tail[:, 8], 0.01 / 3.5) if len(tail) else math.inf
      tail_mode_transitions = (int(tail[-1, 11]) - int(tail[0, 11])) if len(tail) else math.inf
      tail_target_error = (float(np.max(np.abs(tail_gaps - TARGET_GAP)))
                           if len(tail) and not stopped_case else 0.0 if len(tail) else math.inf)
      recovery_started = recovery_i < len(rows)
      physical_feasible = _lead_case_physically_feasible(ego, closing, stopped_case)
      safety_engaged = bool(np.any(rows[:, 13].astype(bool)))
      phases, release_speed_error, release_gap = _moving_lead_maneuver_quality(rows)
      maneuver_failure = (len([mode for mode in phases if mode == "brake"]) > 1
                          or len([mode for mode in phases if mode == "gas"]) > 2
                          or safety_engaged
                          or release_speed_error > 1.0
                          or abs(release_gap - TARGET_GAP) > (0.5 if closing >= 40 else 0.25))
      # Emergency intervention is an observation, not a prerequisite for a
      # successful planned stop. Require a collision-free, stable stop either
      # way; ordinary braking must not fail merely for avoiding the fallback.
      stopped_case_failure = (min_gap < 0.0 or float(rows[-1, 1]) > 0.5
                              or len(tail) == 0 or tail_gap_span > 0.05
                              or tail_mode_transitions > 2)
      feasible_case_failure = (min_gap < 1.0
                               or maneuver_failure
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
                         "settled_mode_transitions": tail_mode_transitions,
                         "maneuver_phases": phases,
                         "brake_release_speed_error_mph": release_speed_error,
                         "brake_release_gap_s": release_gap})
    assert not failures, f"CR-V fast-closing lead failures: {failures}"

  def test_rolling_lead_stop_does_not_restart_or_hunt(self):
    failures = []
    cases = [(35,), (50,)]
    for (ego,), rows in zip(cases, _parallel_runs(_run_rolling_lead_stop_case_spec, cases), strict=True):
      times = rows[:, 0].astype(float)
      speed = rows[:, 1].astype(float)
      gap = rows[:, 2].astype(float)
      stopped_tail = rows[times >= _rolling_lead_stop_time(ego)]
      maneuver = rows[(times >= LEAD_IN_S + 3.0) & (times <= LEAD_IN_S + 26.0)]
      settled = rows[times >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
      tail_gas_d_p95 = _p95_command_derivative(settled[:, 7])
      tail_brake_d_p95 = _p95_command_derivative(settled[:, 8], 0.01 / 3.5)
      tail_transitions = int(settled[-1, 11]) - int(settled[0, 11])
      late_acceleration = float(np.max(stopped_tail[:, 4].astype(float)))
      maneuver_jerk = np.gradient(maneuver[:, 6].astype(float), DT)
      maneuver_jerk_p95 = float(np.percentile(np.abs(maneuver_jerk), 95))
      maneuver_jerk_max = float(np.max(np.abs(maneuver_jerk)))
      if (float(np.min(gap)) < 1.0 or float(speed[-1]) > 0.5
          or late_acceleration > 0.05 or float(np.ptp(settled[:, 2].astype(float))) > 0.05
          or maneuver_jerk_p95 > ROLLING_LEAD_P95_JERK_MPS3
          or maneuver_jerk_max > ROLLING_LEAD_MAX_JERK_MPS3
          or tail_gas_d_p95 > COMMAND_DERIVATIVE_P95_MAX
          or tail_brake_d_p95 > COMMAND_DERIVATIVE_P95_MAX or tail_transitions > 2):
        failures.append({"ego_mph": ego, "minimum_gap_m": float(np.min(gap)),
                         "terminal_speed_mph": float(speed[-1]),
                         "late_acceleration_mps2": late_acceleration,
                         "physical_jerk_p95_mps3": float(np.percentile(
                           np.abs(np.gradient(maneuver[:, 4].astype(float), DT)), 95)),
                         "physical_jerk_max_mps3": float(np.max(np.abs(
                           np.gradient(maneuver[:, 4].astype(float), DT)))),
                         "maneuver_jerk_p95_mps3": maneuver_jerk_p95,
                         "maneuver_jerk_max_mps3": maneuver_jerk_max,
                         "settled_gap_span_m": float(np.ptp(settled[:, 2].astype(float))),
                         "gas_derivative_p95": tail_gas_d_p95,
                         "brake_derivative_p95": tail_brake_d_p95,
                         "settled_mode_transitions": tail_transitions,
                         "safety_engaged": bool(np.any(rows[:, 13].astype(bool)))})
    assert not failures, f"CR-V rolling-lead stop failures: {failures}"

  def test_lead_speed_deviations_do_not_create_chatter_or_set_speed_overshoot(self):
    failures = []
    cases = [(25, 18, 1), (25, 18, 2), (45, 35, 1), (45, 35, 2),
             (65, 55, 1), (65, 55, 2), (85, 70, 1), (85, 70, 2)]
    for (ego, lead, seed), rows in zip(cases, _parallel_runs(_run_lead_speed_deviation_case_spec, cases), strict=True):
      times = rows[:, 0].astype(float)
      speed = rows[:, 1].astype(float)
      gaps = rows[:, 3].astype(float)
      settled = rows[times >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
      tail_modes = settled[:, 10]
      tail_gas_d_p95 = _p95_command_derivative(settled[:, 7])
      tail_brake_d_p95 = _p95_command_derivative(settled[:, 8], 0.01 / 3.5)
      tail_transitions = int(settled[-1, 11]) - int(settled[0, 11])
      direct_reversals = _direct_mode_reversals(tail_modes)
      if (float(np.min(gaps)) < 1.0 or float(np.max(speed)) - ego > LEAD_SET_SPEED_OVERSHOOT_MPH
          or tail_transitions > 8 or direct_reversals > 0
          or tail_gas_d_p95 > 0.02 or tail_brake_d_p95 > 0.02):
        failures.append({"ego_mph": ego, "lead_mph": lead, "seed": seed,
                         "minimum_time_gap_s": float(np.min(gaps)),
                         "peak_speed_mph": float(np.max(speed)),
                         "set_speed_overshoot_mph": float(np.max(speed)) - ego,
                         "gas_derivative_p95": tail_gas_d_p95,
                         "brake_derivative_p95": tail_brake_d_p95,
                         "settled_mode_transitions": tail_transitions,
                         "gas_brake_reversals": direct_reversals})
    assert not failures, f"CR-V lead-speed deviation failures: {failures}"

  def test_far_lead_closes_gap_without_exceeding_set_speed(self):
    failures = []
    cases = [(25,), (45,), (65,)]
    for (target_lead,), rows in zip(cases, _parallel_runs(_run_far_lead_case_spec, cases), strict=True):
      ego = target_lead + 10.0
      times = rows[:, 0].astype(float)
      speed = rows[:, 1].astype(float)
      time_gap = rows[:, 3].astype(float)
      settled = rows[times >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
      initial_time_gap = float(rows[0, 3])
      final_time_gap = float(time_gap[-1])
      settled_gap_error = float(np.max(np.abs(settled[:, 3].astype(float) - TARGET_GAP)))
      tail_transitions = int(settled[-1, 11]) - int(settled[0, 11])
      tail_gas_d_p95 = _p95_command_derivative(settled[:, 7])
      tail_brake_d_p95 = _p95_command_derivative(settled[:, 8], 0.01 / 3.5)
      if (float(np.max(speed)) - ego > LEAD_SET_SPEED_OVERSHOOT_MPH
          or final_time_gap >= initial_time_gap - 1.0 or settled_gap_error > 0.25
          or float(np.ptp(settled[:, 3].astype(float))) > 0.05 or tail_transitions > 2
          or tail_gas_d_p95 > COMMAND_DERIVATIVE_P95_MAX
          or tail_brake_d_p95 > COMMAND_DERIVATIVE_P95_MAX):
        failures.append({"target_lead_mph": target_lead, "set_speed_mph": ego,
                         "initial_time_gap_s": initial_time_gap,
                         "final_time_gap_s": final_time_gap,
                         "settled_target_error_s": settled_gap_error,
                         "peak_speed_mph": float(np.max(speed)),
                         "set_speed_overshoot_mph": float(np.max(speed)) - ego,
                         "settled_time_gap_span_s": float(np.ptp(settled[:, 3].astype(float))),
                         "settled_mode_transitions": tail_transitions,
                         "gas_derivative_p95": tail_gas_d_p95,
                         "brake_derivative_p95": tail_brake_d_p95})
    assert not failures, f"CR-V far-lead convergence failures: {failures}"
