#!/usr/bin/env python3
"""Run the CR-V quality matrix and write a human-readable HTML report."""

from __future__ import annotations

import html
import hashlib
import json
import math
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
import subprocess

import numpy as np

from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics
from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_multiphase_braking import (
  run_multiphase_stop, stop_failures, stop_metrics,
)

from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import (
  A_CRUISE_MIN,
  COMMAND_DERIVATIVE_P95_MAX,
  FAR_LEAD_GAP_TIME_S,
  GRADE_FEASIBILITY_MARGIN,
  LEAD_IN_S,
  LEAD_SET_SPEED_OVERSHOOT_MPH,
  POST_TARGET_SETTLE_S,
  ROLLING_LEAD_MAX_JERK_MPS3,
  ROLLING_LEAD_P95_JERK_MPS3,
  TARGET_GAP,
  TOTAL_RUN_DURATION_S,
  _direct_mode_reversals,
  _lead_case_physically_feasible,
  _moving_lead_maneuver_quality,
  _p95_command_derivative,
  _p95_jerk,
  _run_cruise,
  _run_far_lead_case,
  _run_lead_case,
  _run_lead_speed_deviation_case,
  _run_rolling_lead_stop_case,
  _run_speed_transition,
  _rolling_lead_stop_time,
  get_max_accel,
)


OUT = Path(__file__).with_name("crv_longitudinal_quality_matrix.html")
OUT_JSON = Path(__file__).with_name("crv_longitudinal_quality_matrix_results.json")


def row(kind, case, passed, metrics, data, target):
  # Keep the dashboard artifact compact; all gates above use the full 20 Hz
  # trace. The 2 Hz series preserves maneuver phases and long-run settling.
  sample = data[::10]
  indices = ({"time_s": 0, "ego_speed_mph": 1, "acceleration_mps2": 2,
              "gas_command": 3, "brake_request": 4, "brake_intensity": 5,
              "actuator_mode": 6, "mode_transitions": 7, "planner_acceleration_mps2": 8}
             if kind.startswith("Speed") else
             {"time_s": 0, "ego_speed_mph": 1, "gas_command": 3,
              "brake_intensity": 4, "brake_request": 5, "actuator_mode": 6,
              "mode_transitions": 7, "acceleration_mps2": 8,
              "planner_acceleration_mps2": 9}
             if kind == "Cruise" else
             {"time_s": 0, "ego_speed_mph": 1, "gap_m": 2, "time_gap_s": 3,
              "acceleration_mps2": 4, "lead_speed_mph": 5,
              "planner_acceleration_mps2": 6, "gas_command": 7,
              "brake_intensity": 8, "brake_request": 9,
              "actuator_mode": 10, "mode_transitions": 11,
              "predictive_brake_mps2": 12, "safety_override": 13})
  signals = {name: [str(value) if name == "actuator_mode" else float(value)
                    for value in sample[:, index]] for name, index in indices.items()}
  signals["controller_acceleration_mps2"] = [float(value) for value in sample[:, -1]]
  signals["observed_speed_mph"] = [float(value) for value in sample[:, -3]]
  signals["observed_acceleration_mps2"] = [float(value) for value in sample[:, -2]]
  signals["target_speed_mph"] = [float(target)] * len(sample)
  times = np.asarray(signals["time_s"])
  accel = np.asarray(signals["acceleration_mps2"])
  signals["jerk_mps3"] = np.gradient(accel, times).tolist()
  return {"kind": kind, "case": case, "passed": passed, "metrics": metrics, "signals": signals}


def _speed_row(start, target):
  data = _run_speed_transition(start, target)
  speed = data[:, 1].astype(float)
  crossing = _crossing(data, target, start)
  if start < target:
    extreme = float(np.max(speed))
    passed = math.isfinite(crossing) and extreme - target <= 1.0
    metrics = {"target_mph": target, "peak_mph": extreme, "overshoot_mph": extreme - target}
    kind, case = "Speed up", f"{start} → {target} mph"
  else:
    extreme = float(np.min(speed))
    passed = math.isfinite(crossing) and target - extreme <= 1.0
    metrics = {"target_mph": target, "trough_mph": extreme, "undershoot_mph": target - extreme}
    kind, case = "Speed down", f"{start} → {target} mph"
  settled = data[data[:, 0].astype(float) >= crossing] if math.isfinite(crossing) else data[:0]
  settled_duration = float(settled[-1, 0] - settled[0, 0]) if len(settled) else 0.0
  settled_error = float(np.max(np.abs(settled[:, 1].astype(float) - target))) if len(settled) else math.inf
  stable = data[data[:, 0].astype(float) >= float(data[-1, 0]) - POST_TARGET_SETTLE_S]
  gas_d_p95 = _p95_command_derivative(stable[:, 3])
  brake_d_p95 = _p95_command_derivative(stable[:, 5], 0.01 / 3.5)
  passed = (passed and settled_duration >= POST_TARGET_SETTLE_S and settled_error <= 1.0
            and gas_d_p95 <= COMMAND_DERIVATIVE_P95_MAX
            and brake_d_p95 <= COMMAND_DERIVATIVE_P95_MAX)
  metrics.update({"crossing_s": crossing, "settled_duration_s": settled_duration,
                  "settled_error_mph": settled_error, "gas_derivative_p95": gas_d_p95,
                  "brake_derivative_p95": brake_d_p95, "mode": data[-1, 6],
                  "transitions": int(data[-1, 7])})
  return row(kind, case, passed, metrics, data, target)


def _cruise_row(target, grade):
  data = _run_cruise(target, grade)
  settled = data[data[:, 0].astype(float) >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
  speed = settled[:, 1].astype(float)
  gas = settled[:, 3].astype(float)
  brake = settled[:, 4].astype(float)
  speed_error = float(np.max(np.abs(speed - target)))
  gas_span = float(np.ptp(gas))
  brake_span = float(np.ptp(brake))
  gas_d_p95 = _p95_command_derivative(settled[:, 3])
  brake_d_p95 = _p95_command_derivative(settled[:, 4], 0.01 / 3.5)
  transitions = int(settled[-1, 7]) - int(settled[0, 7])
  required_accel = 9.81 * math.sin(math.atan(grade / 100.0)) + 0.012
  feasible = (A_CRUISE_MIN + GRADE_FEASIBILITY_MARGIN <= required_accel
              <= get_max_accel(target * 0.44704) - GRADE_FEASIBILITY_MARGIN)
  passed = ((feasible and speed_error <= 1.0 and gas_span <= 0.05 and brake_span <= 0.05
             and gas_d_p95 <= COMMAND_DERIVATIVE_P95_MAX
             and brake_d_p95 <= COMMAND_DERIVATIVE_P95_MAX)
            or (not feasible and gas_span <= 0.10 and brake_span <= 0.10)) and transitions <= 2
  return row("Cruise", f"{target} mph @ {grade:+d}%", passed,
             {"target_mph": target, "grade_percent": grade, "settled_mode": settled[-1, 6],
              "speed_error_mph": speed_error, "feasible": feasible,
              "required_accel_mps2": required_accel, "gas_span": gas_span,
              "brake_intensity_span": brake_span, "transition_count": transitions,
              "gas_derivative_p95": gas_d_p95, "brake_derivative_p95": brake_d_p95,
              "jerk_p95": _p95_jerk(settled)}, data, target)


def _lead_row(ego, closing, stopped):
  data = _run_lead_case(ego, closing, stopped)
  times = data[:, 0].astype(float)
  stopped_case = stopped or closing >= ego
  gaps = data[:, 2 if stopped_case else 3].astype(float)
  closest = int(np.argmin(gaps))
  recovery = (np.flatnonzero((np.arange(len(data)) > closest) & (gaps >= TARGET_GAP - 0.25))
              if not stopped_case and float(np.min(gaps)) < TARGET_GAP - 0.25 else np.empty(0, dtype=int))
  recovery_i = int(recovery[0]) if len(recovery) else len(data)
  recovery_gap = float(gaps[recovery_i]) if recovery_i < len(data) else math.inf
  later_min = float(np.min(gaps[recovery_i + 1:])) if recovery_i + 1 < len(data) else recovery_gap
  recovery_s = float(times[recovery_i] - times[closest]) if recovery_i < len(data) else math.inf
  min_gap = float(gaps[closest])
  tail = data[times >= float(data[-1, 0]) - POST_TARGET_SETTLE_S]
  tail_gaps = tail[:, 2 if stopped_case else 3].astype(float)
  tail_span = float(np.ptp(tail_gaps)) if len(tail) else math.inf
  tail_gas_d_p95 = _p95_command_derivative(tail[:, 7]) if len(tail) else math.inf
  tail_brake_d_p95 = _p95_command_derivative(tail[:, 8], 0.01 / 3.5) if len(tail) else math.inf
  tail_target_error = (float(np.max(np.abs(tail_gaps - TARGET_GAP)))
                       if len(tail) and not stopped_case else 0.0 if len(tail) else math.inf)
  tail_transitions = int(tail[-1, 11]) - int(tail[0, 11]) if len(tail) else math.inf
  physical_feasible = _lead_case_physically_feasible(ego, closing, stopped_case)
  safety_engaged = bool(np.any(data[:, 13].astype(bool)))
  phases, speed_undershoot, match_gap = _moving_lead_maneuver_quality(data)
  maneuver_failure = (speed_undershoot > 1.0 or not math.isfinite(match_gap)
                      or (not safety_engaged and (phases.count("brake") > 1
                          or phases.count("gas") > 2
                          or abs(match_gap - TARGET_GAP) > 0.25)))
  terminal_failure = (min_gap < 0.0 or float(data[-1, 1]) > 0.5 or len(tail) == 0
                      or tail_span > 0.05 or tail_transitions > 2)
  moving_failure = (min_gap < 1.0 or maneuver_failure
                    or (recovery_i < len(data) and (recovery_s > 15.0
                        or later_min < recovery_gap - 1e-3))
                    or len(tail) == 0 or tail_target_error > 0.25
                    or tail_transitions > 2
                    or tail_gas_d_p95 > COMMAND_DERIVATIVE_P95_MAX
                    or tail_brake_d_p95 > COMMAND_DERIVATIVE_P95_MAX)
  passed = (not terminal_failure if stopped_case else
            (not moving_failure if physical_feasible else not terminal_failure))
  return row("Lead", f"{ego} mph, {closing} mph closing, {'stopped' if stopped else 'moving'}",
             passed, {"ego_mph": ego, "closing_mph": closing, "stopped": stopped,
                      "physical_feasible": physical_feasible, "safety_engaged": safety_engaged,
                      "minimum_time_gap_s": min_gap, "recovery_s": recovery_s,
                      "recovery_gap_s": recovery_gap, "later_min_gap_s": later_min,
                      "target_gap_s": TARGET_GAP, "settled_gap_span_s": tail_span,
                      "settled_target_error_s": tail_target_error,
                      "settled_mode_transitions": tail_transitions,
                      "gas_derivative_p95": tail_gas_d_p95,
                      "brake_derivative_p95": tail_brake_d_p95,
                      "maneuver_phases": phases,
                      "velocity_undershoot_mph": speed_undershoot,
                      "velocity_match_gap_s": match_gap}, data, ego)


def _rolling_stop_row(ego):
  data = _run_rolling_lead_stop_case(ego)
  times = data[:, 0].astype(float)
  settled = data[times >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
  stopped_tail = data[times >= _rolling_lead_stop_time(ego)]
  tail_gas_d_p95 = _p95_command_derivative(settled[:, 7])
  tail_brake_d_p95 = _p95_command_derivative(settled[:, 8], 0.01 / 3.5)
  tail_transitions = int(settled[-1, 11]) - int(settled[0, 11])
  maneuver = data[(times >= LEAD_IN_S + 3.0) & (times <= _rolling_lead_stop_time(ego))]
  maneuver_jerk = np.gradient(maneuver[:, 6].astype(float), 1.0 / 20.0)
  maneuver_jerk_p95 = float(np.percentile(np.abs(maneuver_jerk), 95))
  maneuver_jerk_max = float(np.max(np.abs(maneuver_jerk)))
  minimum_gap = float(np.min(data[:, 2].astype(float)))
  terminal_speed = float(data[-1, 1])
  late_acceleration = float(np.max(stopped_tail[:, 4].astype(float)))
  settled_gap_span = float(np.ptp(settled[:, 3].astype(float)))
  passed = (minimum_gap >= 1.0 and terminal_speed <= 0.5 and late_acceleration <= 0.05
            and settled_gap_span <= 0.05 and tail_gas_d_p95 <= COMMAND_DERIVATIVE_P95_MAX
            and tail_brake_d_p95 <= COMMAND_DERIVATIVE_P95_MAX and tail_transitions <= 2
            and maneuver_jerk_p95 <= ROLLING_LEAD_P95_JERK_MPS3
            and maneuver_jerk_max <= ROLLING_LEAD_MAX_JERK_MPS3)
  return row("Rolling lead stop", f"{ego} mph ego", passed,
             {"ego_mph": ego, "minimum_gap_m": minimum_gap, "terminal_speed_mph": terminal_speed,
              "late_acceleration_mps2": late_acceleration, "settled_time_gap_span_s": settled_gap_span,
              "maneuver_jerk_p95_mps3": maneuver_jerk_p95,
              "maneuver_jerk_max_mps3": maneuver_jerk_max,
              "gas_derivative_p95": tail_gas_d_p95, "brake_derivative_p95": tail_brake_d_p95,
              "settled_mode_transitions": tail_transitions,
              "safety_engaged": bool(np.any(data[:, 13].astype(bool)))}, data, ego)


def _lead_speed_deviation_row(ego, lead, seed):
  data = _run_lead_speed_deviation_case(ego, lead, seed)
  times = data[:, 0].astype(float)
  settled = data[times >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
  gaps = data[:, 3].astype(float)
  tail_modes = settled[:, 10]
  peak_speed = float(np.max(data[:, 1].astype(float)))
  tail_gas_d_p95 = _p95_command_derivative(settled[:, 7])
  tail_brake_d_p95 = _p95_command_derivative(settled[:, 8], 0.01 / 3.5)
  tail_transitions = int(settled[-1, 11]) - int(settled[0, 11])
  direct_reversals = _direct_mode_reversals(tail_modes)
  minimum_time_gap = float(np.min(gaps))
  overshoot = peak_speed - ego
  passed = (minimum_time_gap >= 1.0 and overshoot <= LEAD_SET_SPEED_OVERSHOOT_MPH
            and tail_transitions <= 8 and direct_reversals == 0
            and tail_gas_d_p95 <= 0.02 and tail_brake_d_p95 <= 0.02)
  return row("Lead speed deviations", f"{ego} mph ego / {lead} mph nominal / seed {seed}", passed,
             {"ego_mph": ego, "lead_mph": lead, "seed": seed,
              "minimum_time_gap_s": minimum_time_gap, "peak_speed_mph": peak_speed,
              "set_speed_overshoot_mph": overshoot, "gas_derivative_p95": tail_gas_d_p95,
              "brake_derivative_p95": tail_brake_d_p95, "settled_mode_transitions": tail_transitions,
              "gas_brake_reversals": direct_reversals}, data, ego)


def _far_lead_row(ego):
  data = _run_far_lead_case(ego)
  set_speed = ego + 10.0
  times = data[:, 0].astype(float)
  settled = data[times >= TOTAL_RUN_DURATION_S - POST_TARGET_SETTLE_S]
  time_gap = data[:, 3].astype(float)
  initial_time_gap = float(data[0, 3])
  final_time_gap = float(time_gap[-1])
  peak_speed = float(np.max(data[:, 1].astype(float)))
  settled_gap_error = float(np.max(np.abs(settled[:, 3].astype(float) - TARGET_GAP)))
  settled_gap_span = float(np.ptp(settled[:, 3].astype(float)))
  tail_transitions = int(settled[-1, 11]) - int(settled[0, 11])
  tail_gas_d_p95 = _p95_command_derivative(settled[:, 7])
  tail_brake_d_p95 = _p95_command_derivative(settled[:, 8], 0.01 / 3.5)
  overshoot = peak_speed - set_speed
  passed = (overshoot <= LEAD_SET_SPEED_OVERSHOOT_MPH
            and final_time_gap < initial_time_gap - 1.0 and settled_gap_error <= 0.25
            and settled_gap_span <= 0.05 and tail_transitions <= 2
            and tail_gas_d_p95 <= COMMAND_DERIVATIVE_P95_MAX
            and tail_brake_d_p95 <= COMMAND_DERIVATIVE_P95_MAX)
  return row("Far lead convergence", f"{set_speed} mph set / {ego} mph lead / {FAR_LEAD_GAP_TIME_S:.1f}s initial gap", passed,
             {"target_lead_mph": ego, "set_speed_mph": set_speed, "initial_time_gap_s": initial_time_gap,
              "final_time_gap_s": final_time_gap, "settled_target_error_s": settled_gap_error,
              "peak_speed_mph": peak_speed, "set_speed_overshoot_mph": overshoot,
              "settled_time_gap_span_s": settled_gap_span, "settled_mode_transitions": tail_transitions,
              "gas_derivative_p95": tail_gas_d_p95, "brake_derivative_p95": tail_brake_d_p95}, data, set_speed)


def _multiphase_stop_row(ego, pulse_amplitude=0.22):
  data = run_multiphase_stop(ego, pulse_amplitude=pulse_amplitude)
  metrics = stop_metrics(data)
  metrics["failures"] = stop_failures(metrics)
  # Preserve 20 Hz for these short sensor-triggered events; 2 Hz sampling
  # would hide the exact command spike the new regression reproduces.
  result = row("Multiphase braking", f"{ego} mph / rolling stop / {pulse_amplitude:.2f} m/s raw-speed excursion",
               not metrics["failures"], metrics, data, ego)
  indices = {"time_s": 0, "ego_speed_mph": 1, "gap_m": 2, "time_gap_s": 3,
             "acceleration_mps2": 4, "lead_speed_mph": 5, "planner_acceleration_mps2": 6,
             "gas_command": 7, "brake_intensity": 8, "brake_request": 9,
             "actuator_mode": 10, "mode_transitions": 11, "predictive_brake_mps2": 12,
             "safety_override": 13, "observed_speed_mph": 14, "observed_acceleration_mps2": 15,
             "controller_acceleration_mps2": 16}
  result["signals"] = {name: data[:, index].astype(str if name == "actuator_mode" else float).tolist()
                       for name, index in indices.items()}
  result["signals"]["target_speed_mph"] = [ego] * len(data)
  result["signals"]["jerk_mps3"] = np.gradient(data[:, 4].astype(float), data[:, 0].astype(float)).tolist()
  return result


def run_matrix():
  jobs = []
  for target in (15, 25, 35, 45, 55, 65, 75, 85, 90):
    jobs.append((_speed_row, (0, target)))
  for target in (65, 55, 45, 35, 25, 15, 5, 0):
    jobs.append((_speed_row, (90, target)))
  for target in range(10, 91, 10):
    for grade in range(-15, 16, 3):
      jobs.append((_cruise_row, (target, grade)))
  for ego in (25, 45, 65, 85):
    for closing in (1, 3, 5, 10, 15, 20, 30, 40):
      for stopped in (False, True):
        jobs.append((_lead_row, (ego, closing, stopped)))
  for ego in (35, 50):
    jobs.append((_rolling_stop_row, (ego,)))
  for case in ((25, 18, 1), (25, 18, 2), (45, 35, 1), (45, 35, 2),
               (65, 55, 1), (65, 55, 2), (85, 70, 1), (85, 70, 2)):
    jobs.append((_lead_speed_deviation_row, case))
  for ego in (25, 45, 65):
    jobs.append((_far_lead_row, (ego,)))
  for ego in (25, 35, 50):
    jobs.append((_multiphase_stop_row, (ego,)))
  for pulse_amplitude in (0.0, 0.16, 0.30):
    jobs.append((_multiphase_stop_row, (50, pulse_amplitude)))

  with ProcessPoolExecutor(max_workers=32) as executor:
    return list(executor.map(_run_job, jobs))


def _run_job(job):
  function, args = job
  return function(*args)


def _crossing(data, target, start):
  direction = np.sign(target - start)
  crossed = np.flatnonzero(direction * (data[:, 1].astype(float) - target) >= 0.0)
  return float(data[crossed[0], 0]) if len(crossed) else math.inf


def write_report(rows):
  root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
  paths = {"planner_sha256": "openpilot/selfdrive/controls/lib/longitudinal_planner.py",
           "inner_sha256": "openpilot/selfdrive/controls/lib/longcontrol.py",
           "plant_sha256": "openpilot/selfdrive/test/longitudinal_maneuvers/honda_vehicle.py",
           "carstate_sha256": "opendbc_repo/opendbc/car/honda/carstate.py"}
  metadata = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "dynamics": asdict(HondaDynamics()),
              **{k: hashlib.sha256((root / p).read_bytes()).hexdigest() for k, p in paths.items()}}
  passed = sum(r["passed"] for r in rows)
  failed = len(rows) - passed
  body = []
  for r in rows:
    status = "PASS" if r["passed"] else "FAIL"
    status_class = "pass" if r["passed"] else "fail"
    metrics = "<br>".join(f"{html.escape(str(k))}={html.escape(str(v))}" for k, v in r["metrics"].items())
    body.append(
      "".join((f"<tr class='{status_class}'><td>{status}</td><td>{html.escape(r['kind'])}</td>",
               f"<td>{html.escape(r['case'])}</td><td>{metrics}</td></tr>"))
    )
  generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
  gate_text = "".join(("speed error ≤1 mph; settled cruise speed error ≤1 mph, gas/brake span ≤5%, ",
                       f"transitions ≤2; lead minimum gap ≥1.0 s and recovery ≤15 s to {TARGET_GAP:.2f} ±0.25 s"))
  document = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>CR-V longitudinal quality matrix</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222 }}
table {{ border-collapse: collapse; width: 100% }} th, td {{ border: 1px solid #ccc; padding: .45rem; vertical-align: top; text-align: left }}
th {{ background: #222; color: white; position: sticky; top: 0 }} .pass {{ background: #e9f7ed }} .fail {{ background: #fdeaea }}
.summary {{ font-size: 1.1rem; margin-bottom: 1rem }} .count {{ font-weight: 700 }}
</style></head><body>
<h1>CR-V longitudinal quality regression matrix</h1>
<p>Generated {generated}. Gates: {gate_text}.</p>
<p>Local simulation, not a hardware validation. Source base {metadata['source_commit'][:12]};
planner SHA-256 {metadata['planner_sha256']}. Includes the calibrated actuator and openpilot speed observer.</p>
<p class='summary'><span class='count'>{passed} PASS</span> &nbsp; <span class='count'>{failed} FAIL</span> &nbsp; {len(rows)} total cases</p>
<table><thead><tr><th>Status</th><th>Matrix</th><th>Case</th><th>Measured diagnostics</th></tr></thead><tbody>{''.join(body)}</tbody></table>
</body></html>"""
  OUT.write_text(document)
  OUT_JSON.write_text(json.dumps({"schema_version": 1,
                                  "generated_at": datetime.now(UTC).isoformat(),
                                  **metadata,
                                  "results": rows}, default=str, separators=(",", ":")))
  print(f"wrote {OUT} ({passed} pass, {failed} fail, {len(rows)} total)")


if __name__ == "__main__":
  write_report(run_matrix())
