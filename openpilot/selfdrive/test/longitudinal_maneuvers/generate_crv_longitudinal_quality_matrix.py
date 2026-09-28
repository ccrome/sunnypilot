#!/usr/bin/env python3
"""Run the CR-V quality matrix and write a human-readable HTML report."""

from __future__ import annotations

import html
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import (
  TARGET_GAP,
  _p95_jerk,
  _run_cruise,
  _run_lead_case,
  _run_speed_transition,
)


OUT = Path(__file__).with_name("crv_longitudinal_quality_matrix.html")


def row(kind, case, passed, metrics):
  return {"kind": kind, "case": case, "passed": passed, "metrics": metrics}


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
  metrics.update({"crossing_s": crossing, "mode": data[-1, 6], "transitions": int(data[-1, 7])})
  return row(kind, case, passed, metrics)


def _cruise_row(target, grade):
  data = _run_cruise(target, grade)
  settled = data[data[:, 0].astype(float) >= 25.0]
  speed = settled[:, 1].astype(float)
  gas = settled[:, 3].astype(float)
  brake = settled[:, 4].astype(float)
  speed_error = float(np.max(np.abs(speed - target)))
  gas_span = float(np.ptp(gas))
  brake_span = float(np.ptp(brake))
  transitions = int(settled[-1, 7]) - int(settled[0, 7])
  passed = speed_error <= 1.0 and gas_span <= 0.05 and brake_span <= 0.05 and transitions <= 2
  return row("Cruise", f"{target} mph @ {grade:+d}%", passed,
             {"target_mph": target, "grade_percent": grade, "settled_mode": settled[-1, 6],
              "speed_error_mph": speed_error, "gas_span": gas_span,
              "brake_intensity_span": brake_span, "transition_count": transitions,
              "jerk_p95": _p95_jerk(settled)})


def _lead_row(ego, closing, stopped):
  data = _run_lead_case(ego, closing, stopped)
  times = data[:, 0].astype(float)
  gaps = data[:, 3].astype(float)
  closest = int(np.argmin(gaps))
  recovery = np.flatnonzero((np.arange(len(data)) > closest) & (gaps >= TARGET_GAP - 0.25))
  recovery_i = int(recovery[0]) if len(recovery) else len(data)
  recovery_gap = float(gaps[recovery_i]) if recovery_i < len(data) else math.inf
  later_min = float(np.min(gaps[recovery_i + 1:])) if recovery_i + 1 < len(data) else recovery_gap
  recovery_s = float(times[recovery_i] - times[closest]) if recovery_i < len(data) else math.inf
  min_gap = float(gaps[closest])
  passed = min_gap >= 1.0 and recovery_s <= 15.0 and later_min >= recovery_gap - 1e-3
  return row("Lead", f"{ego} mph, {closing} mph closing, {'stopped' if stopped else 'moving'}",
             passed, {"ego_mph": ego, "closing_mph": closing, "stopped": stopped,
                      "minimum_time_gap_s": min_gap, "recovery_s": recovery_s,
                      "recovery_gap_s": recovery_gap, "later_min_gap_s": later_min,
                      "target_gap_s": TARGET_GAP})


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
<p class='summary'><span class='count'>{passed} PASS</span> &nbsp; <span class='count'>{failed} FAIL</span> &nbsp; {len(rows)} total cases</p>
<table><thead><tr><th>Status</th><th>Matrix</th><th>Case</th><th>Measured diagnostics</th></tr></thead><tbody>{''.join(body)}</tbody></table>
</body></html>"""
  OUT.write_text(document)
  print(f"wrote {OUT} ({passed} pass, {failed} fail, {len(rows)} total)")


if __name__ == "__main__":
  write_report(run_matrix())
