"""Fast CR-V quality subset for planner iteration.

This intentionally keeps the production-rate Plant and the full regression
gates, but selects representative cases so a tuning iteration completes in
under two minutes.
"""

from __future__ import annotations

import math

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.longitudinal_maneuvers import test_crv_longitudinal_quality_regression as q


class TestCrvLongitudinalQualitySubset(OpenpilotTestCase):
  def test_representative_speed_transitions(self):
    failures = []
    for start, target in ((0, 15), (0, 45), (0, 75), (90, 45), (90, 15)):
      rows = q._run_speed_transition(start, target)
      speed = rows[:, 1].astype(float)
      crossing = q._crossing_time(rows, target, start)
      extreme = float(np.max(speed) if target > start else np.min(speed))
      error = extreme - target if target > start else target - extreme
      settled = rows[rows[:, 0].astype(float) >= crossing] if math.isfinite(crossing) else rows[:0]
      settled_duration = float(settled[-1, 0] - settled[0, 0]) if len(settled) else 0.0
      settled_error = float(np.max(np.abs(settled[:, 1].astype(float) - target))) if len(settled) else math.inf
      if (not math.isfinite(crossing) or error > 1.0
          or settled_duration < q.POST_TARGET_SETTLE_S or settled_error > 1.0):
        failures.append({"start_mph": start, "target_mph": target, "extreme_mph": extreme,
                         "error_mph": error, "crossing_s": crossing,
                         "settled_duration_s": settled_duration, "settled_error_mph": settled_error,
                         "mode": rows[-1, 6], "transitions": int(rows[-1, 7])})
    assert not failures, f"CR-V subset speed failures: {failures}"

  def test_representative_cruise_grades(self):
    failures = []
    for target, grade in ((30, 0), (30, -3), (30, 3)):
      rows = q._run_cruise(target, grade)
      settled = rows[rows[:, 0].astype(float) >= q.RUN_DURATION_S - q.POST_TARGET_SETTLE_S]
      speed_error = float(np.max(np.abs(settled[:, 1].astype(float) - target)))
      gas_span = float(np.ptp(settled[:, 3].astype(float)))
      brake_span = float(np.ptp(settled[:, 4].astype(float)))
      transitions = int(settled[-1, 7]) - int(settled[0, 7])
      if speed_error > 1.0 or gas_span > 0.05 or brake_span > 0.05 or transitions > 2:
        failures.append({"target_mph": target, "grade_percent": grade,
                         "speed_error_mph": speed_error, "gas_span": gas_span,
                         "brake_span": brake_span, "transitions": transitions,
                         "jerk_p95": q._p95_jerk(settled)})
    assert not failures, f"CR-V subset cruise failures: {failures}"

  def test_representative_lead_recovery(self):
    failures = []
    for ego, closing, stopped in ((25, 10, False), (45, 20, False),
                                  (65, 10, False), (25, 15, True)):
      rows = q._run_lead_case(ego, closing, stopped)
      times = rows[:, 0].astype(float)
      gaps = rows[:, 3].astype(float)
      closest = int(np.argmin(gaps))
      recovery = np.flatnonzero((np.arange(len(rows)) > closest) &
                                (gaps >= q.TARGET_GAP - 0.25))
      recovery_i = int(recovery[0]) if len(recovery) else len(rows)
      recovery_gap = float(gaps[recovery_i]) if recovery_i < len(rows) else math.inf
      later_min = float(np.min(gaps[recovery_i + 1:])) if recovery_i + 1 < len(rows) else recovery_gap
      recovery_s = float(times[recovery_i] - times[closest]) if recovery_i < len(rows) else math.inf
      min_gap = float(gaps[closest])
      tail_start = max(times[recovery_i] if recovery_i < len(rows) else math.inf,
                       q.RUN_DURATION_S - q.POST_TARGET_SETTLE_S)
      tail = rows[times >= tail_start] if math.isfinite(tail_start) else rows[:0]
      tail_span = float(np.ptp(tail[:, 3].astype(float))) if len(tail) else math.inf
      tail_transitions = int(tail[-1, 11]) - int(tail[0, 11]) if len(tail) else math.inf
      tail_target_error = (float(np.max(np.abs(tail[:, 3].astype(float) - q.TARGET_GAP)))
                           if len(tail) else math.inf)
      settled = (stopped and tail_span <= 0.05
                 or not stopped and tail_span <= 0.05 and tail_target_error <= 0.25)
      if (min_gap < 1.0 or recovery_s > 15.0 or later_min < recovery_gap - 1e-3
          or len(tail) == 0 or tail[-1, 0] - tail[0, 0] < q.POST_TARGET_SETTLE_S
          or not settled or tail_transitions > 2):
        failures.append({"ego_mph": ego, "closing_mph": closing, "stopped": stopped,
                         "minimum_time_gap_s": min_gap, "recovery_s": recovery_s,
                         "recovery_gap_s": recovery_gap, "later_min_gap_s": later_min,
                         "settled_gap_span_s": tail_span, "settled_target_error_s": tail_target_error,
                         "settled_mode_transitions": tail_transitions})
    assert not failures, f"CR-V subset lead failures: {failures}"
