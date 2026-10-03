#!/usr/bin/env python3
"""Paired controller comparisons against an immutable installed source revision.

The baseline planner is loaded from Git, not approximated by another control
law. Other production modules must match the baseline before comparison.
Actuator variants are uncertainty probes, not claims of calibrated hardware.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics


BASELINE = "d896499f511aa57806ed3c81ea6caf30da2b961c"
PLANNER = "openpilot/selfdrive/controls/lib/longitudinal_planner.py"
INNER = "openpilot/selfdrive/controls/lib/longcontrol.py"
CASES = [("speed", 0, 15), ("speed", 90, 15),
         ("lead", 25, 10, False), ("lead", 45, 20, False), ("lead", 65, 10, False),
         ("lead", 25, 20, False), ("lead", 45, 40, False), ("lead", 25, 25, True),
         ("rolling", 35), ("rolling", 50), ("deviation", 45, 35, 1), ("far", 45)]
DYNAMICS = {"nominal": HondaDynamics(),
            "slower": HondaDynamics(delay=0.8, gas_tau=1.5, brake_tau=0.7),
            "faster": HondaDynamics(delay=0.1, gas_tau=0.7, brake_tau=0.3),
            "low_authority": HondaDynamics(gas_gain=1.3, brake_gain=.8, gas_tau=.8, gas_release_tau=.2,
                                           delay=.2, speed_noise_std=.01, seed=1),
            "high_authority": HondaDynamics(gas_gain=1.9, brake_gain=1.1, gas_tau=.4, gas_release_tau=.08,
                                            speed_noise_std=.01, seed=2)}


def _run(spec):
  variant, response, case = spec
  import openpilot.selfdrive.controls.lib.longitudinal_planner as planner
  import openpilot.selfdrive.controls.lib.longcontrol as inner
  if variant == "installed":
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{PLANNER}"])
    exec(compile(source, planner.__file__, "exec"), planner.__dict__)
  else:
    # A worker can previously have run the baseline; restore current code.
    source = Path(planner.__file__).read_bytes()
    exec(compile(source, planner.__file__, "exec"), planner.__dict__)
  source = (subprocess.check_output(["git", "show", f"{BASELINE}:{INNER}"]) if variant == "installed"
            else Path(inner.__file__).read_bytes())
  exec(compile(source, inner.__file__, "exec"), inner.__dict__)
  from openpilot.selfdrive.test.longitudinal_maneuvers import honda_vehicle
  honda_vehicle.LongControl = inner.LongControl
  from openpilot.selfdrive.test.longitudinal_maneuvers import plant, test_crv_longitudinal_quality_regression as q
  plant.LongitudinalPlanner = planner.LongitudinalPlanner
  p = DYNAMICS[response]
  kind, *args = case
  runners = {"speed": q._run_speed_transition, "lead": q._run_lead_case,
             "rolling": q._run_rolling_lead_stop_case, "deviation": q._run_lead_speed_deviation_case,
             "far": q._run_far_lead_case}
  rows = runners[kind](*args, actuator_parameters=p)
  speed_case = kind == "speed"
  t = rows[:, 0].astype(float)
  v = rows[:, 1].astype(float)
  accel = rows[:, 2 if speed_case else 4].astype(float)
  gas = rows[:, 3 if speed_case else 7].astype(float)
  brake = rows[:, 5 if speed_case else 8].astype(float)
  modes = rows[:, 6 if speed_case else 10]
  active = (t >= q.LEAD_IN_S) & (t < q.LEAD_IN_S + 60)
  tail = t >= t[-1] - q.POST_TARGET_SETTLE_S
  jerk = np.gradient(accel, q.DT)
  noncoast = modes[active & (modes != "coast")]
  reversals = int(np.count_nonzero(noncoast[1:] != noncoast[:-1]))
  metrics = {"jerk_p95": float(np.percentile(np.abs(jerk[active]), 95)),
             "jerk_peak": float(np.max(np.abs(jerk[active]))), "mode_reversals": reversals,
             "gas_variation": float(np.sum(np.abs(np.diff(gas[active])))),
             "brake_variation": float(np.sum(np.abs(np.diff(brake[active])))),
             "tail_gas_span": float(np.ptp(gas[tail])), "tail_brake_span": float(np.ptp(brake[tail]))}
  if not speed_case:
    gap = rows[:, 2].astype(float)
    lead = rows[:, 5].astype(float)
    metrics.update(minimum_clearance_m=float(np.min(gap)),
                   minimum_moving_time_gap_s=float(np.min(rows[active & (v > 1), 3].astype(float))),
                   safety_duration_s=float(np.sum(rows[:, 13].astype(bool)) * q.DT),
                   undershoot_lead_mph=float(max(0, np.max((lead - v)[active]))),
                   final_speed_mph=float(v[-1]), final_clearance_m=float(gap[-1]))
  else:
    reached = np.flatnonzero((t >= q.LEAD_IN_S) & (np.abs(v - args[1]) <= 1.0))
    metrics.update(overshoot_mph=float(max(0, np.max(v[active]) - args[1])) if args[1] > args[0]
                   else float(max(0, args[1] - np.min(v[active]))),
                   settled_speed_error_mph=float(np.max(np.abs(v[tail] - args[1]))),
                   target_arrival_s=float(t[reached[0]] - q.LEAD_IN_S) if len(reached) else None)
  trace = {"label": f"{variant} / {response} / {case}", "time": t.tolist(), "ego": v.tolist(),
           "accel": accel.tolist(), "jerk": jerk.tolist(), "gas": gas.tolist(), "brake": brake.tolist(),
           "mode": modes.tolist(), "controller_accel": rows[:, -1].astype(float).tolist(),
           "planner_accel": rows[:, 8 if speed_case else 6].astype(float).tolist(),
           "observed_speed": rows[:, -3].astype(float).tolist(), "observed_accel": rows[:, -2].astype(float).tolist(),
           "lead": None if speed_case else rows[:, 5].astype(float).tolist(),
           "gap": None if speed_case else rows[:, 2].astype(float).tolist(),
           "time_gap": None if speed_case else rows[:, 3].astype(float).tolist(),
           "safety": None if speed_case else rows[:, 13].astype(float).tolist(),
           "target": args[1] if speed_case else args[0] + 10 if kind == "far" else args[0], "grade": None,
           "brake_request": rows[:, 4 if speed_case else 9].astype(float).tolist(),
           "transitions": rows[:, 7 if speed_case else 11].astype(float).tolist(),
           "predictive_brake": None if speed_case else rows[:, 12].astype(float).tolist(),
           "planner_jerk": np.gradient(rows[:, 8 if speed_case else 6].astype(float), q.DT).tolist()}
  return {"variant": variant, "response": response, "case": list(case), "metrics": metrics, "trace": trace}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=32)
  parser.add_argument("--responses", nargs="+", choices=list(DYNAMICS), default=list(DYNAMICS))
  parser.add_argument("--case-kind", choices=["speed", "lead", "rolling", "deviation", "far"])
  args = parser.parse_args()
  # Replay the immutable planner + inner loop. Honda CAN must remain identical.
  tracked = ["opendbc_repo"]
  subprocess.run(["git", "diff", "--exit-code", BASELINE, "--", *tracked], check=True)
  import openpilot.selfdrive.controls.lib.longitudinal_planner as planner
  import openpilot.selfdrive.controls.lib.longcontrol as inner
  current_source = Path(planner.__file__).read_bytes()
  cases = [case for case in CASES if args.case_kind is None or case[0] == args.case_kind]
  specs = [(variant, response, case) for variant in ("installed", "candidate")
           for response in args.responses for case in cases]
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    results = list(pool.map(_run, specs))
  data = {"schema_version": 1, "generated_at": datetime.now(UTC).isoformat(), "baseline_commit": BASELINE,
          "candidate_planner_sha256": hashlib.sha256(current_source).hexdigest(),
          "candidate_inner_sha256": hashlib.sha256(Path(inner.__file__).read_bytes()).hexdigest(),
          "dynamics": {k: asdict(v) for k, v in DYNAMICS.items()},
          "results": [{k: v for k, v in r.items() if k != "trace"} for r in results],
          "traces": [r["trace"] for r in results]}
  args.output.write_text(json.dumps(data, separators=(",", ":")))
  for a, b in zip(results[:len(results) // 2], results[len(results) // 2:], strict=True):
    print(a["response"], a["case"], "installed", a["metrics"], "candidate", b["metrics"], flush=True)


if __name__ == "__main__":
  main()
