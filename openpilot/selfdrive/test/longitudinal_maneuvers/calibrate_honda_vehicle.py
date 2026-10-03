#!/usr/bin/env python3
"""Identify Honda CAN response on whole-drive training/validation splits.

Offline dependencies: pandas, scipy, numba. Never fit planner acceleration as
the vehicle input. Use actual powertrain CAN and preserve the 100 Hz speed
measurement stream so the production speed estimator can be replayed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares

from opendbc.can import CANParser
from opendbc.car import Bus
from opendbc.car.common.simple_kalman import get_kalman_gain
from opendbc.car.honda.values import CAR, DBC
from openpilot.tools.lib.logreader import LogReader


DT = 0.01
A = np.array([[1., DT], [0., 1.]])
K = get_kalman_gain(DT, A, np.array([[1., 0.]]), np.diag([0., 100.]), 0.3).ravel()


def extract_segment(spec):
  path, custom = spec
  parser = CANParser(DBC[CAR.HONDA_CRV_5G][Bus.pt], [("ACC_CONTROL", 0)], 1)
  rows = {"state": [], "command": [], "orientation": []}
  for m in LogReader(str(path), only_union_types=True):
    service = m.which()
    t = m.logMonoTime / 1e9
    if service == "carParams":
      if custom != bool(m.carParams.openpilotLongitudinalControl):
        raise ValueError(f"Longitudinal mode changed within {path}")
    elif service == "carState":
      s = m.carState
      rows["state"].append((t, s.vEgoRaw, s.vEgo, s.aEgo, s.gasPressed, s.brakePressed, s.cruiseState.enabled))
    elif service == "carControl":
      pitch = m.carControl.orientationNED
      rows["orientation"].append((t, float(pitch[1]) if len(pitch) == 3 else 0.))
    elif service in ("can", "sendcan") and ((service == "sendcan") == custom):
      frames = [(c.address, bytes(c.dat), c.src) for c in getattr(m, service) if c.address == 0x1df]
      if frames and 0x1df in parser.update([(m.logMonoTime, frames)]):
        c = parser.vl["ACC_CONTROL"]
        rows["command"].append((t, c["ACCEL_COMMAND"], max(0., c["GAS_COMMAND"]), c["BRAKE_REQUEST"], c["CONTROL_ON"]))
  return rows


def extract_route(route, log_root, out, workers, refresh=False):
  path = out / f"{route}.parquet"
  if path.exists() and not refresh:
    return pd.read_parquet(path)
  files = sorted(log_root.glob(f"{route}--*/rlog.zst"), key=lambda p: int(p.parent.name.rsplit("--", 1)[1]))
  if not files:
    raise FileNotFoundError(route)
  custom = next(bool(m.carParams.openpilotLongitudinalControl) for m in LogReader(str(files[0]), only_union_types=True)
                if m.which() == "carParams")
  with ProcessPoolExecutor(max_workers=workers) as pool:
    parts = list(pool.map(extract_segment, [(p, custom) for p in files]))
  names = {"state": ["t", "raw", "speed", "acceleration", "gas_pressed", "brake_pressed", "enabled"],
           "command": ["t", "accel", "gas", "braking", "on"], "orientation": ["t", "pitch"]}
  frames = {k: pd.DataFrame([row for part in parts for row in part[k]], columns=v).sort_values("t")
            for k, v in names.items()}
  d = frames["state"].drop_duplicates("t")
  for k in ("command", "orientation"):
    d = pd.merge_asof(d, frames[k], on="t", direction="backward", tolerance=0.15)
  d.to_parquet(path, index=False)
  return d


@njit(cache=True)
def estimator_replay(raw, initial_speed, initial_accel, k):
  result = np.empty((len(raw), 2))
  v, a = initial_speed, initial_accel
  for n in range(len(raw)):
    error = raw[n] - v
    v, a = v + DT * a + k[0] * error, a + k[1] * error
    result[n] = v, a
  return result


def observer_metrics(d):
  # Initial observer state is unknown. Discard its first second rather than
  # resetting repeatedly to the measured acceleration (which hides errors).
  replay = estimator_replay(d.raw.to_numpy(), d.speed.iloc[0], d.acceleration.iloc[0], K)
  error = replay[100:] - d[["speed", "acceleration"]].to_numpy()[100:]
  valid = d.t.diff().between(.007, .013).to_numpy()[100:]
  return {"speed_rmse": float(np.sqrt(np.mean(error[valid, 0] ** 2))),
          "acceleration_rmse": float(np.sqrt(np.mean(error[valid, 1] ** 2))),
          "raw_repeat_fraction": float(np.mean(np.diff(d.raw.to_numpy()) == 0.))}


def windows(d, seconds=5.):
  valid = d.enabled.eq(True) & d.on.gt(0) & ~d.gas_pressed.eq(True) & ~d.brake_pressed.eq(True) & d.speed.gt(2)
  valid &= d[["accel", "gas", "braking", "pitch"]].notna().all(axis=1)
  indices = np.flatnonzero(valid)
  groups = np.split(indices, np.flatnonzero((np.diff(indices) != 1) | (np.diff(d.t.to_numpy()[indices]) > .025)) + 1)
  result = []
  # Five-second free rollouts, not one-step teacher-forced prediction. Use
  # a one-second history to initialize actuator state; score the last four.
  count = round(seconds / DT)
  for group in groups:
    for begin in range(0, len(group) - count + 1, count):
      x = d.iloc[group[begin:begin + count]].copy()
      if x.t.diff().dropna().between(.005, .015).mean() < .98:
        continue
      result.append(x[["raw", "speed", "acceleration", "accel", "gas", "braking", "pitch"]].to_numpy(float))
  return np.array(result)


@njit(cache=True)
def predict(w, p, kind, k):
  # Independent drive/brake channels retain their own physical state on
  # release. They cannot instantaneously turn residual engine effort into
  # brake effort just because the command mode changed.
  gas_gain, brake_gain, gas_tau, release_tau, brake_tau, delay, load, drag = p
  n_delay = int(round(delay / DT))
  predictions = np.empty((len(w), w.shape[1], 2))
  for window in range(len(w)):
    x = w[window]
    v = x[0, 0]
    estimated_v, estimated_a = x[0, 1], x[0, 2]
    resistance = load + drag * v * v + 9.81 * np.sin(x[0, 6])
    drive = max(0., estimated_a + resistance)
    brake = min(0., estimated_a + resistance)
    for i in range(len(x)):
      c = x[max(0, i - n_delay)]
      gas = c[4] / 1600. * 2.2 if kind == 0 else c[3] if c[4] > 0 else 0.
      gas = max(0., gas) * (1. - c[5])
      gas_target = gas_gain * gas
      brake_target = brake_gain * min(0., c[3]) * c[5]
      tau = gas_tau if gas_target > drive else release_tau
      drive += (1. - np.exp(-DT / tau)) * (gas_target - drive)
      brake += (1. - np.exp(-DT / brake_tau)) * (brake_target - brake)
      acceleration = drive + brake - load - drag * v * v - 9.81 * np.sin(x[i, 6])
      v = max(0., v + DT * acceleration)
      error = v - estimated_v
      estimated_v, estimated_a = estimated_v + DT * estimated_a + k[0] * error, estimated_a + k[1] * error
      predictions[window, i] = estimated_v, estimated_a
  return predictions


def errors(p, w, kind):
  prediction = predict(w, p, kind, K)
  # Score only after history initialization. Speed and acceleration have
  # separate observable roles, with speed error weighted over a one-second
  # equivalent interval. No fitted per-window biases or state resets.
  residual = prediction[:, 100:, :] - w[:, 100:, 1:3]
  return residual[:, ::5].ravel()


def metrics(w, p, kind):
  e = predict(w, p, kind, K)[:, 100:] - w[:, 100:, 1:3]
  return {"windows": len(w), "speed_rmse_mps": float(np.sqrt(np.mean(e[:, :, 0] ** 2))),
          "acceleration_rmse_mps2": float(np.sqrt(np.mean(e[:, :, 1] ** 2)))}


@njit(cache=True)
def predict_previous(w, k):
  """Exact previous single-state physics, with the observer added fairly."""
  result = np.empty((len(w), w.shape[1], 2))
  for n in range(len(w)):
    x = w[n]
    v, ev, ea = x[0, 0], x[0, 1], x[0, 2]
    effort = ea + .012 + .000046064 * v * v + 9.81 * np.sin(x[0, 6])
    for i in range(len(x)):
      c = x[max(0, i - 30)]
      drive = 1.0179 * min(c[3], 0.) if c[5] else 1.3003 * c[4] / 1600. * 2.2
      tau = .5189 if c[5] else 1.1018
      effort += (1. - np.exp(-DT / tau)) * (drive - effort)
      a = effort - .012 - .000046064 * v * v - 9.81 * np.sin(x[i, 6])
      v = max(0., v + DT * a)
      error = v - ev
      ev, ea = ev + DT * ea + k[0] * error, ea + k[1] * error
      result[n, i] = ev, ea
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", nargs="+", required=True)
  parser.add_argument("--validate", nargs="+", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--log-root", type=Path, default=Path("/srv/storage/openpilot/route-data"))
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument("--refresh", action="store_true", help="Regenerate this tool's extracted caches")
  args = parser.parse_args()
  if set(args.train) & set(args.validate):
    raise ValueError("A whole route cannot be in both training and validation")
  args.output.mkdir(parents=True, exist_ok=True)
  data = {}
  for route in args.train + args.validate:
    print("Extracting", route, flush=True)
    data[route] = extract_route(route, args.log_root, args.output, args.workers, args.refresh)
  observer = {r: observer_metrics(d) for r, d in data.items()}
  print("Production estimator replay:", observer, flush=True)
  ws = {r: windows(d) for r, d in data.items()}
  # Equal route weighting prevents a long stock highway drive from drowning
  # out short custom-command excursions. Deterministic subsampling, no random
  # selection of easy windows.
  count = min(len(ws[r]) for r in args.train if len(ws[r]))
  training = np.concatenate([ws[r][np.linspace(0, len(ws[r]) - 1, count).astype(int)] for r in args.train if len(ws[r])])
  old = {}
  for r, w in ws.items():
    if len(w):
      e = predict_previous(w, K)[:, 100:] - w[:, 100:, 1:3]
      old[r] = {"windows": len(w), "speed_rmse_mps": float(np.sqrt(np.mean(e[:, :, 0] ** 2))),
                "acceleration_rmse_mps2": float(np.sqrt(np.mean(e[:, :, 1] ** 2)))}
  report = {"train_routes": args.train, "validation_routes": args.validate, "observer_replay": observer,
            "previous_model": old, "models": {}}
  initial = [1.3, 1., .4, .15, .3, .15, .1, .0001]
  lower = [.1, .1, .02, .02, .02, 0., 0., 0.]
  upper = [3., 3., 2., 2., 2., .8, 1., .005]
  # Delay is discontinuous at the 100 Hz transport boundary. Profile it
  # explicitly instead of asking a finite-difference optimizer to discover it.
  for kind, name in enumerate(("gas_effort", "acceleration_request")):
    best = None
    for delay in (0., .05, .1, .15, .2, .3, .4):
      indices = np.array([0, 1, 2, 3, 4, 6, 7])
      def residual(values, indices=indices, delay=delay, kind=kind):
        p = np.array(initial)
        p[indices] = values
        p[5] = delay
        return errors(p, training, kind)
      fit = least_squares(residual, np.array(initial)[indices], bounds=(np.array(lower)[indices], np.array(upper)[indices]),
                          loss="soft_l1", f_scale=.2, max_nfev=100)
      p = np.array(initial)
      p[indices] = fit.x
      p[5] = delay
      if best is None or fit.cost < best[0]:
        best = fit.cost, p, fit.success
      print(name, "delay", delay, "cost", round(fit.cost, 3), flush=True)
    cost, p, success = best
    names = ("gas_gain", "brake_gain", "gas_tau", "gas_release_tau", "brake_tau", "delay", "rolling", "drag")
    result = {"parameters": dict(zip(names, p.tolist(), strict=True)),
              "training_cost": cost, "optimizer_success": bool(success),
              "routes": {r: metrics(w, p, kind) for r, w in ws.items() if len(w)}}
    report["models"][name] = result
    print(name, json.dumps(result), flush=True)
  (args.output / "calibration.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
