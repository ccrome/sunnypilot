#!/usr/bin/env python3
"""Show measured Honda response versus free-rollout vehicle models in Plotly."""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openpilot.selfdrive.test.longitudinal_maneuvers.calibrate_honda_vehicle import K, predict, predict_previous, windows
from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory", type=Path)
  parser.add_argument("--provenance", type=Path)
  args = parser.parse_args()
  report = json.loads((args.directory / "calibration.json").read_text())
  model = report["models"]["gas_effort"]
  p = np.array(list(model["parameters"].values()))
  figures, labels = [], []
  for route in report["train_routes"] + report["validation_routes"]:
    w = windows(pd.read_parquet(args.directory / f"{route}.parquet"))
    new, old = predict(w, p, 0, K), predict_previous(w, K)
    error = np.mean((new[:, 100:, 1] - w[:, 100:, 2]) ** 2, axis=1)
    # Representative windows plus the worst predicted acceleration window;
    # never choose only attractive fits.
    selected = sorted(set(np.linspace(0, len(w) - 1, min(12, len(w))).astype(int)) | {int(np.argmax(error))})
    for index in selected:
      x = w[index]
      label = f"{route} / window {index} / {'held out' if route in report['validation_routes'] else 'training'}"
      labels.append(label)
      t = np.arange(len(x)) * .01
      f = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Speed", "Observed acceleration", "Honda CAN commands", "Recorded pitch"))
      for values, name, panel in [(x[:, 1] * 2.23694, "Measured mph", 1),
                                  (old[index, :, 0] * 2.23694, "Previous model mph", 1),
                                  (new[index, :, 0] * 2.23694, "Corrected model mph", 1),
                                  (x[:, 2], "Measured aEgo", 2), (old[index, :, 1], "Previous model aEgo", 2),
                                  (new[index, :, 1], "Corrected model aEgo", 2),
                                  (x[:, 3], "ACCEL_COMMAND", 3), (x[:, 4] / 1600, "Gas / full scale", 3),
                                  (x[:, 5], "Brake request", 3), (x[:, 6], "Pitch radians", 4)]:
        f.add_trace(go.Scatter(x=t, y=values, name=name, mode="lines"), row=panel, col=1)
      f.update_layout(height=950, hovermode="x unified", title=label)
      f.add_vline(x=1., line_dash="dot")
      figures.append(json.loads(f.to_json()))
  rows = []
  for route, values in model["routes"].items():
    previous = report["previous_model"][route]
    rows.append({"route": route, "split": "held out" if route in report["validation_routes"] else "training",
                 "windows": values["windows"], "old speed RMSE m/s": previous["speed_rmse_mps"],
                 "new speed RMSE m/s": values["speed_rmse_mps"],
                 "old acceleration RMSE m/s²": previous["acceleration_rmse_mps2"],
                 "new acceleration RMSE m/s²": values["acceleration_rmse_mps2"]})
  table = pd.DataFrame(rows).to_html(index=False, float_format=lambda x: f"{x:.3f}")
  options = ''.join(f'<option value="{i}">{html.escape(label)}</option>' for i, label in enumerate(labels))
  page = ''.join(['<!doctype html><meta charset="utf-8"><title>Honda vehicle calibration</title>',
                  '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
                  '<style>body{font:15px system-ui;margin:24px}td,th{padding:6px}select{font:16px system-ui}</style>',
                  '<h1>Honda CAN-input vehicle calibration</h1><p>Five-second free rollouts with actual CAN inputs and recorded pitch. ',
                  'First second initializes the response; remaining four seconds are scored. Entire routes are held out. ',
                  'Includes representative windows and each route’s worst acceleration fit. ',
                  'This is a grey-box model, not proof of complete Honda ECU behavior.</p>', table,
                  '<p>The production speed-estimator replay agrees with recorded aEgo within numerical precision. ',
                  'Closed-loop tests additionally use 50 Hz sample-and-hold, wheel-speed quantization, and a seeded ',
                  '0.006 m/s disturbance proxy. Its exact noise law is not identified.</p>',
                  f'<select id="case">{options}</select><div id="plot"></div><script>',
                  'const figures=', json.dumps(figures, separators=(",", ":")), ';',
                  'const select=document.getElementById("case");function draw(){let f=figures[Number(select.value)];',
                  'Plotly.react("plot",f.data,f.layout,{responsive:true,displaylogo:false})}select.onchange=draw;draw();</script>'])
  (args.directory / "report.html").write_text(page)
  if args.provenance:
    report["simulation_dynamics"] = asdict(HondaDynamics())
    report["limitations"] = ["Grey-box dynamics, not a Honda ECU emulator", "Five-second rollout validation",
                             "Pitch calibration and unobserved load remain uncertainties",
                             "Noise distribution is a deterministic uncertainty proxy, not uniquely identified"]
    args.provenance.write_text(json.dumps(report, indent=2))
  print(args.directory / "report.html")


if __name__ == "__main__":
  main()
