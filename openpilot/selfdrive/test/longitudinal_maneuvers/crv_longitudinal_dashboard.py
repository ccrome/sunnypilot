#!/usr/bin/env python3
"""Live Plotly Dash viewer for CR-V longitudinal regression artifacts.

The dashboard watches this directory for JSON result artifacts.  A result file
may contain either ``{"results": [...]}`` records or the older
``{"traces": [...]}`` representation.  This keeps the viewer useful for the
full matrix, representative traces, and future regression suites without
hard-coding their case lists.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from dash import Dash, Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots
import plotly.graph_objects as go


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE
RESULT_SUFFIX = "results.json"


def _json_default(value):
  if hasattr(value, "item"):
    return value.item()
  raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _as_float_list(values):
  if values is None:
    return None
  result = []
  for value in values:
    if value is None:
      result.append(None)
    else:
      result.append(float(value))
  return result


def _trace_to_result(trace):
  signals = {
    "time_s": trace.get("time"),
    "ego_speed_mph": trace.get("ego"),
    "lead_speed_mph": trace.get("lead"),
    "target_speed_mph": ([trace["target"]] * len(trace["time"])
                          if trace.get("target") is not None else None),
    "gap_m": trace.get("gap"),
    "time_gap_s": trace.get("time_gap"),
    "acceleration_mps2": trace.get("accel"),
    "observed_acceleration_mps2": trace.get("observed_accel"),
    "planner_acceleration_mps2": trace.get("planner_accel"),
    "predictive_brake_mps2": trace.get("predictive_brake"),
    "safety_override": trace.get("safety"),
    "jerk_mps3": trace.get("jerk"),
    "planner_jerk_mps3": trace.get("planner_jerk"),
    "gas_command": trace.get("gas"),
    "brake_intensity": trace.get("brake"),
    "brake_request": trace.get("brake_request"),
    "actuator_mode": trace.get("mode"),
    "mode_transitions": trace.get("transitions"),
  }
  return {
    "id": trace.get("label", "trace"),
    "kind": "Trace",
    "case": trace.get("label", "trace"),
    "passed": None,
    "metrics": {"grade_percent": trace.get("grade")} if trace.get("grade") is not None else {},
    "signals": signals,
  }


def _normalize_payload(payload, source):
  if isinstance(payload, dict) and isinstance(payload.get("results"), list):
    records = payload["results"]
    metadata = {key: value for key, value in payload.items() if key != "results"}
  elif isinstance(payload, dict) and isinstance(payload.get("traces"), list):
    records = [_trace_to_result(trace) for trace in payload["traces"]]
    metadata = {key: value for key, value in payload.items() if key != "traces"}
  elif isinstance(payload, list):
    records = payload
    metadata = {}
  else:
    return None

  normalized = []
  for index, record in enumerate(records):
    if not isinstance(record, dict):
      continue
    item = dict(record)
    item.setdefault("id", f"{source.name}:{index}")
    item.setdefault("kind", "Regression")
    item.setdefault("case", item["id"])
    item.setdefault("passed", None)
    item.setdefault("metrics", {})
    item.setdefault("signals", {})
    item["source"] = source.name
    item["signals"] = {key: _as_float_list(value) if key not in {"actuator_mode"} else value
                        for key, value in item["signals"].items()}
    normalized.append(item)
  if not normalized:
    return None
  return {"source": source.name, "metadata": metadata, "records": normalized}


def discover_artifacts(results_dir=DEFAULT_RESULTS_DIR):
  artifacts = []
  for path in sorted(results_dir.glob(f"*{RESULT_SUFFIX}")):
    try:
      payload = json.loads(path.read_text())
      normalized = _normalize_payload(payload, path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
      continue
    if normalized is not None:
      artifacts.append(normalized)
  return artifacts


def _flatten_records(artifacts):
  return [record for artifact in artifacts for record in artifact["records"]]


def _format_value(value):
  if isinstance(value, float):
    if not math.isfinite(value):
      return str(value)
    return f"{value:.6g}"
  if value is None:
    return ""
  return str(value)


def _summary_figure(records):
  passed = sum(record["passed"] is True for record in records)
  failed = sum(record["passed"] is False for record in records)
  unknown = len(records) - passed - failed
  figure = go.Figure()
  figure.add_annotation(text=f"{passed} pass · {failed} fail · {unknown} un-gated / trace-only",
                        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                        font={"size": 24})
  figure.update_layout(height=130, margin={"l": 20, "r": 20, "t": 15, "b": 15},
                       xaxis={"visible": False}, yaxis={"visible": False})
  return figure


def _trace_figure(record):
  signals = record.get("signals", {})
  time = signals.get("time_s")
  if not time:
    figure = go.Figure()
    figure.add_annotation(text="This result has diagnostics only; no time-series signals were provided.",
                          x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return figure

  figure = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                         subplot_titles=("speed", "acceleration / safety", "jerk",
                                         "gas / brake / mode", "gap"))

  def add(name, key, row, color, dash="solid", secondary=False):
    values = signals.get(key)
    if values is None:
      return
    if len(values) != len(time):
      return
    if values and isinstance(values[0], str):
      mapping = {"gas": 1.0, "coast": 0.0, "brake": -1.0}
      values = [mapping.get(value, None) for value in values]
    figure.add_trace(go.Scatter(x=time, y=values, name=name, mode="lines",
                                line={"color": color, "dash": dash, "width": 1.6}),
                     row=row, col=1, secondary_y=secondary)

  add("ego mph", "ego_speed_mph", 1, "#1769aa")
  add("lead mph", "lead_speed_mph", 1, "#d35400")
  add("target mph", "target_speed_mph", 1, "#777", "dot")
  add("physical m/s²", "acceleration_mps2", 2, "#00897b")
  add("openpilot observed m/s²", "observed_acceleration_mps2", 2, "#795548", "dot")
  add("planner m/s²", "planner_acceleration_mps2", 2, "#7b1fa2", "dot")
  add("predictive brake", "predictive_brake_mps2", 2, "#c62828", "dot")
  add("safety override", "safety_override", 2, "#111", "dash")
  add("physical jerk", "jerk_mps3", 3, "#ad1457")
  add("planner jerk", "planner_jerk_mps3", 3, "#7b1fa2", "dot")
  add("gas", "gas_command", 4, "#2e7d32")
  add("brake intensity", "brake_intensity", 4, "#8e24aa")
  add("brake request", "brake_request", 4, "#c62828")
  add("mode", "actuator_mode", 4, "#455a64", "dash")
  add("gap m", "gap_m", 5, "#ef6c00")
  add("time gap s", "time_gap_s", 5, "#1565c0")
  figure.update_layout(height=1000, hovermode="x unified", title=record.get("case", record["id"]),
                       margin={"l": 65, "r": 25, "t": 70, "b": 40},
                       legend={"orientation": "h", "y": 1.02, "yanchor": "bottom"})
  figure.update_xaxes(title_text="time (s)", row=5, col=1)
  return figure


def create_app(results_dir=DEFAULT_RESULTS_DIR):
  app = Dash(__name__)
  app.title = "CR-V longitudinal regressions"
  app.layout = html.Div([
    html.H1("CR-V longitudinal regression dashboard"),
    html.P("Results are read dynamically from JSON artifacts in " + str(results_dir)),
    html.Div([
      html.Button("Refresh results", id="refresh", n_clicks=0),
      dcc.Dropdown(id="artifact", clearable=False, placeholder="Select a result artifact"),
      dcc.Dropdown(id="case", clearable=False, placeholder="Select a case"),
    ], className="controls"),
    dcc.Store(id="artifact-store"),
    dcc.Graph(id="summary", config={"displaylogo": False}),
    html.Div(id="case-metadata", className="metadata"),
    dcc.Graph(id="trace", config={"displaylogo": False}),
    dash_table.DataTable(id="diagnostics", page_size=25, sort_action="native",
                         filter_action="native", style_table={"overflowX": "auto"},
                         style_cell={"textAlign": "left", "padding": "6px"}),
  ], style={"fontFamily": "system-ui", "margin": "24px"})

  @app.callback(
    Output("artifact-store", "data"),
    Output("artifact", "options"),
    Output("artifact", "value"),
    Input("refresh", "n_clicks"),
    State("artifact", "value"),
  )
  def refresh_artifacts(_clicks, current_source):
    artifacts = discover_artifacts(results_dir)
    options = [{"label": artifact["source"], "value": artifact["source"]} for artifact in artifacts]
    available = {option["value"] for option in options}
    value = current_source if current_source in available else options[0]["value"] if options else None
    return artifacts, options, value

  @app.callback(
    Output("case", "options"),
    Output("case", "value"),
    Input("artifact-store", "data"),
    Input("artifact", "value"),
    State("case", "value"),
  )
  def update_cases(artifacts, source, current_case):
    artifact = next((item for item in (artifacts or []) if item["source"] == source), None)
    records = artifact["records"] if artifact else []
    options = [{"label": record["case"], "value": record["id"]} for record in records]
    available = {option["value"] for option in options}
    value = current_case if current_case in available else options[0]["value"] if options else None
    return options, value

  @app.callback(
    Output("summary", "figure"),
    Output("trace", "figure"),
    Output("case-metadata", "children"),
    Output("diagnostics", "data"),
    Output("diagnostics", "columns"),
    Input("artifact-store", "data"),
    Input("artifact", "value"),
    Input("case", "value"),
  )
  def update_view(artifacts, source, case_id):
    records = _flatten_records(artifacts or [])
    artifact = next((item for item in (artifacts or []) if item["source"] == source), None)
    selected = next((record for record in (artifact["records"] if artifact else [])
                     if record["id"] == case_id), None)
    metadata = ""
    if selected:
      status = "PASS" if selected["passed"] is True else "FAIL" if selected["passed"] is False else "TRACE"
      metrics = " · ".join(f"{key}={_format_value(value)}" for key, value in selected["metrics"].items())
      metadata = html.Div([html.B(f"{status}: {selected['case']}"), html.Br(), metrics])
    table_rows = []
    for record in records:
      row = {"source": record["source"], "kind": record["kind"], "case": record["case"],
             "status": "PASS" if record["passed"] is True else "FAIL" if record["passed"] is False else "TRACE"}
      row.update({key: _format_value(value) for key, value in record["metrics"].items()})
      table_rows.append(row)
    columns = [{"name": key, "id": key} for key in sorted({key for row in table_rows for key in row})]
    return _summary_figure(records), _trace_figure(selected or {"id": "none", "case": "No case selected", "signals": {}}), metadata, table_rows, columns

  return app


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8050)
  parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
  args = parser.parse_args()
  create_app(args.results_dir).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
  main()
