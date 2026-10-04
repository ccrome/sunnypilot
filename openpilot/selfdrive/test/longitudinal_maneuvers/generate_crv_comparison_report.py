#!/usr/bin/env python3
"""Render paired, immutable controller experiment results without rerunning them."""
# ruff: noqa: E501
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SIGNALS = {"time_s": "time", "ego_speed_mph": "ego", "lead_speed_mph": "lead",
           "acceleration_mps2": "accel", "planner_acceleration_mps2": "planner_accel",
           "controller_acceleration_mps2": "controller_accel", "jerk_mps3": "jerk",
           "observed_acceleration_mps2": "observed_accel", "observed_speed_mph": "observed_speed",
           "planner_jerk_mps3": "planner_jerk", "gas_command": "gas", "brake_intensity": "brake",
           "brake_request": "brake_request", "actuator_mode": "mode", "mode_transitions": "transitions",
           "gap_m": "gap", "time_gap_s": "time_gap", "safety_override": "safety",
           "predictive_brake_mps2": "predictive_brake"}
HTML = r'''<!doctype html><html><head><meta charset="utf-8">
<title>CR-V immutable baseline versus evaluated controller</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font:14px system-ui;margin:24px}select{font:16px system-ui;padding:8px}
table{border-collapse:collapse;margin:16px 0}td,th{padding:5px 12px;border:1px solid #ddd;text-align:left}</style>
</head><body><h1>Immutable baseline versus evaluated controller</h1>
<p id="revision"></p><p>Baseline: dashed. Evaluated controller: solid. Same inputs and vehicle response in each pair.
The baseline is the recorded source revision, not necessarily the version currently installed on comma4.
Actuator variants probe uncertainty; they are not independently calibrated vehicles. No automatic refresh.</p>
<select id="case"></select><table id="metrics"></table><div id="plot"></div><script>
const data=__DATA__, n=data.results.length/2, select=document.getElementById('case');
document.getElementById('revision').textContent=`Baseline ${data.baseline_commit.slice(0,12)}; candidate planner SHA-256 ${data.candidate_planner_sha256}; generated ${data.generated_at}`;
for(let i=0;i<n;i++){const r=data.results[i],o=document.createElement('option');o.value=i;o.textContent=`${r.response}: ${r.case.join(' / ')}`;select.appendChild(o)}
function render(){const i=Number(select.value), a=data.results[i],b=data.results[i+n], series=[];
const table=document.getElementById('metrics');table.replaceChildren();
const header=document.createElement('tr');for(const x of ['Metric','Baseline','Evaluated controller']){const c=document.createElement('th');c.textContent=x;header.appendChild(c)}table.appendChild(header);
if(a.passed!=null){const row=document.createElement('tr');for(const v of ['Regression',a.passed?'PASS':'FAIL',b.passed?'PASS':'FAIL']){const c=document.createElement('td');c.textContent=v;row.appendChild(c)}table.appendChild(row)}
for(const k of Object.keys(a.metrics)){const row=document.createElement('tr');for(const v of [k,a.metrics[k],b.metrics[k]]){const c=document.createElement('td');c.textContent=typeof v==='number'?v.toFixed(4):v;row.appendChild(c)}table.appendChild(row)}
for(const [j,r] of [[i,a],[i+n,b]]){const t=data.traces[j],name=r.variant==='installed'?'baseline':'evaluated',dash=name==='baseline'?'dash':'solid';
const add=(label,key,color,xaxis,yaxis)=>{if(t[key]!=null)series.push({x:t.time,y:t[key],name:`${name} ${label}`,type:'scatter',mode:'lines',line:{color,dash,width:1.8},xaxis,yaxis})};
add('ego mph','ego','#1769aa','x','y');add('planner accel','planner_accel','#7b1fa2','x2','y2');add('physical accel','accel','#00897b','x2','y2');
add('observed accel','observed_accel','#795548','x2','y2');add('Honda effort command','controller_accel','#9467bd','x2','y2');
add('safety','safety','#111','x2','y3');add('jerk','jerk','#ad1457','x3','y4');
add('gas','gas','#2e7d32','x4','y5');add('brake','brake','#c62828','x4','y5');add('gap m','gap','#ef6c00','x5','y6');add('time gap','time_gap','#1565c0','x5','y7');}
const t=data.traces[i];if(t.lead!==null)series.push({x:t.time,y:t.lead,name:'lead mph',type:'scatter',mode:'lines',line:{color:'#d35400'},xaxis:'x',yaxis:'y'});
Plotly.react('plot',series,{height:1120,hovermode:'x unified',margin:{l:70,r:90,t:45,b:45},legend:{orientation:'h'},
xaxis:{domain:[0,1],anchor:'y',title:'time (s)'},xaxis2:{anchor:'y2',matches:'x'},xaxis3:{anchor:'y4',matches:'x'},xaxis4:{anchor:'y5',matches:'x'},xaxis5:{anchor:'y6',matches:'x'},
yaxis:{domain:[.83,1],title:'speed (mph)'},yaxis2:{domain:[.64,.8],title:'acceleration (m/s²)'},yaxis3:{overlaying:'y2',side:'right',title:'safety',range:[0,1]},
yaxis4:{domain:[.45,.61],title:'jerk (m/s³)'},yaxis5:{domain:[.26,.42],title:'gas / brake'},yaxis6:{domain:[.04,.22],title:'gap (m)'},yaxis7:{overlaying:'y6',side:'right',title:'time gap (s)'}},{responsive:true,displaylogo:false});}
select.onchange=render;render();</script></body></html>'''


def render_report(data, output, plotly_js=None):
  from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_multiphase_braking import stop_failures
  for record in data["results"]:
    record["passed"] = not stop_failures(record["metrics"]) if record["case"][0] == "multiphase" else None
  document = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
  if plotly_js is not None:
    document = document.replace('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
                                "<script>" + plotly_js.read_text() + "</script>")
  output.write_text(document)
  records = []
  for record, trace in zip(data["results"], data["traces"], strict=True):
    signals = {key: trace.get(value) for key, value in SIGNALS.items()}
    signals["target_speed_mph"] = [trace["target"]] * len(trace["time"])
    records.append({"id": trace["label"], "kind": "Paired experiment", "case": trace["label"],
                    "passed": record["passed"], "metrics": record["metrics"], "signals": signals})
  metadata = {k: v for k, v in data.items() if k not in {"results", "traces"}}
  dashboard_file = output.with_name(output.stem + "_results.json")
  dashboard_file.write_text(json.dumps({**metadata, "results": records}, separators=(",", ":")))
  print(f"wrote {output} and {dashboard_file}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=Path)
  parser.add_argument("--output", type=Path, default=HERE / "crv_controller_comparison.html")
  parser.add_argument("--plotly-js", type=Path, help="Embed a local Plotly bundle for offline/self-contained charts")
  args = parser.parse_args()
  render_report(json.loads(args.input.read_text()), args.output, args.plotly_js)


if __name__ == "__main__":
  main()
