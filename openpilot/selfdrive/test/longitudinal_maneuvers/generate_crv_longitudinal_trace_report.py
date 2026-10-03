#!/usr/bin/env python3
"""Generate an interactive HTML trace viewer for CR-V longitudinal cases."""
# ruff: noqa: E501

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from openpilot.selfdrive.test.longitudinal_maneuvers import test_crv_longitudinal_quality_regression as q


OUT = Path(__file__).with_name("crv_longitudinal_trace_report.html")
OUT_JSON = Path(__file__).with_name("crv_longitudinal_trace_results.json")
CASES = (
  ("speed", 0, 15), ("speed", 0, 45), ("speed", 0, 75),
  ("speed", 90, 45), ("speed", 90, 15),
  ("cruise", 30, 0), ("cruise", 30, -3), ("cruise", 30, 3),
  ("lead", 25, 10, False), ("lead", 45, 20, False),
  ("lead", 65, 10, False), ("lead", 25, 15, True),
  ("rolling", 35), ("deviation", 45, 35, 1), ("far", 45),
)


def _trace(spec):
  kind = spec[0]
  if kind == "speed":
    start, target = spec[1:]
    rows = q._run_speed_transition(start, target)
    label = f"Speed: {start} → {target} mph"
    values = (rows[:, 0], rows[:, 1], None, target, None, None, None,
              rows[:, 2], rows[:, 8], rows[:, 3], rows[:, 5], rows[:, 4], rows[:, 6], rows[:, 7], None, None)
  elif kind == "cruise":
    target, grade = spec[1:]
    rows = q._run_cruise(target, grade)
    label = f"Cruise: {target} mph at {grade:+d}%"
    values = (rows[:, 0], rows[:, 1], None, target, grade, None, None,
              rows[:, 8], rows[:, 9], rows[:, 3], rows[:, 4], rows[:, 5], rows[:, 6], rows[:, 7], None, None)
  elif kind in {"lead", "deviation", "far", "rolling"}:
    if kind == "lead":
      ego, closing, stopped = spec[1:]
      rows = q._run_lead_case(ego, closing, stopped)
      label = f"Lead: {ego} mph, {closing} mph closing, {'stopped' if stopped else 'moving'}"
    elif kind == "rolling":
      ego = spec[1]
      rows = q._run_rolling_lead_stop_case(ego)
      label = f"Rolling lead stop: {ego} mph ego"
    elif kind == "deviation":
      ego, lead, seed = spec[1:]
      rows = q._run_lead_speed_deviation_case(ego, lead, seed)
      label = f"Lead deviations: {ego} mph ego / {lead} mph nominal / seed {seed}"
    else:
      target_lead = spec[1]
      ego = target_lead + 10.0
      rows = q._run_far_lead_case(target_lead)
      label = f"Far lead convergence: {ego} mph set / {target_lead} mph lead"
    values = (rows[:, 0], rows[:, 1], rows[:, 5], ego, None, rows[:, 2], rows[:, 3],
              rows[:, 4], rows[:, 6], rows[:, 7], rows[:, 8], rows[:, 9], rows[:, 10], rows[:, 11],
              rows[:, 12], rows[:, 13])
  else:
    raise ValueError(f"unknown trace kind: {kind}")

  time, ego, lead, target, grade, gap, time_gap, accel, planner_accel, gas, brake, brake_request, mode, transitions, predictive_brake, safety = values
  sample_dt = float(np.median(np.diff(np.asarray(time, dtype=float))))
  jerk = np.gradient(np.asarray(accel, dtype=float), sample_dt)
  planner_jerk = np.gradient(np.asarray(planner_accel, dtype=float), sample_dt)
  def clean(value):
    if value is None:
      return None
    return [None if isinstance(x, str) else float(x) for x in value]
  return {"label": label, "time": clean(time), "ego": clean(ego), "lead": clean(lead),
          "target": target, "grade": grade, "gap": clean(gap), "time_gap": clean(time_gap),
          "accel": clean(accel), "planner_accel": clean(planner_accel), "controller_accel": clean(rows[:, -1]), "gas": clean(gas),
          "jerk": clean(jerk), "planner_jerk": clean(planner_jerk),
          "brake": clean(brake), "brake_request": clean(brake_request),
          "predictive_brake": clean(predictive_brake), "safety": clean(safety), "mode": list(mode),
          "transitions": clean(transitions)}


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>CR-V longitudinal traces</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{font:14px system-ui,sans-serif;margin:24px;color:#222;background:#fafafa}
select{font:16px;padding:6px;min-width:420px}.plot{width:100%;max-width:1100px;height:1120px;background:white;border:1px solid #ccc;margin:12px 0 22px}
.meta{background:white;border:1px solid #ddd;padding:10px;max-width:1080px}.legend span{margin-right:18px}.swatch{display:inline-block;width:12px;height:12px;margin-right:4px}
</style></head><body>
<h1>CR-V longitudinal quality traces</h1>
<p>Select a representative case to inspect the actual closed-loop signals. Gas is normalized 0–1 from the Honda Bosch 0–1600 command; brake is normalized from the commanded negative acceleration.</p>
<p><a href="http://localhost:8050/">Open the live Dash dashboard</a>. The dashboard refreshes only when you press its refresh button.</p>
<select id="case"></select><div class="meta" id="meta"></div>
<div id="trace" class="plot"></div>
<script>
const traces=__TRACE_DATA__;
const colors={ego:'#1769aa',lead:'#d35400',target:'#777',planner:'#7b1fa2',actual:'#00897b',jerk:'#ad1457',gas:'#2e7d32',brake:'#c62828',gap:'#ef6c00',timegap:'#1565c0'};
const $=id=>document.getElementById(id);
traces.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=x.label;$('case').appendChild(o)});
function render(){
 const t=traces[$('case').value];
 $('meta').innerHTML=`<b>${t.label}</b> &nbsp; final mode: ${t.mode[t.mode.length-1]} &nbsp; mode transitions: ${t.transitions[t.transitions.length-1]}${t.grade===null?'':` &nbsp; grade: ${t.grade}%`}`;
 const data=[];
 const add=(name,values,color,xaxis,yaxis,dash='solid')=>{if(values!==null)data.push({x:t.time,y:values,name,type:'scatter',mode:'lines',connectgaps:false,line:{color,width:2,dash},xaxis,yaxis})};
 add('ego mph',t.ego,colors.ego,'x','y');
 add('lead mph',t.lead,colors.lead,'x','y');
 add('target mph',t.time.map(()=>t.target),colors.target,'x','y');
 add('planner m/s²',t.planner_accel,colors.planner,'x2','y2');
 add('LongControl m/s²',t.controller_accel,'#9467bd','x2','y2');
 add('physical m/s²',t.accel,colors.actual,'x2','y2');
 add('predictive brake m/s²',t.predictive_brake,colors.brake,'x2','y2','dot');
 add('safety override (1=engaged)',t.safety===null?null:t.safety.map(v=>v?1:0),'#000','x2','y3');
 add('physical jerk m/s³',t.jerk,colors.jerk,'x3','y4');
 add('planner jerk m/s³',t.planner_jerk,colors.planner,'x3','y4','dot');
 add('gas command 0–1',t.gas,colors.gas,'x4','y5');
 add('brake request',t.brake_request,colors.brake,'x4','y5');
 add('brake intensity 0–1',t.brake,'#8e24aa','x4','y5');
 add('mode (gas=1, coast=0, brake=-1)',t.mode.map(m=>m==='gas'?1:m==='brake'?-1:0),'#455a64','x4','y6');
 add('gap m',t.gap,colors.gap,'x5','y7');
 add('time gap s',t.time_gap,colors.timegap,'x5','y8');
 const grid={showgrid:true,gridcolor:'#e5e5e5'};
 const layout={height:1120,margin:{l:70,r:85,t:45,b:45},hovermode:'x unified',showlegend:true,paper_bgcolor:'white',plot_bgcolor:'white',title:{text:'CR-V longitudinal trace (shared time axis)',x:0.02,xanchor:'left'},
   xaxis:{domain:[0,1],anchor:'y',title:'time (s)',...grid},
   xaxis2:{domain:[0,1],anchor:'y2',matches:'x',showticklabels:false,...grid},
   xaxis3:{domain:[0,1],anchor:'y4',matches:'x',showticklabels:false,...grid},
   xaxis4:{domain:[0,1],anchor:'y5',matches:'x',showticklabels:false,...grid},
   xaxis5:{domain:[0,1],anchor:'y7',matches:'x',showticklabels:false,...grid},
   yaxis:{domain:[.83,1],title:'speed (mph)',...grid},
   yaxis2:{domain:[.64,.80],title:'acceleration (m/s²)',...grid},
   yaxis3:{overlaying:'y2',anchor:'x2',side:'right',title:'safety',range:[0,1],showgrid:false},
   yaxis4:{domain:[.45,.61],title:'jerk (m/s³)',...grid},
   yaxis5:{domain:[.26,.42],title:'actuator command',range:[-0.05,1.05],...grid},
   yaxis6:{overlaying:'y5',anchor:'x4',side:'right',title:'mode',range:[-1.2,1.2],tickvals:[-1,0,1],showgrid:false},
   yaxis7:{domain:[.04,.22],title:'gap (m)',...grid},
   yaxis8:{overlaying:'y7',anchor:'x5',side:'right',title:'time gap (s)',showgrid:false},
 };
 Plotly.react('trace',data,layout,{responsive:true,displaylogo:false});
}
$('case').onchange=render;render();
</script></body></html>"""


def main():
  with ProcessPoolExecutor(max_workers=12) as executor:
    traces = list(executor.map(_trace, CASES))
  OUT.write_text(HTML.replace("__TRACE_DATA__", json.dumps(traces, separators=(",", ":"))))
  OUT_JSON.write_text(json.dumps({"schema_version": 1,
                                  "generated_at": datetime.now(UTC).isoformat(),
                                  "plant_model": "production LongControl + Honda CarController + decoded CAN + delayed grey-box dynamics",
                                  "traces": traces}, separators=(",", ":")))
  print(f"wrote {OUT} ({len(traces)} traces)")


if __name__ == "__main__":
  main()
