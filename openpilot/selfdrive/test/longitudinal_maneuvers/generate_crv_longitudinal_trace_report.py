#!/usr/bin/env python3
"""Generate an interactive HTML trace viewer for CR-V longitudinal cases."""
# ruff: noqa: E501

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from openpilot.selfdrive.test.longitudinal_maneuvers import test_crv_longitudinal_quality_regression as q


OUT = Path(__file__).with_name("crv_longitudinal_trace_report.html")
CASES = (
  ("speed", 0, 15), ("speed", 0, 45), ("speed", 0, 75),
  ("speed", 90, 45), ("speed", 90, 15),
  ("cruise", 30, 0), ("cruise", 30, -3), ("cruise", 30, 3),
  ("lead", 25, 10, False), ("lead", 45, 20, False),
  ("lead", 65, 10, False), ("lead", 25, 15, True),
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
  else:
    ego, closing, stopped = spec[1:]
    rows = q._run_lead_case(ego, closing, stopped)
    label = f"Lead: {ego} mph, {closing} mph closing, {'stopped' if stopped else 'moving'}"
    values = (rows[:, 0], rows[:, 1], rows[:, 5], ego, None, rows[:, 2], rows[:, 3],
              rows[:, 4], rows[:, 6], rows[:, 7], rows[:, 8], rows[:, 9], rows[:, 10], rows[:, 11],
              rows[:, 12], rows[:, 13])

  time, ego, lead, target, grade, gap, time_gap, accel, planner_accel, gas, brake, brake_request, mode, transitions, predictive_brake, safety = values
  def clean(value):
    if value is None:
      return None
    return [None if isinstance(x, str) else float(x) for x in value]
  return {"label": label, "time": clean(time), "ego": clean(ego), "lead": clean(lead),
          "target": target, "grade": grade, "gap": clean(gap), "time_gap": clean(time_gap),
          "accel": clean(accel), "planner_accel": clean(planner_accel), "gas": clean(gas),
          "brake": clean(brake), "brake_request": clean(brake_request),
          "predictive_brake": clean(predictive_brake), "safety": clean(safety), "mode": list(mode),
          "transitions": clean(transitions)}


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>CR-V longitudinal traces</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{font:14px system-ui,sans-serif;margin:24px;color:#222;background:#fafafa}
select{font:16px;padding:6px;min-width:420px}.plot{width:100%;max-width:1100px;height:320px;background:white;border:1px solid #ccc;margin:12px 0 22px}
.meta{background:white;border:1px solid #ddd;padding:10px;max-width:1080px}.legend span{margin-right:18px}.swatch{display:inline-block;width:12px;height:12px;margin-right:4px}
</style></head><body>
<h1>CR-V longitudinal quality traces</h1>
<p>Select a representative case to inspect the actual closed-loop signals. Gas is normalized 0–1 from the Honda Bosch 0–1600 command; brake is normalized from the commanded negative acceleration.</p>
<select id="case"></select><div class="meta" id="meta"></div>
<div id="speed" class="plot"></div>
<div id="accel" class="plot"></div>
<div id="actuator" class="plot"></div>
<div id="gap" class="plot"></div>
<script>
const traces=__TRACE_DATA__;
const colors={ego:'#1769aa',lead:'#d35400',target:'#777',planner:'#7b1fa2',actual:'#00897b',gas:'#2e7d32',brake:'#c62828',gap:'#ef6c00',timegap:'#1565c0'};
const $=id=>document.getElementById(id);
traces.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=x.label;$('case').appendChild(o)});
function draw(id, series, title, yLabel, dualAxis=false, secondaryTitle=''){
 const t=traces[$('case').value], active=series.filter(s=>s.data!==null);
 const layout={title:{text:title,x:0.02,xanchor:'left'},height:320,margin:{l:65,r:dualAxis?70:25,t:45,b:55},hovermode:'x unified',showlegend:true,paper_bgcolor:'white',plot_bgcolor:'white',xaxis:{title:'time (s)',showgrid:true,gridcolor:'#e5e5e5'},yaxis:{title:yLabel,showgrid:true,gridcolor:'#e5e5e5'}};
 if(dualAxis) layout.yaxis2={title:secondaryTitle,overlaying:'y',side:'right',showgrid:false};
 if(!active.length) layout.annotations=[{text:'No lead trace for this case',showarrow:false,font:{size:16,color:'#666'},xref:'paper',yref:'paper',x:0.5,y:0.5}];
 const data=active.map(s=>({x:t.time,y:s.data,name:s.name,type:'scatter',mode:'lines',connectgaps:false,line:{color:s.color,width:2},...(s.axis?{yaxis:s.axis}:{})}));
 Plotly.react(id,data,layout,{responsive:true,displaylogo:false});
}
function render(){const t=traces[$('case').value];$('meta').innerHTML=`<b>${t.label}</b> &nbsp; final mode: ${t.mode[t.mode.length-1]} &nbsp; mode transitions: ${t.transitions[t.transitions.length-1]}${t.grade===null?'':` &nbsp; grade: ${t.grade}%`}`;
 draw('speed',[{name:'ego mph',data:t.ego,color:colors.ego},{name:'lead mph',data:t.lead,color:colors.lead},{name:'target mph',data:t.time.map(()=>t.target),color:colors.target}],'Speed','mph');
 draw('accel',[{name:'planner m/s²',data:t.planner_accel,color:colors.planner},{name:'physical m/s²',data:t.accel,color:colors.actual},{name:'predictive brake m/s²',data:t.predictive_brake,color:colors.brake},{name:'safety override (1=engaged)',data:t.safety===null?null:t.safety.map(v=>v?1:0),color:'#000',axis:'y2'}],'Acceleration and lead braking','m/s²',true,'safety override');
 draw('actuator',[{name:'gas command 0–1',data:t.gas,color:colors.gas},{name:'brake request',data:t.brake_request,color:colors.brake},{name:'brake intensity 0–1',data:t.brake,color:'#8e24aa'},{name:'mode (gas=1, coast=0, brake=-1)',data:t.mode.map(m=>m==='gas'?1:m==='brake'?-1:0),color:'#455a64',axis:'y2'}],'Honda gas / coast / brake commands','command (normalized)',true,'mode');
 draw('gap',[{name:'gap m',data:t.gap,color:colors.gap},{name:'time gap s',data:t.time_gap,color:colors.timegap,axis:'y2'}],'Lead gap','gap (m)',true,'time gap (s)');}
$('case').onchange=render;render();
</script></body></html>"""


def main():
  with ProcessPoolExecutor(max_workers=12) as executor:
    traces = list(executor.map(_trace, CASES))
  OUT.write_text(HTML.replace("__TRACE_DATA__", json.dumps(traces, separators=(",", ":"))))
  print(f"wrote {OUT} ({len(traces)} traces)")


if __name__ == "__main__":
  main()
