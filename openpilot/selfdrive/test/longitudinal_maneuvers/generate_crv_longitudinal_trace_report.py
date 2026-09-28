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
              rows[:, 2], rows[:, 8], rows[:, 3], rows[:, 5], rows[:, 6], rows[:, 7])
  elif kind == "cruise":
    target, grade = spec[1:]
    rows = q._run_cruise(target, grade)
    label = f"Cruise: {target} mph at {grade:+d}%"
    values = (rows[:, 0], rows[:, 1], None, target, grade, None, None,
              rows[:, 8], rows[:, 9], rows[:, 3], rows[:, 4], rows[:, 6], rows[:, 7])
  else:
    ego, closing, stopped = spec[1:]
    rows = q._run_lead_case(ego, closing, stopped)
    label = f"Lead: {ego} mph, {closing} mph closing, {'stopped' if stopped else 'moving'}"
    values = (rows[:, 0], rows[:, 1], rows[:, 5], ego, None, rows[:, 2], rows[:, 3],
              rows[:, 4], rows[:, 6], rows[:, 7], rows[:, 8], rows[:, 10], rows[:, 11])

  time, ego, lead, target, grade, gap, time_gap, accel, planner_accel, gas, brake, mode, transitions = values
  def clean(value):
    if value is None:
      return None
    return [None if isinstance(x, str) else float(x) for x in value]
  return {"label": label, "time": clean(time), "ego": clean(ego), "lead": clean(lead),
          "target": target, "grade": grade, "gap": clean(gap), "time_gap": clean(time_gap),
          "accel": clean(accel), "planner_accel": clean(planner_accel), "gas": clean(gas),
          "brake": clean(brake), "mode": list(mode), "transitions": clean(transitions)}


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>CR-V longitudinal traces</title>
<style>
body{font:14px system-ui,sans-serif;margin:24px;color:#222;background:#fafafa}
select{font:16px;padding:6px;min-width:420px} canvas{display:block;width:100%;max-width:1100px;height:220px;background:white;border:1px solid #ccc;margin:12px 0 22px}
.meta{background:white;border:1px solid #ddd;padding:10px;max-width:1080px}.legend span{margin-right:18px}.swatch{display:inline-block;width:12px;height:12px;margin-right:4px}
</style></head><body>
<h1>CR-V longitudinal quality traces</h1>
<p>Select a representative case to inspect the actual closed-loop signals. Gas is normalized 0–1 from the Honda Bosch 0–1600 command; brake is normalized from the commanded negative acceleration.</p>
<select id="case"></select><div class="meta" id="meta"></div>
<canvas id="speed" width="1100" height="220"></canvas>
<canvas id="accel" width="1100" height="220"></canvas>
<canvas id="actuator" width="1100" height="220"></canvas>
<canvas id="gap" width="1100" height="220"></canvas>
<script>
const traces=__TRACE_DATA__;
const colors={ego:'#1769aa',lead:'#d35400',target:'#777',planner:'#7b1fa2',actual:'#00897b',gas:'#2e7d32',brake:'#c62828',gap:'#ef6c00',timegap:'#1565c0'};
const $=id=>document.getElementById(id);
traces.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=x.label;$('case').appendChild(o)});
function finite(v){return v!==null&&Number.isFinite(v)}
function draw(id, series, title, yLabel){
 const c=$(id),ctx=c.getContext('2d'),w=c.width,h=c.height,p={l:58,r:18,t:28,b:30};ctx.clearRect(0,0,w,h);
 const active=series.filter(s=>s.data!==null), t=traces[$('case').value], xs=t.time; let vals=[]; active.forEach(s=>s.data.forEach(v=>{if(finite(v))vals.push(v)}));
 if(!vals.length)return; let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.1;lo-=pad;hi+=pad;
 const x=v=>p.l+(v-xs[0])/(xs[xs.length-1]-xs[0])*(w-p.l-p.r), y=v=>h-p.b-(v-lo)/(hi-lo)*(h-p.t-p.b);
 ctx.strokeStyle='#ddd';ctx.fillStyle='#555';ctx.font='12px system-ui';ctx.beginPath();ctx.moveTo(p.l,p.t);ctx.lineTo(p.l,h-p.b);ctx.lineTo(w-p.r,h-p.b);ctx.stroke();
 ctx.fillText(title,p.l,17);ctx.fillText(yLabel,4,p.t+10);ctx.fillText(lo.toFixed(2),4,h-p.b);ctx.fillText(hi.toFixed(2),4,p.t+4);ctx.fillText('time (s)',w-70,h-8);
 active.forEach(s=>{ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();let on=false;s.data.forEach((v,i)=>{if(!finite(v)){on=false;return}const X=x(xs[i]),Y=y(v);if(!on)ctx.moveTo(X,Y);else ctx.lineTo(X,Y);on=true});ctx.stroke()});
 const lx=p.l+8;active.forEach((s,i)=>{ctx.fillStyle=s.color;ctx.fillRect(lx+i*145,h-18,10,10);ctx.fillStyle='#333';ctx.fillText(s.name,lx+14+i*145,h-8)});
}
function render(){const t=traces[$('case').value];$('meta').innerHTML=`<b>${t.label}</b> &nbsp; final mode: ${t.mode[t.mode.length-1]} &nbsp; mode transitions: ${t.transitions[t.transitions.length-1]}${t.grade===null?'':` &nbsp; grade: ${t.grade}%`}`;
 draw('speed',[{name:'ego mph',data:t.ego,color:colors.ego},{name:'lead mph',data:t.lead,color:colors.lead},{name:'target mph',data:t.time.map(()=>t.target),color:colors.target}],'Speed','mph');
 draw('accel',[{name:'planner m/s²',data:t.planner_accel,color:colors.planner},{name:'physical m/s²',data:t.accel,color:colors.actual}],'Acceleration','m/s²');
 draw('actuator',[{name:'gas 0–1',data:t.gas,color:colors.gas},{name:'brake 0–1',data:t.brake,color:colors.brake}],'Honda actuator requests','normalized');
 draw('gap',[{name:'gap m',data:t.gap,color:colors.gap},{name:'time gap s',data:t.time_gap,color:colors.timegap}],'Lead gap','mixed units');}
$('case').onchange=render;render();
</script></body></html>"""


def main():
  with ProcessPoolExecutor(max_workers=12) as executor:
    traces = list(executor.map(_trace, CASES))
  OUT.write_text(HTML.replace("__TRACE_DATA__", json.dumps(traces, separators=(",", ":"))))
  print(f"wrote {OUT} ({len(traces)} traces)")


if __name__ == "__main__":
  main()
