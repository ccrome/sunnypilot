"""Immutable stop comparison and dashboard-compatible results; no auto refresh."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
import time

from openpilot.selfdrive.test.longitudinal_maneuvers.generate_crv_longitudinal_quality_matrix import row
from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics
from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_terminal_stop import (
  CASES, run_terminal_stop, terminal_failures, terminal_metrics,
)

HERE = Path(__file__).resolve().parent
BASELINE_COMMIT = '22515b2993116dbc6ffb17a1665addd40bc33f22'


def installed_baseline():
  from openpilot.selfdrive.test.longitudinal_maneuvers import plant, honda_vehicle
  namespaces = []
  for path in ('openpilot/selfdrive/controls/lib/longitudinal_planner.py', 'openpilot/selfdrive/controls/lib/longcontrol.py'):
    namespace = {'__name__': 'immutable_' + Path(path).stem}
    exec(subprocess.check_output(['git', 'show', BASELINE_COMMIT + ':' + path], text=True), namespace)
    namespaces.append(namespace)
  plant.LongitudinalPlanner = namespaces[0]['LongitudinalPlanner']
  baseline_control = namespaces[1]['LongControl']

  class BaselineControl(baseline_control):
    def update(self, active, CS, a_target, should_stop, accel_limits, *unused):
      return super().update(active, CS, a_target, should_stop, accel_limits)

  honda_vehicle.LongControl = BaselineControl


def run_case(case):
  data = run_terminal_stop(*case)
  metrics = terminal_metrics(data, case[-1])
  failures = terminal_failures(metrics, case[-1])
  result = row('Terminal stop', str(case), not failures, {**metrics, 'failures': failures}, data, case[0])
  result['signals']['stop_phase'] = data[::10, 14].astype(int).tolist()
  result['signals']['stop_reference_speed_mph'] = (data[::10, 15].astype(float) * 2.236936).tolist()
  for name, index in (('stop_reference_acceleration_mps2', 16), ('stop_reference_jerk_mps3', 17),
                      ('required_clearance_m', 18), ('safety_pressure', 19)):
    result['signals'][name] = data[::10, index].astype(float).tolist()
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--label', choices=('baseline', 'candidate'), required=True)
  parser.add_argument('--workers', type=int, default=10)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  if args.label == 'baseline':
    installed_baseline()
  start = time.monotonic()
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    results = list(pool.map(run_case, CASES))
  # Explicit provenance is more useful than HEAD for an uncommitted iteration.
  source = HERE.parents[1] / 'controls/lib/longitudinal_planner.py'
  document = {'generated_at': datetime.now(UTC).isoformat(), 'label': args.label,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'elapsed_s': time.monotonic() - start,
              'planner_sha256': hashlib.sha256(subprocess.check_output(['git', 'show', BASELINE_COMMIT +
                ':openpilot/selfdrive/controls/lib/longitudinal_planner.py']) if args.label == 'baseline' else source.read_bytes()).hexdigest(),
              'results': results}
  for name, path in (('controller', 'openpilot/selfdrive/controls/lib/longcontrol.py'),
                     ('terminal_stop', 'openpilot/selfdrive/controls/lib/terminal_stop.py'),
                     ('plant', 'openpilot/selfdrive/test/longitudinal_maneuvers/honda_vehicle.py'),
                     ('stop_cases', 'openpilot/selfdrive/test/longitudinal_maneuvers/test_crv_terminal_stop.py')):
    content = (subprocess.check_output(['git', 'show', BASELINE_COMMIT + ':' + path])
               if args.label == 'baseline' and name == 'controller' else Path(path).read_bytes())
    document[name + '_sha256'] = None if args.label == 'baseline' and name == 'terminal_stop' else hashlib.sha256(content).hexdigest()
    if name == 'controller':
      document['controller_source'] = content.decode()
  document['planner_source'] = (subprocess.check_output(['git', 'show', BASELINE_COMMIT +
    ':openpilot/selfdrive/controls/lib/longitudinal_planner.py'], text=True) if args.label == 'baseline' else source.read_text())
  document['dynamics'] = asdict(HondaDynamics())
  document['terminal_stop_source'] = (HERE.parents[1] / 'controls/lib/terminal_stop.py').read_text() if args.label == 'candidate' else None
  document['passed'] = sum(r['passed'] for r in results)
  document['failed'] = len(results) - document['passed']
  (args.output / (args.label + '.json')).write_text(json.dumps(document, separators=(',', ':')))
  (args.output / (args.label + '_results.json')).write_text(json.dumps(document, separators=(',', ':')))
  archive = args.output / 'runs'
  archive.mkdir(exist_ok=True)
  (archive / (datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f') + '-' + args.label + '.json')).write_text(
    json.dumps(document, separators=(',', ':')))
  combined = [json.loads(p.read_text()) for p in (args.output / 'baseline.json', args.output / 'candidate.json') if p.exists()]
  summary = '<table border="1" cellpadding="5"><tr><th>Case (mph, grade %, model bias, rolling mph)</th>'
  summary += ''.join('<th>' + escape(run['label']) + '</th>' for run in combined) + '</tr>'
  cases = list(dict.fromkeys(result['case'] for run in combined for result in run['results']))
  for case in cases:
    summary += '<tr><td>' + escape(case) + '</td>'
    summary += ''.join('<td>' + ('PASS' if next((result['passed'] for result in run['results']
      if result['case'] == case), None) is True else 'FAIL' if any(result['case'] == case
      for result in run['results']) else '—') + '</td>' for run in combined) + '</tr>'
  summary += '</table>'
  script = r'''
const runs=DATA, select=document.getElementById('case');
const cases=[...new Set(runs.flatMap(run=>run.results.map(result=>result.case)))];
for(const name of cases){let o=document.createElement('option');o.value=name;o.textContent=name;select.appendChild(o)}
function render(){let name=select.value,traces=[],table=document.getElementById('metrics');table.replaceChildren();
for(const run of runs){let r=run.results.find(result=>result.case===name),dash=run.label==='baseline'?'dash':'solid';
if(!r){let h=document.createElement('h3');h.textContent=run.label+': not run';table.appendChild(h);continue}
let s=r.signals;
let h=document.createElement('h3');h.textContent=run.label+': '+(r.passed?'PASS':'FAIL');table.appendChild(h);
let pre=document.createElement('pre');pre.textContent=JSON.stringify(r.metrics,null,2);table.appendChild(pre);
const add=(key,name,n)=>{if(s[key])traces.push({x:s.time_s,y:s[key],name:run.label+' '+name,
type:'scatter',mode:'lines',line:{dash},xaxis:n===1?'x':'x'+n,yaxis:n===1?'y':'y'+n})};
add('ego_speed_mph','ego mph',1);add('lead_speed_mph','lead mph',1);
add('observed_speed_mph','observed mph',1);add('stop_reference_speed_mph','stop reference mph',1);
add('acceleration_mps2','physical acceleration',2);add('planner_acceleration_mps2','planner',2);add('controller_acceleration_mps2','Honda command',2);
add('stop_reference_acceleration_mps2','stop reference acceleration',2);
add('jerk_mps3','physical jerk',3);add('gas_command','gas',4);add('brake_intensity','brake intensity',4);add('brake_request','brake request',4);
add('gap_m','gap m',5);add('required_clearance_m','required clearance',5);add('safety_pressure','safety pressure',5);
add('safety_override','safety',5);add('stop_phase','stop phase',5);}
let layout={height:1200,hovermode:'x unified',legend:{orientation:'h'},margin:{l:80,r:30,t:80,b:45}};
let labels=['speed (mph)','acceleration (m/s²)','jerk (m/s³)','gas / brake','gap / phase / safety'];
for(let n=1;n<=5;n++){let suffix=n===1?'':n;layout['xaxis'+suffix]={anchor:'y'+suffix,
matches:n===1?undefined:'x',title:n===5?'time (s)':undefined};
layout['yaxis'+suffix]={domain:[1-n*.2+.02,1-(n-1)*.2-.02],title:labels[n-1]}}
Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false});}
select.onchange=render;render();
'''
  plotly = Path('/srv/storage/openpilot/crv-tuning-work/analysis-output/plotly.min.js').read_text()
  html = ('<!doctype html><meta charset="utf-8"><title>CR-V terminal stop comparison</title>'
          + '<h1>CR-V terminal stop: installed baseline versus candidate</h1>'
          + '<p>Same causal physics and observations. 195-second runs; final 60 seconds assessed. '
          + 'No automatic refresh. Recorded drive charts: '
          + '<a href="/crv-tuning-work/analysis-output/00000047--dd29f6f277-stops/report.html">drive report</a>.</p>'
          + summary
          + '<select id="case"></select><div id="plot"></div><div id="metrics"></div><script>'
          + plotly + '</script><script>' + script.replace('DATA', json.dumps(combined)) + '</script>')
  (args.output / 'report.html').write_text(html)
  print(json.dumps({k: v for k, v in document.items() if k not in
    ('results', 'terminal_stop_source', 'controller_source', 'planner_source')}), flush=True)
  for result in results:
    print(('PASS ' if result['passed'] else 'FAIL ') + result['case'] + ' ' + str(result['metrics']), flush=True)


if __name__ == '__main__':
  main()
