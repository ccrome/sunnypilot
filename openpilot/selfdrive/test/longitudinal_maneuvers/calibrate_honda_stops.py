"""Identify low-speed CAN dynamics, keeping whole routes out of parameter fitting."""
import argparse
from dataclasses import asdict, replace
from html import escape
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from openpilot.selfdrive.test.longitudinal_maneuvers.calibrate_honda_vehicle import DT, K, estimator_replay, extract_route, predict, windows
from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaDynamics


def stop_windows(data):
  scale = float(np.median((data.raw / data.wheel)[data.wheel.gt(6.) & data.raw.gt(6.)]))
  observed = data.wheel.to_numpy() * scale
  estimated = estimator_replay(observed, observed[0], 0., K)
  data = data.copy()
  data['raw'] = observed
  data['speed'], data['acceleration'] = estimated[:, 0], estimated[:, 1]
  zero = data.wheel.lt(.01)
  groups = (zero != zero.shift()).cumsum()
  result, centers = [], []
  for _, held in data[zero].groupby(groups[zero]):
    if held.t.iloc[-1] - held.t.iloc[0] < .5:
      continue
    center = float(held.t.iloc[0])
    history = data[data.t.between(center-20., center)]
    faster = history[history.wheel*scale > 2.5]
    if not len(faster):
      continue
    start = float(faster.t.iloc[-1])
    frame = data[data.t.between(start, start+10.)]
    if (len(frame) < 900 or frame.wheel.max() < 2. or frame.gas_pressed.any()
        or frame.brake_pressed.any() or frame.on.min() <= 0.
        or not frame[['raw', 'speed', 'acceleration', 'accel', 'gas', 'braking', 'pitch']].notna().all().all()):
      continue
    times = np.arange(1000)*DT + start
    indices = np.searchsorted(data.t.to_numpy(), times, side='right')-1
    values = data.iloc[indices][['raw', 'speed', 'acceleration', 'accel', 'gas', 'braking', 'pitch', 'hold']].to_numpy(float)
    result.append(np.column_stack((values, times-float(data.t.iloc[0]))))
    centers.append(center-float(data.t.iloc[0]))
  return np.asarray(result), centers, scale


@njit(cache=True)
def predict_stops(windows_array, low_parameters, cutoff, high, k):
  creep, scale, creep_release_tau, buildup_tau, brake_preload, brake_gain = low_parameters
  gas_gain, _, gas_tau, release_tau, brake_tau, delay, load, drag = high
  result = np.empty((len(windows_array), windows_array.shape[1], 3))
  n_delay = int(round(delay/DT))
  for window in range(len(windows_array)):
    x = windows_array[window]
    v, ev, ea = x[0, 0], x[0, 1], x[0, 2]
    force = ea + load + drag*v*v + 9.81*np.sin(x[0, 6])
    drive, brake = max(0., force), min(0., force)
    idle = 0.
    demand = brake
    raw = v
    for i in range(len(x)):
      c = x[max(0, i-n_delay)]
      low_weight = np.exp(-(v/scale)**2)
      drive_target = gas_gain*c[4]/1600.*2.2*(1.-c[5])
      tau = gas_tau if drive_target > drive else release_tau
      drive += (1.-np.exp(-DT/tau))*(drive_target-drive)
      idle_target = creep*low_weight*(1.-c[5])
      idle_tau = release_tau if idle_target > idle else creep_release_tau
      idle += (1.-np.exp(-DT/idle_tau))*(idle_target-idle)
      resistance = load+drag*v*v+9.81*np.sin(x[i, 6])
      desired_decel = min(brake_gain*min(c[3], 0.), -brake_preload*low_weight)
      hold = c[7] if x.shape[1] > 7 else 0.
      desired_decel = min(desired_decel, -.5*hold)
      brake_target = min(0., desired_decel+resistance-drive-idle)*c[5]
      demand += (1.-np.exp(-DT/max(buildup_tau*low_weight, 1e-6)))*(brake_target-demand)
      brake += (1.-np.exp(-DT/brake_tau))*(demand-brake)
      a = drive+idle+brake-load-drag*v*v-9.81*np.sin(x[i, 6])
      v = max(0., v+DT*a)
      if i % 2 == 0:
        # Four channels can disappear separately at the observed wheel cutoff.
        raw = 0.
        for offset in (-.018, -.006, .006, .018):
          wheel = max(0., v+offset)
          raw += wheel*.25*(wheel >= cutoff)
      error = raw-ev
      ev, ea = ev+DT*ea+k[0]*error, ea+k[1]*error
      result[window, i] = ev, ea, v
  return result


def score(prediction, w):
  error = prediction[:, 100:, :2]-w[:, 100:, 1:3]
  return {'stops': len(w), 'speed_rmse_mps': float(np.sqrt(np.mean(error[:, :, 0]**2))),
          'acceleration_rmse_mps2': float(np.sqrt(np.mean(error[:, :, 1]**2)))}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--train', nargs='+', required=True)
  parser.add_argument('--validate', nargs='+', required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--log-root', type=Path, default=Path('/srv/storage/openpilot/route-data'))
  parser.add_argument('--workers', type=int, default=16)
  parser.add_argument('--refresh', action='store_true', help='Re-extract CAN caches, preserving incomplete source segments')
  args = parser.parse_args()
  if set(args.train) & set(args.validate):
    raise ValueError('Training and validation routes must be disjoint')
  args.output.mkdir(parents=True, exist_ok=True)
  datasets = {r: extract_route(r, args.log_root, args.output, args.workers, args.refresh) for r in args.train+args.validate}
  ws, centers, scales = {}, {}, {}
  for route, data in datasets.items():
    ws[route], centers[route], scales[route] = stop_windows(data)
    print(route, len(ws[route]), 'ACC stops', centers[route], flush=True)
  cutoff = float(np.median([datasets[r][['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr']]
                            .where(lambda x: x > 0.).min().min()*scales[r] for r in args.train]))
  # Equal weight for custom and stock drives; no validation samples in fitting.
  count = max(len(ws[r]) for r in args.train)
  training = np.concatenate([ws[r][np.linspace(0, len(ws[r])-1, count).astype(int)] for r in args.train])
  legacy = json.loads(Path(__file__).with_name('honda_calibration.json').read_text())['simulation_dynamics']
  p = HondaDynamics(**legacy)
  high = np.array([p.gas_gain, p.brake_gain, p.gas_tau, p.gas_release_tau, p.brake_tau, p.delay, p.rolling, p.drag])
  best = None
  for initial in ([.4, 2.5, .5, .2, .4, 1.1], [.8, 3., .8, .35, .6, 1.2], [.6, 2., .3, .1, .3, 1.]):
    def residual(values):
      predicted = predict_stops(training, values, cutoff, high, K)
      return (predicted[:, 100::5, :2]-training[:, 100::5, 1:3]).ravel()
    fit = least_squares(residual, initial, bounds=([0., .5, .02, 0., 0., .5], [1.5, 4., 1.5, .8, 1.5, 2.]),
                        loss='soft_l1', f_scale=.15, max_nfev=160, diff_step=.005)
    print('Fit', fit.cost, fit.x, fit.success, flush=True)
    if best is None or fit.cost < best.cost:
      best = fit
  values = np.round(best.x, 4)
  cutoff = round(cutoff, 4)
  parameter_names = ('creep_accel', 'creep_speed', 'creep_release_tau', 'brake_buildup_tau', 'brake_preload', 'brake_gain')
  applied = replace(p, **dict(zip(parameter_names, values.tolist(), strict=True)), wheel_speed_cutoff=cutoff)
  report = {'train_routes': args.train, 'validation_routes': args.validate,
            'parameters': dict(zip(parameter_names, values.tolist(), strict=True)), 'wheel_speed_cutoff': cutoff,
            'simulation_dynamics': asdict(applied), 'fitted_parameters_unrounded': best.x.tolist(),
            'optimizer_success': bool(best.success), 'routes': {},
            'legacy_dynamics': legacy,
            'method': '100 Hz causal CAN replay; 10-second free rollout from one measured initial state. '
                      + 'First second excluded from scoring; equal route weighting during fitting. '
                      + 'Calibration observer omits stochastic noise and fine quantization; runtime plant retains both.',
            'limitations': ['Wheel/transmission/model speeds are proxies, not independent physical-speed ground truth.',
                           'Latest route excluded from fitting but used earlier to identify the missing phenomena.',
                           'Pose pitch includes body motion and is not independently measured road grade.',
                           'Per-wheel cutoff fitted from training CAN minima; creep/buildup are grey-box terms.',
                           'Hold deceleration and wheel-channel spread are assumed, not independently identified.',
                           'Latest-route stop acceleration error worsens despite improved velocity error.',
                           'Observer spikes at wheel cutoff are not independent measurements of physical jerk.']}
  predictions = {}
  for route, w in ws.items():
    old = predict(w, high, 0, K)
    sensor_only = predict_stops(w, [0., values[1], p.gas_release_tau, 0., 0., p.brake_gain], cutoff, high, K)
    new = predict_stops(w, values, cutoff, high, K)
    report['routes'][route] = {'stop_times_s': centers[route], 'old': score(old, w),
                              'net_brake_cutoff': score(sensor_only, w), 'new': score(new, w),
                              'control_speed_scale': scales[route],
                              'per_stop': [{'time_s': center, 'old': score(old[i:i+1], w[i:i+1]),
                                            'new': score(new[i:i+1], w[i:i+1])}
                                           for i, center in enumerate(centers[route])]}
    predictions[route] = old, sensor_only, new
    print(route, json.dumps(report['routes'][route]), flush=True)
    high_windows = windows(datasets[route])
    if len(high_windows):
      report['routes'][route]['higher_speed'] = {'old': score(predict(high_windows, high, 0, K), high_windows),
        'new': score(predict_stops(high_windows, values, cutoff, high, K), high_windows)}
  (args.output/'stop-calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False))
  import plotly.graph_objects as go
  from plotly.offline import get_plotlyjs
  from plotly.subplots import make_subplots
  sections = []
  for route, w in ws.items():
    for index, center in enumerate(centers[route]):
      fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          subplot_titles=['Speed (mph)', 'Observer acceleration (m/s²)',
                                          'Observer jerk (m/s³; not physical jerk ground truth)', 'Recorded CAN commands'])
      time = w[index, :, 8]
      for name, prediction in zip(('Old plant', 'Net brake + cutoff', 'Updated plant'), predictions[route], strict=True):
        fig.add_trace(go.Scatter(x=time, y=prediction[index, :, 0]*2.236936, name=name+' speed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=prediction[index, :, 1], name=name+' acceleration'), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=np.gradient(prediction[index, :, 1], DT), name=name+' observer jerk'), row=3, col=1)
      fig.add_trace(go.Scatter(x=time, y=predictions[route][2][index, :, 2]*2.236936,
                              name='Updated plant physical speed', line={'dash': 'dot'}), row=1, col=1)
      fig.add_trace(go.Scatter(x=time, y=w[index, :, 1]*2.236936, name='Recorded wheel-observer speed'), row=1, col=1)
      fig.add_trace(go.Scatter(x=time, y=w[index, :, 2], name='Recorded wheel-observer acceleration'), row=2, col=1)
      fig.add_trace(go.Scatter(x=time, y=np.gradient(w[index, :, 2], DT), name='Recorded wheel-observer jerk'), row=3, col=1)
      fig.add_trace(go.Scatter(x=time, y=w[index, :, 3], name='CAN acceleration request'), row=4, col=1)
      fig.add_trace(go.Scatter(x=time, y=w[index, :, 4]/1600., name='CAN gas /1600'), row=4, col=1)
      fig.add_trace(go.Scatter(x=time, y=w[index, :, 5], name='CAN brake request'), row=4, col=1)
      fig.add_trace(go.Scatter(x=time, y=w[index, :, 7], name='CAN standstill hold'), row=4, col=1)
      fig.update_layout(height=1050, hovermode='x unified', title=f'{route}, stop {center:.2f}s — '
                        + ('TRAINING' if route in args.train else 'HELD OUT FROM FITTING'))
      sections.append(fig.to_html(full_html=False, include_plotlyjs=False))
  summary = '<table border="1" cellpadding="6"><tr><th>Route / role</th><th>Stops</th><th>Speed RMSE old → new (m/s)</th>'
  summary += '<th>Observer acceleration RMSE old → new (m/s²)</th></tr>'
  for route, metrics in report['routes'].items():
    old, new = metrics['old'], metrics['new']
    summary += (f'<tr><td>{escape(route)} / {"training" if route in args.train else "held out"}</td><td>{new["stops"]}</td>'
                + f'<td>{old["speed_rmse_mps"]:.3f} → {new["speed_rmse_mps"]:.3f}</td>'
                + f'<td>{old["acceleration_rmse_mps2"]:.3f} → {new["acceleration_rmse_mps2"]:.3f}</td></tr>')
  summary += '</table>'
  (args.output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Low-speed plant validation</title>'
      + '<script>'+get_plotlyjs()+'</script><h1>Honda low-speed plant validation</h1>' + summary
      + '<p>Velocity fidelity improves overall, but not for every stop; latest-route acceleration error worsens. '
      + 'Wheel-cutoff observer spikes are not independent physical jerk measurements.</p>'
      + '<details><summary>Parameters, methodology and per-stop results</summary><pre>'+escape(json.dumps(report, indent=2))
      + '</pre></details><p>Actual CAN input, free rollouts with one initial state; no controller retuning or per-stop state resets. '
      + 'Static plots; no automatic refresh.</p>'+''.join(sections))


if __name__ == '__main__':
  main()
