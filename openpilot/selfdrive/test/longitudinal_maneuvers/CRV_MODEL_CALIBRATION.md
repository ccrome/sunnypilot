# CR-V vehicle model and controller experiment — October 3, 2026

This is an experimental source snapshot, **not a hardware-qualified release**.
Installed release `86ffa4f` was reported and observed to cycle gas/brake badly.
No new device update was performed during this investigation.

## Vehicle identification

The previous harness fed perfect physical acceleration into LongControl. It
also used one effort state, a 0.3 s transport delay, and slow throttle release.
Those assumptions concealed important real feedback/crossover behavior.

The replacement uses actual powertrain ACC_CONTROL CAN inputs, independent
drive/brake effort states, separate throttle-rise/release time constants,
50 Hz speed sampling, quantization, and the production Honda CarState speed
Kalman filter at 100 Hz. A deterministic 0.006 m/s measurement-disturbance
proxy exercises estimator uncertainty; its exact stochastic law is not known.

Training routes: `0000003d--430a4b475f` (custom, recorded build `86ffa4f`) and
`0000003a--5fa3fea7bc` (stock ACC). Entire held-out route:
`0000003b--a28eefbb10` (stock ACC). No segments are shared across splits.
See `honda_calibration.json` for parameters and per-route validation results.

Five-second free rollouts score their final four seconds after initializing
actuator state. Moving samples above 2 m/s with driver pedals released are
used. On 561 held-out windows, speed RMSE improved from 0.763 to 0.324 m/s,
and acceleration RMSE from 0.308 to 0.140 m/s². Feeding recorded raw speeds
through the production observer reproduces logged aEgo to numerical precision.
CAN gas-effort input predicts the held-out data better than treating the CAN
acceleration field alone as net vehicle acceleration.

Reproduce with the log-analysis environment (pandas/scipy/numba are offline
identification dependencies, not new production dependencies):

```bash
export PYTHONPATH=/srv/storage/openpilot/sunnypilot-full:/srv/storage/openpilot/sunnypilot-full/opendbc_repo:/srv/storage/openpilot/sunnypilot-full/msgq_repo
/srv/storage/openpilot/.venv-log-dashboard/bin/python \
  openpilot/selfdrive/test/longitudinal_maneuvers/calibrate_honda_vehicle.py \
  --train 0000003d--430a4b475f 0000003a--5fa3fea7bc \
  --validate 0000003b--a28eefbb10 \
  --output /srv/storage/openpilot/crv-tuning-work/analysis-output/honda-calibration-86ffa4f
```

## Controller correction

- The outer speed/gap reference uses its planned acceleration for damping,
  rather than closing another strong loop around delayed measured acceleration.
- CR-V inner-loop proportional acceleration feedback is removed. Continuous
  acceleration-error integration supplies velocity-error correction without
  amplifying individual wheel-speed/acceleration samples into pedal changes.
- Feedforward compensates identified rolling/drag load and maps net effort
  through the drive/brake gains. The 2.2/full-scale model versus 2.0/full-scale
  CR-V encoder conversion is explicit. Honda CAN encoding is unchanged.
- Safety reachability includes identified brake gain and first-order brake
  buildup. A tau-long hold bounds the first-order response conservatively;
  independent numerical tests exercise that bound. This does not increase a
  TTC threshold or add a filter to measured feedback.

## Verification and limitations

- Representative subset: 3 test groups / 12 cases pass in 28.47 s, retaining
  15 s lead-in, at least 180 s maneuver time, and final-60-second stability.
- Observer, controller, harness, and reachability checks: 313 passed in 1.03 s.
- Existing controls/CR-V regressions: 51 passed, 1 skipped in 199.34 s.
- Full nominal quality matrix: 185 pass, 8 fail out of 193 cases. All 99
  cruise/grade cases settle within 0.083 mph with zero resolved tail-command
  derivative. Speed/grade, rolling stop, far-lead, and lead-deviation gates pass.
- Failed nominal cases are moving leads: 25/15, 25/20, 45/10, 45/15, 45/30,
  45/40, 85/20, and 85/40 mph ego/closing. They fail pedal-release timing and/or
  the blanket no-safety/no-second-brake-phase checks, not the nominal one-second
  time-gap floor or final steady-state command gates. These failures remain
  reported; thresholds were not relaxed to make them pass.
- Some pedal-handoff assertions need physical review: gas can be necessary
  while net acceleration remains negative. Do not assume GAS_COMMAND>0 means
  the vehicle is accelerating.
- Weaker/stronger actuator authority and noisier measurements retain the
  substantial settled-command improvement. Extreme slower-response probes
  (0.8 s transport, 0.7 s brake response) still violate the one-second gap in
  severe approaches. Brake-buildup prediction removes the simulated collisions
  in the tested lead subset, but this is **not sufficient for deployment**.
- Low-speed stopping/contact mechanics, real ECU internal state, perception
  dropouts, grade/pitch calibration, and long-horizon open-loop fidelity are
  not fully identified. Point-mass stopping can exaggerate physical jerk peaks.
- Full-tree Ruff passes. Broader pytest stops during collection because
  `panda/board/jungle/scripts/echo_loopback_test.py` requires missing `termcolor`.

Test setups were corrected without weakening core limits: runs extend to
allow 60 seconds after actual target crossing, stopped-lead starting geometry
uses an actually stopped lead, and a planned stop need not invoke emergency
safety to count as successful. Clearances remain signed so collisions cannot
be hidden by clamping to zero.

## Reports

- `http://localhost:8000/crv-tuning-work/analysis-output/honda-calibration-86ffa4f/report.html`
- `http://localhost:8000/sunnypilot-full/openpilot/selfdrive/test/longitudinal_maneuvers/crv_longitudinal_trace_report.html`
- `http://localhost:8000/sunnypilot-full/openpilot/selfdrive/test/longitudinal_maneuvers/crv_longitudinal_quality_matrix.html`
- `http://localhost:8000/sunnypilot-full/openpilot/selfdrive/test/longitudinal_maneuvers/crv_controller_comparison.html`

Raw paired experiments and test logs are preserved under
`/srv/storage/openpilot/crv-tuning-work/analysis-output/controller-calibrated-*`.
Reports use Plotly and shared time axes. They do not automatically reload or
change the user's selected result. The dashboard adds an observed-acceleration
trace on its next normal launch; the running server was not restarted just to
change the user's current view.
