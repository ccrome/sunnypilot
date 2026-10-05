# Honda low-speed plant identification

This changes the regression vehicle/observer harness, not the production
planner, controller, CAN mapping, or deployed device. Parameters are shared
across all speeds and scenarios; there are no per-case corrections.

## Missing behavior

- Individual recorded wheel channels have a minimum positive reading of
  2.02 km/h, then disappear separately. Zero wheel speed is not proof of zero
  physical speed. The harness converts this to control-speed units and retains
  the production speed estimator, sample-and-hold, quantization and seeded noise.
- Releasing the brake can expose idle/CVT creep even with only 13 CAN gas units.
  Active drive effort, creep, brake pressure demand and brake effort have
  independent causal states. None reset at the gas/brake crossover.
- Negative CAN acceleration is modeled as a net-deceleration request, rather
  than brake force with road resistance subtracted a second time. Required
  brake force accounts for road resistance and residual drive/creep effort.
  Low-speed minimum brake authority and additional pressure buildup are smooth
  grey-box approximations, not an emulator of Honda's receiving ECU.
- Encoded CAN standstill requests pass through the same transport queue and
  finite brake response; they do not teleport the vehicle to zero speed.

## Reproduction

From the repository root, with the repository, opendbc_repo and msgq_repo on
PYTHONPATH, use the analysis environment containing pandas, scipy, numba and
Plotly:

```bash
python -m openpilot.selfdrive.test.longitudinal_maneuvers.calibrate_honda_stops \
  --train 00000044--7400b9a4b2 00000041--9f46bca8e3 0000003a--5fa3fea7bc \
  --validate 00000047--dd29f6f277 0000003b--a28eefbb10 \
  --log-root /srv/storage/openpilot/route-data \
  --output /srv/storage/openpilot/crv-tuning-work/analysis-output/honda-low-speed-calibration
```

Missing caches are extracted in parallel; `--refresh` re-extracts them.
Incomplete compressed segments are excluded and recorded in inventory JSON,
without deleting the original logs. The latest route's incomplete segment 21
is excluded. Outputs include per-stop metrics and static, offline Plotly
charts with shared time axes, wheel-observer versus physical speed, observer
acceleration/jerk, recorded gas/brake and standstill commands.

Each stop is a 10-second CAN-input free rollout from its last measured speed
above 2.5 m/s. Only the initial state is supplied; the first second is excluded
from scoring. Equal route weighting prevents stock-drive duration from
dominating the fit. Thirteen stops train six low-speed dynamics parameters;
eight stops in disjoint whole routes evaluate transfer. Higher-speed windows
are also evaluated. The legacy fit remains in `honda_calibration.json`; applied
defaults and the new comparison are in `honda_low_speed_calibration.json`.

## What is and is not established

Held-out stop velocity RMSE falls from 0.242 to 0.201 m/s on latest custom ACC
and from 1.038 to 0.512 m/s on stock ACC. Higher-speed velocity and observer
acceleration errors improve on both held-out routes. However, latest-route
stop observer acceleration RMSE rises from 0.486 to 0.593 m/s²; individual
stops can worsen, as shown in the report. This is not a uniformly accurate
plant or an on-road validation of controller safety.

The latest route was excluded from parameter fitting but used to discover
missing phenomena, so this is not blind validation. Wheel, transmission and
model speed are proxies, not independent physical-speed ground truth. Pose
pitch includes body motion; final physical jerk cannot be inferred directly
from wheel-cutoff estimator spikes. Hold deceleration and channel spread are
assumptions, not independently identified parameters. The calibration replay
omits sensor noise and fine quantization; a runtime parity check gives physical
speed error below 1e-15 m/s, with small observer differences from quantization.

Nineteen harness tests cover production CAN, causal dynamics, crossover,
hold, cutoff/partial cutoff, creep, estimator parity and determinism. The
installed controller still fails nine of ten terminal-stop cases with this
plant; those failures are retained, not tuned away by changing test gates.
