# CR-V stop-controller experiment — 2026-10-04

This is local, undeployed controller work, not a release candidate. The installed
source baseline is `22515b2993116dbc6ffb17a1665addd40bc33f22`. Comparisons use the
same calibrated Honda plant and perception inputs; no pass thresholds were relaxed.

## Controller

An identified stationary lead commits to a distance-based stopping reference,
rather than repeatedly chasing a speed/headway equilibrium through standstill.
For remaining distance `d`, `v = k*d^(3/4)` gives `a = -3*v²/(4*d)`.
Accounting for unavoidable brake/road-load deceleration `b` gives
`v² = C*d^(3/2) + 2*b*d`, hence `a = -3*v²/(4*d) + b/2`, limited by the
physical deceleration floor. A local cubic speed reference stays nonnegative and
monotone; the existing acceleration/jerk state bounds normal maneuver onset.

At wheel-speed dropout, the reference retains unresolved motion instead of
declaring the car stopped. Only physical standstill enters holding. A committed
stop does not apply gas to correct the final distance. Its acceleration integral
is retained, with measurement confidence reduced where the wheel signal is
censored. Genuine lead movement releases the commitment.

The actuator correction also preserves emergency brake authority through
feedforward, integral feedback, and the stopping state: positive reference jerk
must not cancel a full emergency braking request. Safety is not disabled to
improve a comfort score.

## Frozen results

Runs are 195 seconds with a 15-second lead-in and final 60-second settling gate.
The report JSON records source hashes; the focused report also embeds candidate
controller sources and archives each run. The full comparisons completed with
`source_changed_during_run=false` and matching plant/input hashes.

| Comparison | Installed baseline | Candidate |
| --- | ---: | ---: |
| Eight stationary-lead stops, including grades and ego-model bias | 0/8 | 8/8 |
| Focused stops plus 1 and 3 mph rolling leads | 1/10 | 9/10 |
| Full longitudinal matrix | 156/199 | 155/199 |

Stationary-stop final clearances are 3.15–4.49 m, without gas reapplication or
nominal safety intervention. Final moving deceleration is approximately
0.33 m/s² on level/downhill cases and 0.485 m/s² uphill. These are simulated
results, not hardware validation. Controller/unit/actuation checks: 355 passed;
full-tree Ruff and whitespace checks passed.

Paired full-matrix changes:

- The 25 and 35 mph noisy multiphase-braking cases improve from fail to pass.
- The 35 and 50 mph rolling-lead-stop cases regress: maximum *planner* jerk
  rises to roughly 32 m/s³ when safety requests full braking near wheel dropout.
- The 25 mph / 20 mph closing / moving case regresses, with approximately
  2.53 mph velocity undershoot. Minimum time gap improves to 1.96 seconds, but
  its transient recovery/velocity-match behavior is unacceptable.
- The 37 cruise failures are unchanged; all 17 speed transitions still pass.
- Four noisy multiphase cases still fail, and the 1 mph crawl remains failing.

## Remaining work before deployment

The two rolling-stop regressions lack a committed stop reference when the wheel
signal disappears. Safety conservatively treats unresolved motion as moving and
cannot credit ordinary following with guaranteed retained brake authority.
Suppressing that safety bound would hide the problem. The next controller work
must coordinate late lead-stop intent, censored ego motion, and actual retained
actuation before optimizing comfort.

The test plant also switches between idle creep and a minimum deceleration at
the brake-request boundary. A smooth constant-pedal 1 mph crawl is not represented
faithfully by that binary model. This limitation does not explain away the
rolling-stop or moving-lead controller regressions above.

Reports (server root `/srv/storage/openpilot`):

- [Focused stop comparison](http://localhost:8000/crv-tuning-work/analysis-output/honda-low-speed-calibration/closed-loop-stops/report.html)
- [Full candidate matrix](http://localhost:8000/crv-tuning-work/analysis-output/crv-spatial-stop-controller/final/matrix.html)
- [Full installed-baseline matrix](http://localhost:8000/crv-tuning-work/analysis-output/crv-spatial-stop-controller/baseline/matrix.html)

The focused Plotly report has shared-X speed, acceleration, physical jerk,
gas/brake, and gap/stop-phase/safety subplots. It does not refresh automatically.
