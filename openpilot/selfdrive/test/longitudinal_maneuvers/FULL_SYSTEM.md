# CR-V regression vehicle loop

All Plant instances selecting HONDA_CRV_5G now default to full_system=True.
Other vehicle fingerprints retain their legacy behavior. An explicit
full_system=False is available for planner-only diagnostics.

The loop is planner (20 Hz), production LongControl (100 Hz), production Honda
CarController (100 Hz, ACC_CONTROL transmission at 50 Hz), encoded/decoded
ACC_CONTROL, a transport queue, and first-order physical effort response.
Measured speed and net acceleration return to the planner and LongControl.
Gas response uses GAS_COMMAND; brake response uses BRAKE_REQUEST and
ACCEL_COMMAND. Reported gas/brake traces reflect decoded CAN commands.
Planner acceleration never directly advances the vehicle in this mode.

HondaDynamics holds globally applicable gas/brake gains and time constants,
transport delay, rolling resistance and quadratic drag. Response values use
the existing archived grey-box fit as a starting point; conversion from gas
units to wheel effort and the assumed 0.3 s delay still require independent
validation against drives. Supply actuator_parameters to Plant to test model
uncertainty; do not tune these values separately for regression speeds.

The planner/controller code and CAN serialization are production code. This
harness does not execute perception, radar association, panda safety firmware,
or Honda ECU software. Synthetic radar/model inputs and the physical-response
model remain simulation approximations.

Run/settling durations and physical safety limits are retained. A failure needs
inspection of both the controller and the test assumptions, rather than tuning
the controller merely to satisfy an assertion. The added
test_honda_full_system.py checks transport causality, physical brake response,
response-parameter sensitivity and disabled command inhibition.

## Interpreting results

- The lead feasibility classifier uses ideal jerk-limited braking. It does not
  certify reachability with transport delay, actuator lag, drag, or grade. Keep
  safety intervention visible, but do not equate its presence with a defect
  until the initial condition's actual reachability has been checked.
- A stopped-lead case requires a collision-free, stable stop, not activation of
  emergency safety. Successfully avoiding emergency intervention is valid.
- Raw gas/coast/brake phase counts can include quantized zero-effort requests.
  Inspect command magnitude and duration before interpreting each transition
  as occupant-perceptible hunting. The raw traces remain available.
- Settled command derivatives account for one CAN count of measurement
  resolution, while also checking one-second changes to detect accumulated
  drift. This does not filter feedback or excuse larger oscillations.
- Cruise feasibility currently uses planner acceleration limits, although
  physical grade-holding authority belongs to the actuator/vehicle model.
  Treat that classification as provisional, not as grounds for accepting a
  real speed-control failure.

Compare improvements with independent logged drives and variations of the
physical-response parameters before drawing hardware-readiness conclusions.
Passing the nominal model alone is insufficient.
