"""Stationary vision lead: finish braking rather than close the last metre.

Oct. 4 route 44, bookmark at 219.666 s: vision ego speed underestimated
wheel speed by about 0.45 m/s near 3 mph, although model lead speed remained
near zero. Pose pitch was about -0.02 rad. These are perception disturbances,
not forced ego motion or a retuned vehicle. Exercise production vision fusion.
"""
import math

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant
from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import MPH


CASES = [(speed, grade, bias, rolling) for speed in (25, 35, 50)
         for grade, bias, rolling in ((0, 0.0, 0.0), (-2, 0.45, 0.0))]
CASES += [(35, grade, 0.45, 0.0) for grade in (-3, 3)]
# A sustained 1 mph lead is below this CR-V's measured 1.3 mph wheel-reporting
# cutoff and is not an observable constant-speed tracking target. The relevant
# low-speed case is covered by the rolling-crawl-then-stop regression, where
# radar-relative motion and the lead's stop forecast remain observable.
CASES += [(35, 0, 0.0, 3)]


def run_terminal_stop(speed_mph, grade, model_bias, rolling_mph):
  speed = speed_mph * MPH
  plant = Plant(lead_relevancy=True, speed=speed, distance_lead=2.05 * speed + 3.,
                physics=True, realtime=False, car_fingerprint=CAR.HONDA_CRV_5G,
                personality=log.LongitudinalPersonality.relaxed)
  rows = []
  while plant.current_time < 195.:
    elapsed = max(0., plant.current_time - 15.)
    lead_speed = max(rolling_mph * MPH, speed - 1.5 * elapsed)
    gap = plant.distance_lead - plant.distance
    near_stop = math.exp(-(plant.speed / 1.8) ** 2) * (elapsed > 0.)
    observation = {'d_rel': gap, 'model_lead_speed': lead_speed + .06 * near_stop,
                   'model_speed_std': .07 * near_stop,
                   'model_ego_speed': plant.vehicle.measured_speed - model_bias * near_stop,
                   'a_lead': -1.5 if lead_speed > rolling_mph * MPH else 0.}
    observation['model_lead_future'] = [max(rolling_mph * MPH, speed - 1.5 * (elapsed + future)) + .06 * near_stop
                                        if plant.current_time >= 15. else speed for future in ModelConstants.LEAD_T_IDXS]
    braking_remaining = max(0., lead_speed - rolling_mph * MPH) / 1.5
    observation['model_lead_future_x'] = [gap + RADAR_TO_CAMERA + lead_speed * min(future, braking_remaining)
      - .75 * min(future, braking_remaining) ** 2 + rolling_mph * MPH * max(0., future - braking_remaining)
      if plant.current_time >= 15. else gap + RADAR_TO_CAMERA + speed * future for future in ModelConstants.LEAD_T_IDXS]
    plant.step(v_lead=lead_speed, v_cruise=speed, pitch=math.atan(grade / 100.),
               lead_one=observation, lead_two=observation)
    debug = plant.planner.crv_debug
    rows.append((plant.current_time, plant.speed / MPH, plant.distance_lead - plant.distance,
                 gap / max(plant.speed, .1), plant.acceleration, lead_speed / MPH,
                 plant.planner_acceleration, plant.gas_command, plant.brake_intensity,
                 plant.brake_request, plant.actuator_mode, plant.mode_transitions,
                 plant.predictive_brake, plant.safety_override,
                 int(getattr(plant.planner, 'crv_stop_phase', 0)),
                 *getattr(plant.planner, 'crv_stop_reference', (0., 0., 0.)),
                 float(getattr(plant.planner, 'crv_required_clearance', 0.)),
                 float(getattr(plant.planner, 'crv_lead_follow_urgency', 0.)),
                 plant.vehicle.measured_speed / MPH, plant.vehicle.measured_acceleration, plant.vehicle.output,
                 plant.planner.output_should_stop, debug.get('stationaryLead', False),
                 debug.get('movingLead', False), debug.get('stoppingForecast', False),
                 debug.get('leadStop', False), debug.get('safetyClosingSpeed', 0.),
                 debug.get('speedStd', 0.), debug.get('wheelChannelCount', 4),
                 debug.get('closeClosingStop', False), debug.get('stoppedCloseLead', False),
                 debug.get('stoppedLeadLossHold', False), debug.get('candidateStop', False)))
  return np.asarray(rows, dtype=object)


def terminal_metrics(rows, rolling_mph=0.):
  t, speed, gap, acceleration, command = (rows[:, i].astype(float) for i in (0, 1, 2, 4, -1))
  tail = t >= t[-1] - 60.
  approach = (t > 15.) & (speed > .1) & (speed < 4.) & (rows[:, 5].astype(float) < .1)
  indices = np.flatnonzero(approach)
  first_rest = np.flatnonzero((t > 15.) & (speed < .02))
  moving = (t > 15.) & (speed > .1)
  renewed = float(np.max(np.maximum.accumulate(command[indices]) - command[indices])) if len(indices) else 0.
  final_moving = (t > 15.) & (speed > .02) & (speed * MPH < .15)
  committed = rows[:, 14].astype(int) > 0
  safety_pressure = rows[:, 19].astype(float)
  return {'final_clearance_m': float(gap[-1]), 'minimum_clearance_m': float(np.min(gap)),
          'settled_speed_error_mph': float(np.max(np.abs(speed[tail] - rolling_mph))),
          'low_speed_brake_reapplication_mps2': renewed,
          'low_speed_gas_peak': float(np.max(rows[approach, 7].astype(float))) if len(indices) else 0.,
          'final_moving_deceleration_mps2': float(-np.min(acceleration[final_moving])) if np.any(final_moving) else 0.,
          'safety_pressure_peak': float(np.max(safety_pressure[moving])) if np.any(moving) else 0.,
          'safety_engaged': bool(np.any(safety_pressure[moving] >= 1.0)),
          'committed_gas_peak': float(np.max(rows[committed, 7].astype(float))) if np.any(committed) else 0.,
          'minimum_moving_speed_mph': float(np.min(speed[t > 20.])),
          'stop_time_s': float(t[first_rest[0]]) if len(first_rest) else None,
          'settled_gas_span': float(np.ptp(rows[tail, 7].astype(float))),
          'settled_brake_span': float(np.ptp(rows[tail, 8].astype(float)))}


def terminal_failures(metrics, rolling_mph=0.):
  failures = []
  bounds = {'minimum_clearance_m': (2., math.inf), 'settled_speed_error_mph': (0., .1),
            'settled_gas_span': (0., .05), 'settled_brake_span': (0., .05)}
  if not rolling_mph:
    # A stopped car a little farther from the target is preferable to a
    # brake release and corrective creep; only the minimum safe clearance is
    # a hard stop-distance bound here.
    bounds.update(final_clearance_m=(2., math.inf), low_speed_brake_reapplication_mps2=(0., .8),
                  final_moving_deceleration_mps2=(0., .5), low_speed_gas_peak=(0., .0001),
                  committed_gas_peak=(0., .0001))
    if metrics['safety_engaged']:
      failures.append('nominal_stop_safety')
  elif metrics['minimum_moving_speed_mph'] < rolling_mph - .5:
    failures.append('unnecessary_stop_for_rolling_lead')
  for key, (lower, upper) in bounds.items():
    if not lower <= metrics[key] <= upper:
      failures.append(key)
  return failures


@pytest.mark.parametrize('case', CASES)
def test_stationary_lead_finishes_stop_without_creeping(case):
  rows = run_terminal_stop(*case)
  metrics = terminal_metrics(rows, case[-1])
  assert not terminal_failures(metrics, case[-1]), metrics
