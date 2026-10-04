"""Oct. 3 drive regression: a smooth lead stop must not become brake hunting.

The recorded low-speed raw-speed excursion was about 0.2 m/s over 0.1 s.
Inject a same-sized one-wheel CAN innovation before the production observer;
do not alter physical acceleration or force the ego trajectory. Distance is
causal.
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


LEAD_NOISE_STD = .12 / math.sqrt(2.)


def run_multiphase_stop(ego_mph, actuator_parameters=None, pulse_amplitude=0.22, model_speed_std=LEAD_NOISE_STD):
  speed = ego_mph * MPH
  plant = Plant(lead_relevancy=True, speed=speed, distance_lead=2.05 * speed,
                physics=True, realtime=False, car_fingerprint=CAR.HONDA_CRV_5G,
                personality=log.LongitudinalPersonality.relaxed, actuator_parameters=actuator_parameters)
  pulse_start = None
  crawl = 1.5 * MPH
  braking_time = (speed - crawl) / 1.5

  def lead_state(elapsed):
    if elapsed < braking_time:
      return max(crawl, speed - 1.5 * elapsed), speed * elapsed - 0.75 * elapsed ** 2
    braking_distance = 0.5 * (speed + crawl) * braking_time
    crawl_elapsed = elapsed - braking_time
    if crawl_elapsed < 6.0:
      return crawl, braking_distance + crawl * crawl_elapsed
    stop_elapsed = crawl_elapsed - 6.0
    stop_time = crawl / 0.3
    stop_distance = crawl * 6.0 + crawl * min(stop_elapsed, stop_time) - 0.15 * min(stop_elapsed, stop_time) ** 2
    return max(0.0, crawl - 0.3 * stop_elapsed), braking_distance + stop_distance

  def wheel_sensor_excursion():
    nonlocal pulse_start
    if pulse_start is None and plant.current_time > 15.0 and 1.0 < plant.speed < 1.5:
      pulse_start = plant.vehicle.frame
    elapsed = (plant.vehicle.frame - pulse_start) * 0.01 if pulse_start is not None else -1.0
    # A bounded, finite-duration single-wheel measurement excursion matching
    # the speed scale in bookmark 3. The plant itself continues braking smoothly.
    pulse = pulse_amplitude * math.sin(math.pi * elapsed / 0.12) if 0.0 < elapsed < 0.12 else 0.0
    return pulse

  plant.vehicle.wheel_speed_error = wheel_sensor_excursion
  rows = []
  while plant.current_time < 195.0:
    elapsed = max(0.0, plant.current_time - 15.0)
    # A single continuously decelerating lead, then a crawl and final stop.
    lead_speed, lead_distance = lead_state(elapsed)
    gap = plant.distance_lead - plant.distance
    # Small deterministic radar perturbations, including negative estimated
    # lead velocity at standstill; true lead velocity always stays >= 0.
    disturbance = 0.12 * math.sin(2.0 * math.pi * 0.7 * elapsed) * (elapsed > 0.0)
    # A sinusoid of amplitude A has standard deviation A/sqrt(2). Reporting
    # zero uncertainty while injecting this disturbance falsely claims an
    # exactly known moving/stopped lead and bypasses confidence-based stopping.
    # Keep all gates unchanged; explicit zero-uncertainty counterfactuals remain
    # available via model_speed_std=0.
    lead = {"v_lead": lead_speed + disturbance,
            "model_speed_std": model_speed_std,
            "model_lead_future": [lead_state(elapsed + future)[0] for future in ModelConstants.LEAD_T_IDXS],
            "model_lead_future_x": [gap + RADAR_TO_CAMERA + lead_state(elapsed + future)[1] - lead_distance
                                    for future in ModelConstants.LEAD_T_IDXS],
            "d_rel": gap + 0.25 * math.sin(2.0 * math.pi * 0.9 * elapsed),
            "a_lead": -1.5 if 0.0 < elapsed < braking_time else -0.3 if elapsed > braking_time + 6.0
            and lead_speed > 0.0 else 0.0}
    plant.step(v_lead=lead_speed, v_cruise=speed, lead_one=lead, lead_two=lead)
    debug = plant.planner.crv_debug
    rows.append((plant.current_time, plant.speed / MPH, plant.distance_lead - plant.distance,
                 gap / max(plant.speed, 0.1), plant.acceleration, lead_speed / MPH,
                 plant.planner_acceleration, plant.gas_command, plant.brake_intensity,
                 plant.brake_request, plant.actuator_mode, plant.mode_transitions,
                 plant.predictive_brake, plant.safety_override,
                 getattr(plant.planner, 'crv_stop_phase', 0), *getattr(plant.planner, 'crv_stop_reference', (0., 0., 0.)),
                 getattr(plant.planner, 'crv_required_clearance', 0.), getattr(plant.planner, 'crv_lead_follow_urgency', 0.),
                 plant.vehicle.measured_speed / MPH,
                 plant.vehicle.measured_acceleration, plant.vehicle.output,
                 debug.get("safetyEgoSpeed", 0.), debug.get("safetyClosingSpeed", 0.),
                 debug.get("safetyAcceleration", 0.), debug.get("delayedClosingSpeed", 0.),
                 debug.get("speedStd", 0.), debug.get("accelerationStd", 0.),
                 debug.get("wheelChannelCount", 4)))
  return np.asarray(rows, dtype=object)


def stop_metrics(rows):
  t, speed, gap, accel, planner = (rows[:, i].astype(float) for i in (0, 1, 2, 4, 6))
  low = (t > 15.0) & (speed > 0.3) & (speed < 5.0)
  # A low-speed brake reapplication after deceleration has already tapered
  # is what the drive felt like. Mode-only counting misses this completely.
  indices = np.flatnonzero(low)
  renewed_brake = 0.0
  if len(indices):
    command = planner[indices]
    # Count braking that returns AFTER the command has materially released,
    # not the total change from the beginning of an otherwise monotone stop.
    released = command - np.minimum.accumulate(command)
    reapplied = command - np.minimum.accumulate(command[::-1])[::-1]
    renewed_brake = float(np.max(np.where(released > 0.0, reapplied, 0.0)))
  tail = t >= t[-1] - 60.0
  return {"low_speed_brake_reapplication": renewed_brake,
          "low_speed_peak_deceleration": float(-np.min(accel[low])) if np.any(low) else 0.0,
          "minimum_clearance": float(np.min(gap)), "final_clearance": float(gap[-1]),
          "final_speed": float(np.max(speed[tail])),
          "settled_gas_span": float(np.ptp(rows[tail, 7].astype(float))),
          "settled_brake_span": float(np.ptp(rows[tail, 8].astype(float)))}


def stop_failures(metrics):
  bounds = {"minimum_clearance": (2.0, math.inf),
            "low_speed_brake_reapplication": (0.0, 0.8),
            "low_speed_peak_deceleration": (0.0, 2.0),
            "final_speed": (0.0, 0.1), "final_clearance": (2.0, 4.5),
            "settled_gas_span": (0.0, 0.05), "settled_brake_span": (0.0, 0.05)}
  return [key for key, (lower, upper) in bounds.items() if not lower <= metrics[key] < upper]


@pytest.mark.parametrize("ego_mph", (25, 35, 50))
def test_smooth_rolling_stop_rejects_sensor_triggered_brake_reapplication(ego_mph):
  metrics = stop_metrics(run_multiphase_stop(ego_mph))
  assert not stop_failures(metrics), metrics


@pytest.mark.parametrize("pulse_amplitude", (0.0, 0.16, 0.30))
def test_stop_quality_is_not_specific_to_one_observer_excursion(pulse_amplitude):
  metrics = stop_metrics(run_multiphase_stop(50, pulse_amplitude=pulse_amplitude))
  assert not stop_failures(metrics), metrics
