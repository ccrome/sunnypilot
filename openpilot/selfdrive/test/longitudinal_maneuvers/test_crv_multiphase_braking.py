"""Oct. 3 drive regression: a smooth lead stop must not become brake hunting.

The recorded low-speed raw-speed excursion was about 0.2 m/s over 0.1 s.
Inject that excursion in transmission speed BEFORE production fusion/KF,
not as a physical acceleration or forced ego trajectory. Distance is causal.
"""
import math

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant
from openpilot.selfdrive.test.longitudinal_maneuvers.test_crv_longitudinal_quality_regression import MPH


def run_multiphase_stop(ego_mph, actuator_parameters=None, pulse_amplitude=0.22):
  speed = ego_mph * MPH
  plant = Plant(lead_relevancy=True, speed=speed, distance_lead=2.05 * speed,
                physics=True, realtime=False, car_fingerprint=CAR.HONDA_CRV_5G,
                personality=log.LongitudinalPersonality.relaxed, actuator_parameters=actuator_parameters)
  pulse_start = None

  def transmission_excursion():
    nonlocal pulse_start
    if pulse_start is None and plant.current_time > 15.0 and 1.0 < plant.speed < 1.5:
      pulse_start = plant.vehicle.frame
    elapsed = (plant.vehicle.frame - pulse_start) * 0.01 if pulse_start is not None else -1.0
    # A bounded, finite-duration measurement innovation, matching the raw
    # speed scale in bookmark 3. The plant itself continues braking smoothly.
    pulse = pulse_amplitude * math.sin(math.pi * elapsed / 0.12) if 0.0 < elapsed < 0.12 else 0.0
    return pulse

  plant.vehicle.transmission_speed_error = transmission_excursion
  rows = []
  while plant.current_time < 195.0:
    elapsed = max(0.0, plant.current_time - 15.0)
    # A single continuously decelerating lead, then a crawl and final stop.
    crawl = 1.5 * MPH
    braking_time = (speed - crawl) / 1.5
    lead_speed = max(crawl, speed - 1.5 * elapsed) if elapsed < braking_time + 6.0 else max(
      0.0, crawl - 0.3 * (elapsed - braking_time - 6.0))
    gap = plant.distance_lead - plant.distance
    # Small deterministic radar perturbations, including negative estimated
    # lead velocity at standstill; true lead velocity always stays >= 0.
    disturbance = 0.12 * math.sin(2.0 * math.pi * 0.7 * elapsed) * (elapsed > 0.0)
    lead = {"v_lead": lead_speed + disturbance,
            "d_rel": gap + 0.25 * math.sin(2.0 * math.pi * 0.9 * elapsed),
            "a_lead": -1.5 if 0.0 < elapsed < braking_time else -0.3 if elapsed > braking_time + 6.0
            and lead_speed > 0.0 else 0.0}
    plant.step(v_lead=lead_speed, v_cruise=speed, lead_one=lead, lead_two=lead)
    rows.append((plant.current_time, plant.speed / MPH, plant.distance_lead - plant.distance,
                 gap / max(plant.speed, 0.1), plant.acceleration, lead_speed / MPH,
                 plant.planner_acceleration, plant.gas_command, plant.brake_intensity,
                 plant.brake_request, plant.actuator_mode, plant.mode_transitions,
                 plant.predictive_brake, plant.safety_override, plant.vehicle.measured_speed / MPH,
                 plant.vehicle.measured_acceleration, plant.vehicle.output))
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
    renewed_brake = float(np.max(np.maximum.accumulate(command) - command))
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
