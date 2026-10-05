"""Check the reachability calculation against integrated vehicle kinematics."""
import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import closing_braking_reachability


@pytest.mark.parametrize("speed", (0.0, 0.1, 1.0, 5.0, 15.0))
@pytest.mark.parametrize("acceleration", (-3.5, -2.0, 0.0, 1.5))
@pytest.mark.parametrize("delay", (0.0, 0.8))
@pytest.mark.parametrize("lead_speed", (0.0, 2.0, 10.0))
@pytest.mark.parametrize("headway", (1.0, 2.0))
def test_reachability_matches_integrated_ramp(speed, acceleration, delay, lead_speed, headway):
  # Independent fine-step integration, including an initially accelerating ego
  # at zero relative speed. End once velocity has matched after acceleration
  # becomes non-positive; no negative closing travel is accumulated.
  dt = 0.0001
  time = np.arange(0.0, delay + 10.0, dt)
  accel = np.maximum(-3.5, acceleration - 5.0 * np.maximum(0.0, time - delay))
  velocity = speed + np.cumsum(accel) * dt
  matched = np.flatnonzero((velocity <= 0.0) & (accel <= 0.0))
  end = int(matched[0]) if len(matched) else len(time)
  expected = float(np.sum(np.maximum(0.0, velocity[:end])) * dt)
  distance, _, clearance = closing_braking_reachability(
    speed, acceleration, delay, lead_speed=lead_speed, time_gap=headway)
  assert abs(distance - expected) < 0.005
  travel = np.r_[0.0, np.cumsum(np.maximum(0.0, velocity[:end])) * dt]
  ego_speed = np.r_[lead_speed + speed, lead_speed + np.maximum(0.0, velocity[:end])]
  expected_clearance = float(np.max(travel + np.maximum(2.0, headway * ego_speed)))
  assert abs(clearance - expected_clearance) < 0.01


def test_full_braking_already_committed_does_not_add_a_fresh_ramp():
  distance, _, _ = closing_braking_reachability(10.0, -3.5, 0.8)
  assert distance == pytest.approx(10.0 ** 2 / 7.0)


def test_delay_and_positive_acceleration_consume_more_clearance():
  nominal, _, _ = closing_braking_reachability(10.0, 0.0, 0.3)
  delayed, _, _ = closing_braking_reachability(10.0, 0.0, 0.8)
  accelerating, _, _ = closing_braking_reachability(10.0, 1.0, 0.8)
  assert nominal < delayed < accelerating


@pytest.mark.parametrize("speed", (1., 5., 15.))
@pytest.mark.parametrize("initial_accel", (-2., 0., 1.5))
@pytest.mark.parametrize("tau", (.1, .5, 1.))
@pytest.mark.parametrize("delay", (0., .8))
def test_tau_hold_bounds_first_order_brake_buildup(speed, initial_accel, tau, delay):
  """Independent ramp + first-order response, not another use of the helper.

  Velocity of a decreasing-acceleration reference is concave. Its exponential
  delay convolution is bounded above by shifting the reference by its mean
  delay tau (Jensen's inequality). The extra hold is therefore conservative.
  """
  b, jerk, dt = 3.2375, 5., .0002
  t = np.arange(0., 15., dt)
  u = np.maximum(0., t - delay)
  ramp_end = (initial_accel + b) / jerk
  ramp_accel = initial_accel - jerk * (u - tau + tau * np.exp(-u / tau))
  end_accel = initial_accel - jerk * (ramp_end - tau + tau * np.exp(-ramp_end / tau))
  actual = np.where(u <= ramp_end, ramp_accel,
                    -b + (end_accel + b) * np.exp(-np.maximum(0., u - ramp_end) / tau))
  velocity = speed + np.cumsum(actual) * dt
  end = int(np.flatnonzero(velocity <= 0.)[0])
  travel = np.cumsum(np.maximum(0., velocity[:end])) * dt
  clearance = travel + np.maximum(2., 2. + np.maximum(0., velocity[:end]))
  distance, _, required = closing_braking_reachability(speed, initial_accel, delay + tau,
                                                     max_decel=b, lead_speed=2.)
  assert distance >= travel[-1] - .01
  assert required >= np.max(clearance) - .01
