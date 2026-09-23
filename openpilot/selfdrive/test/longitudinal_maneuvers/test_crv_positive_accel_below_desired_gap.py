"""Regression for positive acceleration while a tracked lead gap is too short."""

from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_safe_obstacle_distance, get_stopped_equivalence_factor, get_T_FOLLOW,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvPositiveAccelBelowDesiredGap(OpenpilotTestCase):
  @staticmethod
  def desired_gap(v_ego: float, v_lead: float) -> float:
    return get_safe_obstacle_distance(v_ego, get_T_FOLLOW()) - get_stopped_equivalence_factor(max(v_lead, 0.0))

  def test_continuously_tracked_lead_does_not_accelerate_below_desired_gap(self):
    """A moving, continuously tracked lead must not receive positive accel below the desired gap."""
    plant = Plant(lead_relevancy=True, speed=8.0, distance_lead=15.0, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)
    below_gap_samples = 0
    unsafe_outputs = []
    for _ in range(30):
      plant.step(v_lead=8.0, prob_lead=1.0, v_cruise=8.0 * 3.6)
      d_rel = plant.distance_lead - plant.distance
      desired_gap = TestCrvPositiveAccelBelowDesiredGap.desired_gap(plant.speed, 8.0)
      if d_rel < desired_gap:
        below_gap_samples += 1
        if plant.planner.output_a_target > 0.0:
          unsafe_outputs.append((d_rel, desired_gap, plant.planner.output_a_target))

    assert below_gap_samples, "fixture no longer reaches the below-desired-gap condition"
    assert not unsafe_outputs, unsafe_outputs
