"""Regression for CR-V resuming too close behind a stopped lead."""

from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_safe_obstacle_distance, get_stopped_equivalence_factor, get_T_FOLLOW
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvPostRestartGapRegression(OpenpilotTestCase):
  def test_restart_does_not_accelerate_before_the_gap_opens(self):
    """A lead restart must not make the CR-V accelerate through a close gap."""
    plant = Plant(lead_relevancy=True, speed=0.0, distance_lead=2.7, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    # Establish the observed state: ego is stopped, the lead is close, and
    # the lead then starts moving while remaining continuously tracked.
    for _ in range(50):
      plant.step(v_lead=0.0, prob_lead=1.0, v_cruise=20.0)

    close_gap_commands = []
    opened_gap_commands = []
    for _ in range(200):
      plant.step(v_lead=2.0, prob_lead=1.0, v_cruise=20.0)
      d_rel = plant.distance_lead - plant.distance
      release_distance = get_safe_obstacle_distance(plant.speed, get_T_FOLLOW()) \
        - get_stopped_equivalence_factor(2.0)
      if d_rel < release_distance:
        close_gap_commands.append((d_rel, plant.planner.output_a_target, plant.speed))
      else:
        opened_gap_commands.append((d_rel, plant.planner.output_a_target, plant.speed))

    assert close_gap_commands, "the lead never opened the test gap"
    assert opened_gap_commands, "the lead never opened the test gap"
    assert all(accel <= 0.0 for _, accel, _ in close_gap_commands), close_gap_commands
    assert any(accel > 0.0 for _, accel, _ in opened_gap_commands), opened_gap_commands
