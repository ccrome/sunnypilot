"""Regression for CR-V no-lead cruise-speed integral feedback."""

from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvCruiseSpeedIntegral(OpenpilotTestCase):
  def test_no_lead_speed_error_builds_and_retains_a_bounded_cruise_bias(self):
    plant = Plant(lead_relevancy=True, speed=10.0, distance_lead=200.0, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    for _ in range(100):
      plant.step(v_lead=10.0, prob_lead=0.0, v_cruise=plant.speed + 0.1)
    learned_bias = plant.planner.crv_cruise_speed_i
    assert 0.01 < learned_bias <= 0.15

    plant.step(v_lead=10.0, prob_lead=0.0, v_cruise=plant.speed)
    assert plant.planner.crv_cruise_speed_i >= learned_bias - 0.001

  def test_tracked_lead_clears_the_cruise_speed_bias(self):
    plant = Plant(lead_relevancy=True, speed=10.0, distance_lead=200.0, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    for _ in range(100):
      plant.step(v_lead=10.0, prob_lead=0.0, v_cruise=plant.speed + 0.1)
    assert plant.planner.crv_cruise_speed_i > 0.01

    plant.step(v_lead=10.0, prob_lead=1.0, v_cruise=plant.speed + 0.1)
    assert plant.planner.crv_cruise_speed_i == 0.0
