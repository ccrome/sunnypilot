"""Regression for CR-V creeping toward a close, tracked lead."""

from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvCloseClosingStopRegression(OpenpilotTestCase):
  def test_close_closing_lead_enters_stopping_state(self):
    """A 1.4 mph CR-V closing at 2.4 m must stop before driver takeover."""
    plant = Plant(lead_relevancy=True, speed=0.63, distance_lead=2.4, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    plant.step(v_lead=0.28, prob_lead=1.0, v_cruise=20.0)

    assert plant.planner.output_should_stop
    assert plant.planner.output_a_target <= 0.0
