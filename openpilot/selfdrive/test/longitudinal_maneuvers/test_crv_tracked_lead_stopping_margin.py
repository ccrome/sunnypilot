"""Regression tests for CR-V tracked-lead stopping-margin protection."""

from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvTrackedLeadStoppingMargin(OpenpilotTestCase):
  def test_close_closing_lead_never_authorizes_positive_acceleration(self):
    """A continuously tracked lead must not receive a positive command near stop."""
    plant = Plant(lead_relevancy=True, speed=1.4, distance_lead=2.7, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    unsafe_outputs = []
    distances = []
    for _ in range(20):
      result = plant.step(v_lead=1.0, prob_lead=1.0, v_cruise=20.0)
      d_rel = result["distance_lead"] - result["distance"]
      v_rel = 1.0 - plant.speed
      distances.append(d_rel)
      if d_rel < 3.0 and v_rel < -0.1 and plant.planner.output_a_target > 0.0:
        unsafe_outputs.append((d_rel, v_rel, plant.planner.output_a_target))

    assert not unsafe_outputs, unsafe_outputs
    assert min(distances) >= 2.0, min(distances)

  def test_opening_lead_is_not_braked_by_margin_guard(self):
    """The guard must release when the continuously tracked lead is opening."""
    plant = Plant(lead_relevancy=True, speed=1.4, distance_lead=2.7, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    outputs = []
    for _ in range(10):
      plant.step(v_lead=2.0, prob_lead=1.0, v_cruise=20.0)
      outputs.append(plant.planner.output_a_target)

    assert max(outputs) > 0.0, max(outputs)
