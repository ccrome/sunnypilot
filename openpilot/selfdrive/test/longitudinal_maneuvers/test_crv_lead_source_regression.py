"""Regression case for a CR-V-like close-lead source dropout.

The planner must keep a short safety hold when a close, closing lead disappears
briefly during tracker switching.
"""

from openpilot.common.test import OpenpilotTestCase
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvLeadSourceRegression(OpenpilotTestCase):
  def test_close_closing_lead_dropout_never_allows_acceleration(self):
    """A short lead-source dropout must not authorize acceleration into a close lead.

    The setup is a compact reproduction of the route-5c 640--710 s failure:
    a close, slower lead is established, then both tracker candidates disappear
    briefly. The planner may report cruise as its source, but must not
    accelerate while the car remains close and closing.
    """
    v_ego = 7.0
    v_lead = 4.5
    plant = Plant(lead_relevancy=True, speed=v_ego, distance_lead=12.0, e2e=False, car_fingerprint=CAR.HONDA_CRV_5G)

    # Establish a credible, slower lead before the simulated source dropout.
    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=1.0, v_cruise=20.0)

    unsafe_outputs = []
    sources = []
    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=0.0, v_cruise=20.0)
      d_rel = plant.distance_lead - plant.distance
      v_rel = v_lead - plant.speed
      sources.append(plant.planner.mpc.source)
      if d_rel < 11.0 and v_rel < -1.0 and plant.planner.output_a_target > 0.0:
        unsafe_outputs.append((d_rel, v_rel, plant.planner.output_a_target))

    assert LongitudinalPlanSource.cruise in sources
    assert not unsafe_outputs, unsafe_outputs
