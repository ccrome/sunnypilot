"""Regression case for CR-V stop release after a close lead-source loss.

The planner must keep a short stop hold when a close stopped lead disappears
briefly during tracker switching.
"""

from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvStopReleaseRegression(OpenpilotTestCase):
  def test_stopped_close_lead_dropout_does_not_authorize_restart(self):
    """A stopped lead lost at 2.7 m must not immediately trigger a restart.

    This is a compact reproduction of route 1e around 2808 s: after a long
    stop behind a lead roughly 2.7 m away, the source changes to cruise and
    the planner may select cruise, but must not restart while the physical lead
    remains close.
    """
    plant = Plant(lead_relevancy=True, speed=0.0, distance_lead=2.7, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    # Establish a stopped, close lead before the simulated tracker dropout.
    for _ in range(20):
      plant.step(v_lead=0.0, prob_lead=1.0, v_cruise=20.0)

    unsafe_outputs = []
    sources = []
    for _ in range(10):
      plant.step(v_lead=0.0, prob_lead=0.0, v_cruise=20.0)
      d_rel = plant.distance_lead - plant.distance
      sources.append(plant.planner.mpc.source)
      if d_rel < 3.0 and (plant.planner.output_a_target > 0.0 or plant.speed > 0.0 or not plant.planner.output_should_stop):
        unsafe_outputs.append((d_rel, plant.planner.output_a_target, plant.speed, plant.planner.output_should_stop))

    assert LongitudinalPlanSource.cruise in sources
    assert not unsafe_outputs, unsafe_outputs
