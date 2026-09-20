"""Regressions for CR-V brief tracker-loss acceleration safety."""

from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvBriefTrackerLossRegression(OpenpilotTestCase):
  def test_low_speed_close_lead_loss_never_allows_acceleration(self):
    """A short tracker loss must not accelerate toward a 12 m, closing lead."""
    v_ego = 7.0
    v_lead = 4.5
    plant = Plant(lead_relevancy=True, speed=v_ego, distance_lead=12.0, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

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

  def test_moderate_speed_closing_lead_loss_never_allows_acceleration(self):
    """A 32 mph CR-V must not accelerate into a 16 m, 2.3 m/s closing lead.

    This verifies that tracker-loss protection applies beyond close, low-speed
    following.
    """
    v_ego = 14.3
    v_lead = 12.0
    plant = Plant(lead_relevancy=True, speed=v_ego, distance_lead=16.0, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=1.0, v_cruise=30.0)

    unsafe_outputs = []
    sources = []
    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=0.0, v_cruise=30.0)
      d_rel = plant.distance_lead - plant.distance
      v_rel = v_lead - plant.speed
      sources.append(plant.planner.mpc.source)
      if d_rel < 16.0 and v_rel < -2.0 and plant.planner.output_a_target > 0.0:
        unsafe_outputs.append((d_rel, v_rel, plant.planner.output_a_target))

    assert LongitudinalPlanSource.cruise in sources
    assert not unsafe_outputs, unsafe_outputs

  def test_highway_speed_closing_lead_loss_never_allows_acceleration(self):
    """A 56 mph CR-V must not accelerate into a 26 m, 5 m/s closing lead.

    This verifies that tracker-loss protection applies at highway speed.
    """
    v_ego = 25.0
    v_lead = 20.0
    plant = Plant(lead_relevancy=True, speed=v_ego, distance_lead=26.0, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=1.0, v_cruise=35.0)

    unsafe_outputs = []
    sources = []
    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=0.0, v_cruise=35.0)
      d_rel = plant.distance_lead - plant.distance
      v_rel = v_lead - plant.speed
      sources.append(plant.planner.mpc.source)
      if d_rel < 26.0 and v_rel < -4.0 and plant.planner.output_a_target > 0.0:
        unsafe_outputs.append((d_rel, v_rel, plant.planner.output_a_target))

    assert LongitudinalPlanSource.cruise in sources
    assert not unsafe_outputs, unsafe_outputs

  def test_high_speed_closing_lead_loss_never_allows_acceleration(self):
    """A 72 mph CR-V must not accelerate into a 45 m, 4.3 m/s closing lead.

    This covers only a brief tracker loss; tracked-lead late braking remains a
    separate L3 concern.
    """
    v_ego = 32.0
    v_lead = 27.7
    plant = Plant(lead_relevancy=True, speed=v_ego, distance_lead=45.0, e2e=False,
                  car_fingerprint=CAR.HONDA_CRV_5G)

    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=1.0, v_cruise=35.0)

    unsafe_outputs = []
    sources = []
    for _ in range(10):
      plant.step(v_lead=v_lead, prob_lead=0.0, v_cruise=35.0)
      d_rel = plant.distance_lead - plant.distance
      v_rel = v_lead - plant.speed
      sources.append(plant.planner.mpc.source)
      if d_rel < 45.0 and v_rel < -4.0 and plant.planner.output_a_target > 0.0:
        unsafe_outputs.append((d_rel, v_rel, plant.planner.output_a_target))

    assert LongitudinalPlanSource.cruise in sources
    assert not unsafe_outputs, unsafe_outputs
