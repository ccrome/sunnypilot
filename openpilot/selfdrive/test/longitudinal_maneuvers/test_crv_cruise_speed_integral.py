"""CR-V cruise has one continuous integral in acceleration tracking."""

import math

import numpy as np

from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class TestCrvCruiseSpeedIntegral(OpenpilotTestCase):
  def test_cruise_reference_does_not_add_a_second_integral(self):
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    arguments = (False, 20. * CV.MPH_TO_MS, 19. * CV.MPH_TO_MS, 0., 0., cp, .05, -.3, 1.)
    unbiased = get_cruise_accel(*arguments, speed_error_bias=0.)
    biased = get_cruise_accel(*arguments, speed_error_bias=.5)
    assert unbiased == biased

  def test_downhill_cruise_settles_without_a_slow_mode_cycle_or_integral_reset(self):
    target = 20. * CV.MPH_TO_MS
    plant = Plant(lead_relevancy=True, speed=target, distance_lead=200., physics=True,
                  realtime=False, car_fingerprint=CAR.HONDA_CRV_5G, sim_rate=20.)
    rows = []
    for _ in range(2400):
      plant.step(v_lead=target, prob_lead=0., v_cruise=target, pitch=math.atan(-.03))
      if plant.current_time >= 60.:
        rows.append((plant.speed / CV.MPH_TO_MS, plant.mode_transitions))
    assert max(abs(speed - 20.) for speed, _ in rows) < 1.
    assert rows[-1][1] - rows[0][1] <= 2
    integral = plant.vehicle.longitudinal.pid.i
    assert np.isfinite(integral)

    # A newly tracked distant lead must not hard-reset the only integrator.
    plant.step(v_lead=target, prob_lead=1., v_cruise=target, pitch=math.atan(-.03))
    assert abs(plant.vehicle.longitudinal.pid.i - integral) < .01
