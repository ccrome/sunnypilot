"""CR-V longitudinal acceleration-feedback regression."""

from types import SimpleNamespace

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longcontrol import LongControl


class TestCrvLongitudinalFeedback(OpenpilotTestCase):
  def test_positive_maneuver_does_not_brake_for_one_acceleration_sample(self):
    """Latest-drive 168.432 s: positive target, transient high aEgo."""
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
    controller = LongControl(cp, sp)
    controller.pid.i = -0.146
    state = SimpleNamespace(aEgo=.607, vEgo=8.694, brakePressed=False,
                            cruiseState=SimpleNamespace(standstill=False))
    output = controller.update(True, state, .107, False, (-3.5, 2.0))
    assert output > 0
    assert controller.pid.p == 0
    assert -.15 < controller.pid.i < -.146  # Correction accumulates, never resets.

  def test_zero_mean_acceleration_disturbance_does_not_cycle_pedals(self):
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
    for speed in (5., 15., 30.):
      controller = LongControl(cp, sp)
      state = SimpleNamespace(aEgo=0., vEgo=speed, brakePressed=False,
                              cruiseState=SimpleNamespace(standstill=False))
      outputs = []
      for n in range(200):
        state.aEgo = .1 + (.5 if n % 2 else -.5)
        outputs.append(controller.update(True, state, .1, False, (-3.5, 2.)))
      assert min(outputs) > 0
      assert max(outputs) - min(outputs) < .002
      assert abs(controller.pid.i) < 1e-9

  def test_low_speed_brake_undershoot_accumulates_bounded_negative_feedback(self):
    """Persistent low-speed creep must strengthen a negative brake request."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.HONDA_CRV_5G)
    controller = LongControl(CP, CP_SP)
    car_state = SimpleNamespace(
      aEgo=0.05,
      vEgo=0.63,
      brakePressed=False,
      cruiseState=SimpleNamespace(standstill=False),
    )

    first_output = controller.update(True, car_state, -0.25, False, (-3.0, 2.0))
    for _ in range(99):
      output = controller.update(True, car_state, -0.25, False, (-3.0, 2.0))

    assert controller.pid.i < -0.01
    assert output < first_output - 0.01

  def test_high_speed_acceleration_error_accumulates_small_bounded_feedback(self):
    """High-speed feed-forward error must receive a small integral correction."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.HONDA_CRV_5G)
    controller = LongControl(CP, CP_SP)
    car_state = SimpleNamespace(
      aEgo=0.10,
      vEgo=30.0,
      brakePressed=False,
      cruiseState=SimpleNamespace(standstill=False),
    )

    first_output = controller.update(True, car_state, -0.10, False, (-3.0, 2.0))
    for _ in range(199):
      output = controller.update(True, car_state, -0.10, False, (-3.0, 2.0))

    assert controller.pid.i < -0.005
    assert output < first_output - 0.005
    assert output >= -3.0
