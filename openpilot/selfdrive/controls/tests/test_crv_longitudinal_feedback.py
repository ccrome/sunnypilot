"""CR-V longitudinal acceleration-feedback regression."""

from types import SimpleNamespace

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longcontrol import LongControl


class TestCrvLongitudinalFeedback(OpenpilotTestCase):
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
