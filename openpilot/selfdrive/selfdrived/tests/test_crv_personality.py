from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived.selfdrived import get_longitudinal_personality
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR


class TestCrvPersonality(OpenpilotTestCase):
  def test_crv_ignores_runtime_personality(self):
    params = Params()
    params.put("LongitudinalPersonality", log.LongitudinalPersonality.aggressive, block=True)
    CP = interfaces[CAR.HONDA_CRV_5G].get_non_essential_params(CAR.HONDA_CRV_5G)
    assert get_longitudinal_personality(CP, params) == log.LongitudinalPersonality.relaxed

  def test_other_cars_keep_runtime_personality(self):
    params = Params()
    params.put("LongitudinalPersonality", log.LongitudinalPersonality.aggressive, block=True)
    CP = interfaces[CAR.HONDA_CIVIC].get_non_essential_params(CAR.HONDA_CIVIC)
    assert get_longitudinal_personality(CP, params) == log.LongitudinalPersonality.aggressive
