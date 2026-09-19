from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.honda.longitudinal_tuning import CRV_LONGITUDINAL_TUNE
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces


class TestHondaLongitudinalTuning(OpenpilotTestCase):
  def test_crv_initializes_fixed_safe_tune(self):
    params = Params()
    CarInterface = interfaces[CAR.HONDA_CRV_5G]
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.HONDA_CRV_5G)
    CI = CarInterface(CP, CP_SP)

    sunnypilot_interfaces.setup_interfaces(CI, params)

    assert [round(v, 3) for v in CI.CP.longitudinalTuning.kiBP] == [0.0, 5.0, 35.0]
    assert [round(v, 3) for v in CI.CP.longitudinalTuning.kiV] == [0.0, 0.0, 0.0]
    assert round(CI.CP.longitudinalActuatorDelay, 3) == 0.8
    assert CI.CP_SP.hondaCrvLongitudinalTune.enabled
    assert CI.CP_SP.hondaCrvLongitudinalTune.tuneId == "safeV1"
    assert CI.CP_SP.hondaCrvLongitudinalTune.revision == 1
    assert CI.CP_SP.hondaCrvLongitudinalTune.followingTime == 1.75

  def test_fixed_following_time_matches_relaxed_mpc_value(self):
    assert CRV_LONGITUDINAL_TUNE.following_time == get_T_FOLLOW(log.LongitudinalPersonality.relaxed)
