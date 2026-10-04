"""Controller-side stopping invariants, independent of vehicle pass gates."""
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.terminal_stop import TerminalStop, vision_motion_confidence


def update(stop, v=5., gap=20., moving=False, standstill=False, enabled=True, coast_accel=-.2):
  return stop.update(v, gap, True, moving, standstill, .05, enabled,
                     coast_accel=coast_accel, predicted=True)


@pytest.mark.parametrize('speed', (.2, .6, 3., 10., 25.))
@pytest.mark.parametrize('gap', (4., 10., 50., 100.))
def test_stop_reference_is_nonnegative_and_monotone(speed, gap):
  stop = TerminalStop()
  update(stop, speed, gap)
  values = np.array([stop.reference(t) for t in np.linspace(0., stop.duration, 101)])
  assert np.min(values[:, 0]) >= -1e-10
  assert np.max(values[:, 1]) <= 1e-10
  assert np.max(np.diff(values[:, 0])) <= 1e-10
  assert np.allclose(values[-1], 0., atol=1e-10)


def test_wheel_dropout_does_not_erase_remaining_motion():
  stop = TerminalStop()
  before = update(stop, .6, 4.5)[0]
  after = update(stop, 0., 4.5)[0]
  assert 0. < after < before
  assert stop.phase == TerminalStop.BRAKING
  update(stop, 0., 4.5, standstill=True)
  assert stop.phase == TerminalStop.HOLDING


def test_genuine_restart_and_disengagement_release_stop_commitment():
  stop = TerminalStop()
  update(stop)
  assert stop.phase == TerminalStop.BRAKING
  # A present moving lead releases a committed stop even with stale intent.
  stop.update(5., 20., False, True, False, .05, True)
  assert stop.phase == TerminalStop.INACTIVE
  update(stop)
  update(stop, enabled=False)
  assert stop.phase == TerminalStop.INACTIVE


def test_terminal_release_of_pressure_never_opens_gas_or_resets_integral():
  cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
  controller = LongControl(cp, sp)
  controller.pid.i = .15
  state = SimpleNamespace(vEgo=.4, aEgo=-4., standstill=False, brakePressed=False,
                          cruiseState=SimpleNamespace(standstill=False))
  for _ in range(100):
    output = controller.update(True, state, -.1, False, (-3.5, 2.), stop_phase=1, v_target=.5, j_target=1.)
    assert output < 0.
    assert controller.pid.i == .15
  assert controller.long_control_state == LongCtrlState.pid  # Wheel dropout is not a physical hold.


@pytest.mark.parametrize('stop_phase', (0, 1))
@pytest.mark.parametrize('should_stop', (False, True))
def test_emergency_request_survives_feedforward_integral_and_stop_ramp(stop_phase, should_stop):
  cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
  controller = LongControl(cp, sp)
  controller.pid.i = .5
  state = SimpleNamespace(vEgo=1., aEgo=-.2, standstill=False, brakePressed=False,
                          cruiseState=SimpleNamespace(standstill=False))
  output = controller.update(True, state, -3.5, should_stop, (-3.5, 2.),
                             stop_phase=stop_phase, j_target=20., safety_pressure=1.)
  assert output == -3.5
  assert .47 < controller.pid.i <= .5  # Continuous correction, never a reset.


def test_disengaged_controller_ignores_stale_safety_pressure():
  cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
  controller = LongControl(cp, sp)
  state = SimpleNamespace(vEgo=1., aEgo=0., standstill=False, brakePressed=False,
                          cruiseState=SimpleNamespace(standstill=False))
  assert controller.update(False, state, -3.5, False, (-3.5, 2.), safety_pressure=1.) == 0.


@pytest.mark.parametrize('tail_speed,tail_std,expected', ((0., 0., True), (.06, .07, True), (.45, .01, False)))
def test_prediction_distinguishes_a_stop_from_a_slow_rolling_lead(tail_speed, tail_std, expected):
  lead = SimpleNamespace(prob=1., v=[2., 1., tail_speed], vStd=[.01, .01, tail_std],
                         t=[0., 1., 2.], x=[20., 22., 23.])
  result = vision_motion_confidence(SimpleNamespace(leadsV3=[lead]), 0)
  assert result[2] == expected
  assert result[3] >= 0.
