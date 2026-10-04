"""Debug signals must survive route-log serialization and clear stale inputs."""
from types import SimpleNamespace

import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


@pytest.mark.parametrize('active,stop_phase,safety_pressure', ((True, 0, 0.), (True, 1, 1.), (False, 1, 1.)))
def test_actuator_debug_roundtrip(active, stop_phase, safety_pressure):
  cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_CRV_5G)
  controller = LongControl(cp, sp)
  controller.pid.i = .15
  state = SimpleNamespace(vEgo=.4, aEgo=-.2, standstill=False, brakePressed=False,
                          cruiseState=SimpleNamespace(standstill=False))
  output = controller.update(active, state, -3.5, False, (-3.5, 2.),
                             stop_phase=stop_phase, j_target=20., safety_pressure=safety_pressure)
  event = messaging.new_message('controlsState')
  controller.write_debug(event.controlsState.init('longitudinalDebug'))
  with log.Event.from_bytes(event.to_bytes()) as received:
    debug = received.controlsState.longitudinalDebug
    assert debug.active == active
    assert debug.terminalStop == bool(stop_phase)
    assert debug.safetyPressure == safety_pressure * active
    assert debug.outputAcceleration == pytest.approx(output)
    assert debug.integral == pytest.approx(controller.pid.i)
    assert debug.feedforward == pytest.approx(controller.pid.f)
    assert debug.pidUpdated == active
    assert debug.targetJerk == 20.


def test_planner_debug_roundtrip_and_departed_lead_clears_safety_inputs():
  plant = Plant(lead_relevancy=True, speed=5., distance_lead=15., physics=True, realtime=False,
                car_fingerprint=CAR.HONDA_CRV_5G)
  plant.step(v_lead=0., v_cruise=5., prob_lead=1.)
  event = messaging.new_message('longitudinalPlan')
  plant.planner.write_crv_debug(event.longitudinalPlan.init('crvDebug'))
  with log.Event.from_bytes(event.to_bytes()) as received:
    debug = received.longitudinalPlan.crvDebug
    assert debug.leadActive
    assert debug.stationaryLead
    assert debug.leadGap > 0.
    assert debug.safetyEgoSpeed > 0.
    assert debug.responseTime > 0.
    assert debug.accelerationJerk == pytest.approx(plant.planner.crv_accel_jerk)
    assert debug.stopReferenceAcceleration == pytest.approx(plant.planner.crv_stop_reference[1])
  plant.step(v_lead=0., v_cruise=5., prob_lead=0.)
  event = messaging.new_message('longitudinalPlan')
  plant.planner.write_crv_debug(event.longitudinalPlan.init('crvDebug'))
  debug = event.longitudinalPlan.crvDebug
  assert not debug.leadActive
  assert debug.leadGap == debug.safetyEgoSpeed == debug.safetyAcceleration == debug.responseTime == 0.
  assert plant.planner.crv_required_clearance == 0.
