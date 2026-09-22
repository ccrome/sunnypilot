#!/usr/bin/env python3
import time
import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU


class Plant:
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               enabled=True, only_lead2=False, only_radar=False, e2e=False, personality=0, force_decel=False, car_fingerprint=None):
    self.rate = 1. / DT_MDL

    if not Plant.messaging_initialized:
      Plant.radar = messaging.pub_sock('radarState')
      Plant.controls_state = messaging.pub_sock('controlsState')
      Plant.selfdrive_state = messaging.pub_sock('selfdriveState')
      Plant.car_state = messaging.pub_sock('carState')
      Plant.plan = messaging.sub_sock('longitudinalPlan')
      Plant.messaging_initialized = True

    self.v_lead_prev = 0.0

    self.distance = 0.
    self.speed = speed
    self.should_stop = False
    self.acceleration = 0.0

    # lead car
    self.lead_relevancy = lead_relevancy
    self.distance_lead = distance_lead
    self.enabled = enabled
    self.only_lead2 = only_lead2
    self.only_radar = only_radar
    self.e2e = e2e
    self.personality = personality
    self.force_decel = force_decel

    self.rk = Ratekeeper(self.rate, print_delay_threshold=100.0)
    self.ts = 1. / self.rate
    time.sleep(0.1)
    self.sm = messaging.SubMaster(['longitudinalPlan'])

    from opendbc.car.honda.values import CAR
    from opendbc.car.honda.interface import CarInterface

    car_fingerprint = CAR.HONDA_CIVIC if car_fingerprint is None else car_fingerprint
    CP = CarInterface.get_non_essential_params(car_fingerprint)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, car_fingerprint)
    self.planner = LongitudinalPlanner(CP, CP_SP, init_v=self.speed)

  @property
  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, prob_lead=1.0, v_cruise=50., pitch=0.0, prob_throttle=1.0,
           lead_one=None, lead_two=None):
    # ******** publish a fake model going straight and fake calibration ********
    # note that this is worst case for MPC, since model will delay long mpc by one time step
    radar = messaging.new_message('radarState')
    control = messaging.new_message('controlsState')
    ss = messaging.new_message('selfdriveState')
    car_state = messaging.new_message('carState')
    lp = messaging.new_message('vehicleParameters')
    car_control = messaging.new_message('carControl')
    model = messaging.new_message('modelV2')
    car_state_sp = messaging.new_message('carStateSP')
    live_map_data_sp = messaging.new_message('liveMapDataSP')
    gps_data = messaging.new_message('gpsLocation')
    a_lead = (v_lead - self.v_lead_prev)/self.ts
    self.v_lead_prev = v_lead

    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      if self.only_radar:
        status = True
      elif prob_lead > .5:
        status = True
      else:
        status = False
    else:
      d_rel = 200.
      prob_lead = 0.0
      status = False

    def make_lead(spec):
      spec = {} if spec is None else spec
      candidate_v_lead = float(spec.get('v_lead', v_lead))
      candidate_prob = float(spec.get('prob', prob_lead))
      candidate_present = bool(spec.get('present', status))
      candidate_d_rel = float(spec.get('d_rel', d_rel))
      candidate_v_rel = float(spec.get('v_rel', candidate_v_lead - self.speed))
      candidate_a_lead = float(spec.get('a_lead', a_lead))

      candidate = log.RadarState.LeadData.new_message()
      candidate.dRel = candidate_d_rel
      candidate.yRel = float(spec.get('y_rel', 0.0))
      candidate.vRel = candidate_v_rel
      candidate.vLead = candidate_v_lead
      candidate.vLeadK = candidate_v_lead
      candidate.aLeadK = candidate_a_lead
      # TODO use real radard logic for this
      candidate.aLeadTau = float(spec.get('a_lead_tau', _LEAD_ACCEL_TAU))
      candidate.present = candidate_present
      candidate.modelProb = candidate_prob
      candidate.radar = bool(spec.get('radar', True))
      return candidate

    if not self.only_lead2:
      radar.radarState.leadOne = make_lead(lead_one)
    radar.radarState.leadTwo = make_lead(lead_two)

    # Simulate model predicting slightly faster speed
    # this is to ensure lead policy is effective when model
    # does not predict slowdown in e2e mode
    position = log.XYZTData.new_message()
    position.x = [float(x) for x in (self.speed + 0.5) * np.array(ModelConstants.T_IDXS)]
    model.modelV2.position = position
    model.modelV2.action.desiredAcceleration = float(self.acceleration + 0.5)
    velocity = log.XYZTData.new_message()
    velocity.x = [float(x) for x in (self.speed + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
    velocity.x[0] = float(self.speed) # always start at current speed
    model.modelV2.velocity = velocity
    acceleration = log.XYZTData.new_message()
    acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
    model.modelV2.acceleration = acceleration
    model.modelV2.meta.disengagePredictions.gasPressProbs = [float(prob_throttle) for _ in range(6)]

    control.controlsState.longControlState = LongCtrlState.pid if self.enabled else LongCtrlState.off
    ss.selfdriveState.enabled = self.enabled
    ss.selfdriveState.experimentalMode = self.e2e
    ss.selfdriveState.personality = self.personality
    control.controlsState.forceDecel = self.force_decel
    car_state.carState.vEgo = float(self.speed)
    car_state.carState.standstill = bool(self.speed < 0.01)
    car_state.carState.vCruise = float(v_cruise * 3.6)
    car_control.carControl.orientationNED = [0., float(pitch), 0.]

    # ******** get controlsState messages for plotting ***
    sm = {'radarState': radar.radarState,
          'carState': car_state.carState,
          'carControl': car_control.carControl,
          'controlsState': control.controlsState,
          'selfdriveState': ss.selfdriveState,
          'vehicleParameters': lp.vehicleParameters,
          'modelV2': model.modelV2,
          'carStateSP': car_state_sp.carStateSP,
          'liveMapDataSP': live_map_data_sp.liveMapDataSP,
          'gpsLocation': gps_data.gpsLocation}
    self.planner.update(sm)
    self.acceleration = self.planner.output_a_target
    if self.planner.output_should_stop:
      self.acceleration = min(-0.5, self.acceleration)
    self.speed = self.speed + self.acceleration * self.ts
    self.should_stop = self.planner.output_should_stop
    fcw = self.planner.fcw
    self.distance_lead = self.distance_lead + v_lead * self.ts

    # ******** run the car ********
    #print(self.distance, speed)
    if self.speed <= 0:
      self.speed = 0
      self.acceleration = 0
    self.distance = self.distance + self.speed * self.ts

    # *** radar model ***
    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
    else:
      d_rel = 200.

    # print at 5hz
    # if (self.rk.frame % (self.rate // 5)) == 0:
    #   print("%2.2f sec   %6.2f m  %6.2f m/s  %6.2f m/s2   lead_rel: %6.2f m  %6.2f m/s"
    #         % (self.current_time, self.distance, self.speed, self.acceleration, d_rel, v_rel))


    # ******** update prevs ********
    self.rk.monitor_time()

    return {
      "distance": self.distance,
      "speed": self.speed,
      "acceleration": self.acceleration,
      "should_stop": self.should_stop,
      "distance_lead": self.distance_lead,
      "fcw": fcw,
    }

# simple engage in standalone mode
def plant_thread():
  plant = Plant()
  while 1:
    plant.step()


if __name__ == "__main__":
  plant_thread()
