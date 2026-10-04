#!/usr/bin/env python3
import time
import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU, RADAR_TO_CAMERA, get_RadarState_from_vision
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.honda.carcontroller import compute_gas_brake
from opendbc.car.honda.hondacan import crv_brake_handoff, crv_gas_handoff
from opendbc.car.honda.values import CAR, CarControllerParams, HondaFlags


class Plant:
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               enabled=True, only_lead2=False, only_radar=False, e2e=False, personality=0, force_decel=False,
               car_fingerprint=None, physics=False, realtime=True, rolling_resistance=0.012,
               sim_rate=None, full_system=None, actuator_parameters=None):
    self.rate = float(sim_rate) if sim_rate is not None else 1. / DT_MDL

    if not Plant.messaging_initialized:
      Plant.radar = messaging.pub_sock('radarState')
      Plant.controls_state = messaging.pub_sock('controlsState')
      Plant.selfdrive_state = messaging.pub_sock('selfdriveState')
      Plant.car_state = messaging.pub_sock('carState')
      Plant.plan = messaging.sub_sock('longitudinalPlan')
      Plant.messaging_initialized = True

    self.v_lead_prev = 0.0
    self.lead_visible_prev = False

    self.distance = 0.
    self.speed = speed
    self.should_stop = False
    self.acceleration = 0.0
    self.planner_acceleration = 0.0
    self.gas_command = 0.0
    self.brake_request = False
    self.brake_intensity = 0.0
    self.actuator_mode = "coast"
    self.mode_transitions = 0
    self.predictive_brake = 0.0
    self.safety_override = False
    self._last_actuator_mode = self.actuator_mode
    self.physics = physics
    self.realtime = realtime
    self.rolling_resistance = rolling_resistance

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
    if realtime:
      time.sleep(0.1)
    self.sm = messaging.SubMaster(['longitudinalPlan'])

    from opendbc.car.honda.interface import CarInterface

    car_fingerprint = CAR.HONDA_CIVIC if car_fingerprint is None else car_fingerprint
    CP = CarInterface.get_non_essential_params(car_fingerprint)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, car_fingerprint)
    self.planner = LongitudinalPlanner(CP, CP_SP, init_v=self.speed)
    self.CP = CP
    self.full_system = car_fingerprint == CAR.HONDA_CRV_5G if full_system is None else full_system
    self.vehicle = None
    if self.full_system:
      from openpilot.selfdrive.test.longitudinal_maneuvers.honda_vehicle import HondaVehicle
      CP.openpilotLongitudinalControl = True
      self.vehicle = HondaVehicle(CP, CP_SP, self.speed, actuator_parameters)

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
    # Lead acquisition changes the reported speed reference; it is not a
    # physical one-frame acceleration of the tracked car.
    a_lead = (v_lead - self.v_lead_prev)/self.ts if self.lead_visible_prev else 0.0
    self.v_lead_prev = v_lead
    self.lead_visible_prev = bool(self.lead_relevancy and prob_lead > .5)

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

    # Optional vision observations go through the production ego/lead fusion.
    # True lead motion and range integration remain independent of perception.
    model_leads = []
    for name, spec in (("leadOne", lead_one), ("leadTwo", lead_two)):
      spec = spec or {}
      observation = log.ModelDataV2.LeadDataV3.new_message()
      observation.prob = float(spec.get('prob', prob_lead))
      observation.t = ModelConstants.LEAD_T_IDXS
      observation.x = spec.get('model_lead_future_x', [float(spec.get('d_rel', d_rel)) + RADAR_TO_CAMERA] * 6)
      observation.y = [0.] * 6
      observation.v = spec.get('model_lead_future', [float(spec.get('model_lead_speed', spec.get('v_lead', v_lead)))] * 6)
      observation.vStd = [float(spec.get('model_speed_std', 0.0))] * 6
      observation.a = [float(spec.get('a_lead', a_lead))] * 6
      model_leads.append(observation)
      if 'model_ego_speed' in spec:
        radar.radarState.__setattr__(name, get_RadarState_from_vision(
          observation, self.vehicle.measured_speed, float(spec['model_ego_speed']), observation.prob))
    model.modelV2.leadsV3 = model_leads

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

    control.controlsState.longControlState = (self.vehicle.longitudinal.long_control_state if self.full_system
                                              else LongCtrlState.pid if self.enabled else LongCtrlState.off)
    ss.selfdriveState.enabled = self.enabled
    ss.selfdriveState.experimentalMode = self.e2e
    ss.selfdriveState.personality = self.personality
    control.controlsState.forceDecel = self.force_decel
    car_state.carState.vEgo = float(self.vehicle.measured_speed if self.full_system else self.speed)
    car_state.carState.aEgo = float(self.vehicle.measured_acceleration if self.full_system else self.acceleration)
    car_state.carState.standstill = bool(self.speed < 0.01)
    car_state.carState.vEgoStd = float(self.vehicle.measured_speed_std if self.full_system else 0.0)
    car_state.carState.aEgoStd = float(self.vehicle.measured_acceleration_std if self.full_system else 0.0)
    car_state.carState.vEgoMeasurementValid = bool(self.vehicle.speed_measurement_valid if self.full_system else True)
    car_state.carState.vEgoWheelCount = int(self.vehicle.wheel_count if self.full_system else 4)
    car_state.carState.vEgoDropoutTime = float(self.vehicle.speed_dropout_time if self.full_system else 0.0)
    car_state.carState.vehicleSensorsInvalid = bool(self.vehicle.vehicle_sensors_invalid if self.full_system else False)
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
    self.planner_acceleration = float(self.planner.output_a_target)
    self.predictive_brake = float(getattr(self.planner, "crv_lead_follow_predictive_brake", 0.0))
    self.safety_override = bool(getattr(self.planner, "crv_lead_follow_safety_override", False))
    if self.full_system:
      self.speed, self.acceleration = self.vehicle.step(
        self.planner_acceleration, self.planner.output_should_stop, self.enabled, pitch, self.ts, v_cruise,
        getattr(self.planner, 'crv_stop_phase', 0), getattr(self.planner, 'crv_stop_reference', (0., 0., 0.)),
        getattr(self.planner, 'crv_lead_follow_urgency', 0.))
      command_accel, gas, self.brake_request = self.vehicle.command
      self.gas_command = gas / 1600.0
      self.brake_intensity = max(0.0, -command_accel / abs(ACCEL_MIN)) if self.brake_request else 0.0
      self.actuator_mode = 'brake' if self.brake_request else 'gas' if gas > 0 else 'coast'
      self.mode_transitions += int(self.actuator_mode != self._last_actuator_mode)
      self._last_actuator_mode = self.actuator_mode
    else:
      self._update_actuator(self.planner_acceleration)
      self.acceleration = self.planner_acceleration
    # Preserve the original stop actuator for legacy, physics-free maneuvers.
    # The optional physics model instead uses the static stop hold below.
    if not self.full_system and not self.physics and self.planner.output_should_stop:
      self.acceleration = min(-0.5, self.acceleration)
    if self.physics and not self.full_system:
      # The planner command is the longitudinal actuator input.  Physics adds
      # deterministic grade gravity and rolling resistance so a level/grade
      # cruise must actually balance the road load.
      # orientationNED pitch uses positive values for an uphill road in the
      # planner's coast model; gravity therefore opposes positive pitch.
      grade_accel = -9.81 * np.sin(float(pitch))
      rolling = self.rolling_resistance if self.speed > 0.01 else 0.0
      self.acceleration += grade_accel - np.sign(self.speed if self.speed > 0.01 else self.acceleration) * rolling
    if not self.full_system:
      self.speed = self.speed + self.acceleration * self.ts
    self.should_stop = self.planner.output_should_stop
    fcw = self.planner.fcw
    self.distance_lead = self.distance_lead + v_lead * self.ts

    # ******** run the car ********
    #print(self.distance, speed)
    # A stopped vehicle with an active stop command is held by static braking.
    # Without this, the point-mass plant can creep indefinitely on tiny
    # positive numerical acceleration despite the planner's stop state.
    if self.speed <= 0 or (not self.full_system and self.physics and self.should_stop and self.speed < 0.05):
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
    if self.realtime:
      self.rk.monitor_time()
    else:
      # Ratekeeper normally advances this before returning from monitor_time;
      # deterministic simulations must advance it without wall-clock pacing.
      self.rk._frame += 1

    return {
      "distance": self.distance,
      "speed": self.speed,
      "acceleration": self.acceleration,
      "should_stop": self.should_stop,
      "distance_lead": self.distance_lead,
      "fcw": fcw,
      "planner_acceleration": self.planner_acceleration,
      "gas_command": self.gas_command,
      "brake_request": self.brake_request,
      "brake_intensity": self.brake_intensity,
      "actuator_mode": self.actuator_mode,
      "mode_transitions": self.mode_transitions,
      "predictive_brake": self.predictive_brake,
      "safety_override": self.safety_override,
      "controller_acceleration": self.vehicle.output if self.full_system else self.planner_acceleration,
      "full_system": self.full_system,
    }

  def _update_actuator(self, accel):
    """Expose the CR-V Bosch actuator request using production mappings."""
    if self.CP.carFingerprint != CAR.HONDA_CRV_5G:
      self.gas_command = float(np.clip(accel / ACCEL_MAX, 0.0, 1.0))
      self.brake_request = bool(accel < 0.0)
    elif self.CP.flags & HondaFlags.BOSCH:
      gas_lookup = float(np.interp(accel, CarControllerParams.BOSCH_GAS_LOOKUP_BP,
                                   CarControllerParams.BOSCH_GAS_LOOKUP_V))
      self.brake_request = crv_brake_handoff(accel, getattr(self, "_crv_brake_active", False), True)
      self._crv_brake_active = self.brake_request
      self._crv_gas_command, self._crv_gas_active = crv_gas_handoff(
        accel, gas_lookup, getattr(self, "_crv_gas_command", 0.0),
        getattr(self, "_crv_gas_active", False), True, self.ts, self.brake_request)
      gas_full_scale = float(CarControllerParams.BOSCH_GAS_LOOKUP_V[-1])
      self.gas_command = float(np.clip(self._crv_gas_command / gas_full_scale, 0.0, 1.0))
    else:
      self.gas_command, brake = compute_gas_brake(accel, self.speed, self.CP)
      self.brake_request = bool(brake > 0.0)
    self.brake_intensity = float(np.clip(-accel / abs(ACCEL_MIN), 0.0, 1.0))
    mode = "brake" if self.brake_request else "gas" if self.gas_command > 1e-4 else "coast"
    if mode != self._last_actuator_mode:
      self.mode_transitions += 1
      self._last_actuator_mode = mode
    self.actuator_mode = mode

# simple engage in standalone mode
def plant_thread():
  plant = Plant()
  while 1:
    plant.step()


if __name__ == "__main__":
  plant_thread()
