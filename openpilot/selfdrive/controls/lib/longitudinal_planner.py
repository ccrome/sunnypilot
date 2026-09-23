#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.honda.values import CAR
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import MPC_SOURCES, LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_safe_obstacle_distance, get_stopped_equivalence_factor, get_T_FOLLOW
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MIN = -1.2
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
ALLOW_THROTTLE_BLEND = 0.2
MIN_ALLOW_THROTTLE_SPEED = 2.5
ALLOW_THROTTLE_SPEED_BLEND = 2.5
LEAD_LOSS_HOLD_TIME = 0.5
LEAD_LOSS_RELEASE_JERK = 1.0
STOPPED_LEAD_DISTANCE = 3.0
STOPPED_LEAD_SPEED = 0.5
STOPPED_LEAD_LOSS_HOLD_TIME = 0.5
CRV_CLOSE_STOP_DISTANCE = 2.5
CRV_CLOSE_STOP_CLOSING_SPEED = 0.2
CRV_CLOSE_STOP_MIN_SPEED = 0.3
CRV_CRUISE_SPEED_I_MIN_SPEED = 5.0
CRV_CRUISE_SPEED_I_GAIN = 0.03
CRV_CRUISE_SPEED_I_LIMIT = 0.15

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def get_throttle_authority(throttle_prob, v_ego):
  """Return continuous authority for positive cruise acceleration."""
  probability_authority = np.clip(
    (throttle_prob - (ALLOW_THROTTLE_THRESHOLD - ALLOW_THROTTLE_BLEND / 2.0)) / ALLOW_THROTTLE_BLEND,
    0.0, 1.0,
  )
  low_speed_authority = 1.0 - np.clip(
    (v_ego - MIN_ALLOW_THROTTLE_SPEED) / ALLOW_THROTTLE_SPEED_BLEND,
    0.0, 1.0,
  )
  return float(max(probability_authority, low_speed_authority))


def smooth_min(values, softness=0.1):
  """Conservative smooth minimum for continuous candidate arbitration."""
  result = float(values[0])
  for value in values[1:]:
    value = float(value)
    result = 0.5 * (result + value - math.sqrt((result - value) ** 2 + softness ** 2))
  return result


def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, throttle_authority,
                     speed_error_bias=0.0):
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    clipped_accel_coast = max(accel_coast, ACCEL_MIN)
    coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2],
                            [max_accel, clipped_accel_coast])
    max_accel = coast_limit + throttle_authority * (max_accel - coast_limit)

  target_accel = np.clip(v_cruise - v_ego + speed_error_bias, A_CRUISE_MIN, max_accel)
  j_cruise = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


def crv_close_closing_should_stop(v_ego, leads):
  """Require the stopping state for a short, shrinking tracked-lead gap."""
  return v_ego > CRV_CLOSE_STOP_MIN_SPEED and any(
    lead.present
    and lead.dRel < CRV_CLOSE_STOP_DISTANCE
    and lead.vRel < -CRV_CLOSE_STOP_CLOSING_SPEED
    for lead in leads
  )


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.throttle_authority = 1.0
    self.is_crv_5g = self.CP.carFingerprint == CAR.HONDA_CRV_5G

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False
    self.crv_cruise_speed_i = 0.0
    self.lead_loss_hold_remaining = 0.0
    self.lead_loss_hold_accel = 0.0
    self.stopped_lead_loss_hold_remaining = 0.0
    self.crv_post_restart_gap_hold = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    self.throttle_authority = get_throttle_authority(throttle_prob, v_ego)
    # Keep the existing message field as a compatibility/display indicator. The
    # planner uses throttle_authority continuously for the actual acceleration.
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target
      self.crv_cruise_speed_i = 0.0
      self.lead_loss_hold_remaining = 0.0
      self.stopped_lead_loss_hold_remaining = 0.0
      self.crv_post_restart_gap_hold = False

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    is_e2e = self.is_e2e(sm)

    lead_present = any(lead.present for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo))
    use_crv_cruise_speed_i = self.is_crv_5g and not is_e2e and not lead_present \
      and self.throttle_authority > 0.0 and v_ego >= CRV_CRUISE_SPEED_I_MIN_SPEED
    if use_crv_cruise_speed_i:
      speed_error = v_cruise - v_ego
      self.crv_cruise_speed_i = float(np.clip(
        self.crv_cruise_speed_i + CRV_CRUISE_SPEED_I_GAIN * self.dt * speed_error * self.throttle_authority,
        -CRV_CRUISE_SPEED_I_LIMIT, CRV_CRUISE_SPEED_I_LIMIT))
    else:
      self.crv_cruise_speed_i = 0.0

    self.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.throttle_authority, self.crv_cruise_speed_i)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if is_e2e:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target = smooth_min([candidate[0] for candidate in candidates])
    plan_source = min(candidates, key=lambda c: c[0])[1]

    if self.is_crv_5g and lead_present and plan_source in MPC_SOURCES:
      # A lead selected by the MPC can disappear for a few frames while tracker
      # candidates switch. Preserve its braking request instead of authorizing
      # cruise acceleration before the path is confirmed clear.
      self.lead_loss_hold_remaining = LEAD_LOSS_HOLD_TIME
      self.lead_loss_hold_accel = min(output_a_target, 0.0)
    elif not lead_present and self.lead_loss_hold_remaining > 0.0:
      elapsed = LEAD_LOSS_HOLD_TIME - self.lead_loss_hold_remaining
      hold_accel = min(0.0, self.lead_loss_hold_accel + LEAD_LOSS_RELEASE_JERK * elapsed)
      output_a_target = min(output_a_target, hold_accel)
      self.lead_loss_hold_remaining = max(0.0, self.lead_loss_hold_remaining - self.dt)
    else:
      self.lead_loss_hold_remaining = 0.0

    stopped_close_lead = any(lead.present and lead.dRel < STOPPED_LEAD_DISTANCE and lead.vLead < STOPPED_LEAD_SPEED
                             for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo))
    stopped_lead_loss_hold_active = False
    if self.is_crv_5g and v_ego < STOPPED_LEAD_SPEED and stopped_close_lead:
      # A stopped lead can disappear briefly during tracker switching. Keep the
      # stop command until the short loss window expires instead of restarting.
      self.stopped_lead_loss_hold_remaining = STOPPED_LEAD_LOSS_HOLD_TIME
    elif self.is_crv_5g and not any(lead.present for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo)) \
        and self.stopped_lead_loss_hold_remaining > 0.0:
      output_a_target = min(output_a_target, 0.0)
      stopped_lead_loss_hold_active = True
      self.stopped_lead_loss_hold_remaining = max(0.0, self.stopped_lead_loss_hold_remaining - self.dt)
    else:
      self.stopped_lead_loss_hold_remaining = 0.0

    tracked_leads = [lead for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo) if lead.present]
    if self.is_crv_5g and v_ego < STOPPED_LEAD_SPEED and stopped_close_lead:
      # Remember that the ego vehicle stopped with an insufficient following
      # gap. A moving lead must open that gap before restart acceleration.
      self.crv_post_restart_gap_hold = True
    elif self.crv_post_restart_gap_hold and not tracked_leads:
      # Lead loss is handled by the existing L1/L2 protections.
      self.crv_post_restart_gap_hold = False

    if self.crv_post_restart_gap_hold and tracked_leads:
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      release_distance = get_safe_obstacle_distance(v_ego, get_T_FOLLOW(sm['selfdriveState'].personality)) \
        - get_stopped_equivalence_factor(max(closest_lead.vLead, 0.0))
      if closest_lead.dRel >= release_distance:
        self.crv_post_restart_gap_hold = False
      else:
        output_a_target = min(output_a_target, 0.0)

    if self.is_crv_5g and tracked_leads:
      # Do not accelerate toward a continuously tracked lead while the gap is
      # below the existing desired following-distance boundary. The planner
      # may coast or brake until the lead opens the gap again.
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      desired_lead_distance = get_safe_obstacle_distance(v_ego, get_T_FOLLOW(sm['selfdriveState'].personality)) \
        - get_stopped_equivalence_factor(max(closest_lead.vLead, 0.0))
      if closest_lead.dRel < desired_lead_distance:
        output_a_target = min(output_a_target, 0.0)

    close_closing_stop = self.is_crv_5g and crv_close_closing_should_stop(
      v_ego, (sm['radarState'].leadOne, sm['radarState'].leadTwo))
    if close_closing_stop:
      output_a_target = min(output_a_target, 0.0)

    self.mpc.source = plan_source
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates) \
      or stopped_lead_loss_hold_active or close_closing_stop
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
