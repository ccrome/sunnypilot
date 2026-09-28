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
# A one-second speed-error response overshoots the CR-V after actuator delay;
# this slower reference is still fast enough for normal set-speed changes.
CRV_CRUISE_SPEED_GAIN = 0.10
CRV_GRADE_ACCEL_GAIN = 9.81 / 5.65
# Fitted toward the several-second correlation of lead speed/gap in the CR-V
# reference drive. Safety/TTC and stop paths below remain immediate.
# The reference drives accept several seconds of lead-speed variation instead
# of chasing every change; use a slower speed reference than gap correction.
CRV_LEAD_FOLLOW_SPEED_TAU_PER_FOLLOW = 2.5
CRV_LEAD_FOLLOW_SPEED_RESPONSE = 0.55
CRV_LEAD_FOLLOW_GAP_TAU_PER_FOLLOW = 1.5
CRV_LEAD_FOLLOW_ACCEL_TAU_PER_FOLLOW = 0.75
CRV_LEAD_FOLLOW_OUTPUT_TAU = 2.00
CRV_LEAD_FOLLOW_I_GAIN = 0.02
CRV_LEAD_FOLLOW_I_LIMIT = 0.15
# Mild MPC deceleration is part of ordinary lead-speed regulation and is
# handled by the filtered reference. Only a clearly meaningful raw brake
# request (or an explicit stop decision) bypasses that reference.
CRV_LEAD_FOLLOW_BRAKE_OVERRIDE = -0.20
CRV_LEAD_FOLLOW_GAP_HYSTERESIS = 0.5
CRV_POST_RESTART_GAP_RELEASE_MARGIN = 1.0
CRV_LEAD_FOLLOW_BRAKE_HOLD_TIME = 0.5
# Engage predictive braking well before the closing lead reaches the old 4 s TTC
# boundary; the plant/actuator delay otherwise allows an unsafe time-gap
# collapse even though the override technically fires.
CRV_LEAD_FOLLOW_SAFETY_TTC = 10.0
CRV_LEAD_FOLLOW_MIN_CLOSING_TIME_GAP = 1.25
# Reference-drive p10 time gap is 1.38 s against a 2.05 s target.
CRV_LEAD_FOLLOW_CLOSE_TOLERANCE = 0.65
CRV_LEAD_FOLLOW_FAR_TOLERANCE = 0.50
CRV_LEAD_FOLLOW_CLOSE_RECOVERY_TAU = 10.0
CRV_LEAD_FOLLOW_FAR_RECOVERY_TAU = 5.0
CRV_LEAD_FOLLOW_TIME_ERROR_GAIN = 0.10
CRV_LEAD_FOLLOW_BIAS_LIMIT = 2.0
CRV_LEAD_FOLLOW_COAST_DOWN_ACCEL = -0.15

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


def sigmoid(value):
  """Numerically stable continuous confidence/pressure mapping."""
  value = float(np.clip(value, -60.0, 60.0))
  return 1.0 / (1.0 + math.exp(-value))


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

  grade_accel = -(accel_coast + 0.3) * CRV_GRADE_ACCEL_GAIN
  target_accel = np.clip(CRV_CRUISE_SPEED_GAIN * (v_cruise - v_ego) + speed_error_bias + grade_accel,
                         A_CRUISE_MIN, max_accel)
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
    self.crv_cruise_speed_error_prev = 0.0
    self.crv_lead_follow_speed = init_v
    self.crv_lead_follow_gap = 0.0
    self.crv_lead_follow_i = 0.0
    self.crv_lead_follow_accel = init_a
    self.crv_lead_follow_time_gap = 0.0
    self.crv_lead_follow_time_error = 0.0
    self.crv_lead_follow_bias = 0.0
    self.crv_lead_follow_predictive_brake = 0.0
    self.crv_lead_follow_safety_override = False
    self.crv_lead_follow_brake_hold_remaining = 0.0
    self.crv_lead_gap_protection = False
    self.lead_loss_hold_remaining = 0.0
    self.lead_loss_hold_accel = 0.0
    self.stopped_lead_loss_hold_remaining = 0.0
    self.crv_post_restart_gap_hold = False
    self.crv_post_restart_gap_armed = False

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
      self.crv_cruise_speed_error_prev = 0.0
      self.crv_lead_follow_speed = v_ego
      self.crv_lead_follow_gap = 0.0
      self.crv_lead_follow_i = 0.0
      self.crv_lead_follow_accel = 0.0
      self.crv_lead_follow_time_gap = 0.0
      self.crv_lead_follow_time_error = 0.0
      self.crv_lead_follow_bias = 0.0
      self.crv_lead_follow_predictive_brake = 0.0
      self.crv_lead_follow_safety_override = False
      self.crv_lead_follow_brake_hold_remaining = 0.0
      self.crv_lead_gap_protection = False
      self.lead_loss_hold_remaining = 0.0
      self.stopped_lead_loss_hold_remaining = 0.0
      self.crv_post_restart_gap_hold = False
      self.crv_post_restart_gap_armed = False

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
      if speed_error * self.crv_cruise_speed_error_prev < 0.0:
        # Do not carry an acceleration bias through the target-speed crossing;
        # that stored bias is the source of the repeatable transition overshoot.
        self.crv_cruise_speed_i = 0.0
      self.crv_cruise_speed_i = float(np.clip(
        self.crv_cruise_speed_i + CRV_CRUISE_SPEED_I_GAIN * self.dt * speed_error * self.throttle_authority,
        -CRV_CRUISE_SPEED_I_LIMIT, CRV_CRUISE_SPEED_I_LIMIT))
      self.crv_cruise_speed_error_prev = speed_error
    else:
      self.crv_cruise_speed_i = 0.0
      self.crv_cruise_speed_error_prev = 0.0

    self.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.throttle_authority, self.crv_cruise_speed_i)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    tracked_leads = [lead for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo) if lead.present]
    lead_follow_active = self.is_crv_5g and bool(tracked_leads) and not is_e2e
    if lead_follow_active:
      # Use the lead as a slowly varying speed reference for normal following.
      # The raw MPC remains available below as an immediate safety override.
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      t_follow = get_T_FOLLOW(sm['selfdriveState'].personality)
      speed_tau = CRV_LEAD_FOLLOW_SPEED_TAU_PER_FOLLOW * t_follow
      gap_tau = CRV_LEAD_FOLLOW_GAP_TAU_PER_FOLLOW * t_follow
      accel_tau = CRV_LEAD_FOLLOW_ACCEL_TAU_PER_FOLLOW * t_follow
      speed_alpha = self.dt / (speed_tau + self.dt)
      gap_alpha = self.dt / (gap_tau + self.dt)
      self.crv_lead_follow_speed += speed_alpha * (closest_lead.vLead - self.crv_lead_follow_speed)
      self.crv_lead_follow_gap += gap_alpha * (closest_lead.dRel - self.crv_lead_follow_gap)
      self.crv_lead_follow_time_gap = self.crv_lead_follow_gap / max(v_ego, 1.0)
      close_boundary = t_follow - CRV_LEAD_FOLLOW_CLOSE_TOLERANCE
      far_boundary = t_follow + CRV_LEAD_FOLLOW_FAR_TOLERANCE
      if self.crv_lead_follow_time_gap < close_boundary:
        time_error = self.crv_lead_follow_time_gap - close_boundary
        recovery_tau = CRV_LEAD_FOLLOW_CLOSE_RECOVERY_TAU
      elif self.crv_lead_follow_time_gap > far_boundary:
        time_error = self.crv_lead_follow_time_gap - far_boundary
        recovery_tau = CRV_LEAD_FOLLOW_FAR_RECOVERY_TAU
      else:
        time_error = 0.0
        recovery_tau = CRV_LEAD_FOLLOW_CLOSE_RECOVERY_TAU if self.crv_lead_follow_bias < 0.0 \
          else CRV_LEAD_FOLLOW_FAR_RECOVERY_TAU
      self.crv_lead_follow_time_error = time_error
      bias_target = np.clip(
        CRV_LEAD_FOLLOW_TIME_ERROR_GAIN * v_ego * time_error,
        -CRV_LEAD_FOLLOW_BIAS_LIMIT, CRV_LEAD_FOLLOW_BIAS_LIMIT)
      bias_alpha = self.dt / (recovery_tau + self.dt)
      self.crv_lead_follow_bias += bias_alpha * (bias_target - self.crv_lead_follow_bias)
      lead_speed_target = self.crv_lead_follow_speed + self.crv_lead_follow_bias
      follow_speed = min(
        v_cruise,
        v_ego + CRV_LEAD_FOLLOW_SPEED_RESPONSE * (lead_speed_target - v_ego))
      speed_error = follow_speed - v_ego
      self.crv_lead_follow_i = float(np.clip(
        self.crv_lead_follow_i + CRV_LEAD_FOLLOW_I_GAIN * self.dt * speed_error,
        -CRV_LEAD_FOLLOW_I_LIMIT, CRV_LEAD_FOLLOW_I_LIMIT))
      nominal_lead_follow_accel = get_cruise_accel(
        is_e2e, follow_speed + self.crv_lead_follow_i, v_ego,
        self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
        accel_coast, self.throttle_authority)
      accel_alpha = self.dt / (accel_tau + self.dt)
      self.crv_lead_follow_accel += accel_alpha * (
        nominal_lead_follow_accel - self.crv_lead_follow_accel)
      lead_follow_accel = self.crv_lead_follow_accel

      # Predict the deceleration needed to remove relative closing speed before
      # reaching the driver's target following distance. This is continuous:
      # zero/positive relative speed contributes almost nothing, while a fast
      # closing lead naturally produces an earlier, stronger brake request.
      target_follow_distance = max(2.0, v_ego * t_follow)
      closing_speed = max(-closest_lead.vRel, 0.0)
      distance_to_target = closest_lead.dRel - target_follow_distance
      predicted_brake = -(closing_speed ** 2) / (2.0 * max(distance_to_target, 1.0))
      closing_confidence = sigmoid((closing_speed - 0.5) / 0.25)
      predictive_limit = ((1.0 - closing_confidence) * ACCEL_MAX
                          + closing_confidence * np.clip(predicted_brake, ACCEL_MIN, 0.0))
      self.crv_lead_follow_predictive_brake = float(np.clip(
        closing_confidence * predicted_brake, ACCEL_MIN, 0.0))
      lead_follow_accel = min(lead_follow_accel, predictive_limit)

      # Only a meaningful raw braking request or an MPC stop decision can
      # override the averaged following reference. Close/closing and stopped
      # lead guards below remain independent hard safety overrides.
      # A raw negative MPC request is only an immediate override for an
      # actual closing-TTC threat. A lead traveling near ego speed should be
      # handled by the averaged reference and coast-only gap guard.
      closing_ttc = closest_lead.dRel / max(-closest_lead.vRel, 1e-3) \
        if closest_lead.vRel < 0.0 else float("inf")
      raw_mpc_brake_request = output_should_stop_mpc or (
        output_a_target_mpc < CRV_LEAD_FOLLOW_BRAKE_OVERRIDE
        and closing_ttc < CRV_LEAD_FOLLOW_SAFETY_TTC)
      if raw_mpc_brake_request:
        self.crv_lead_follow_brake_hold_remaining = CRV_LEAD_FOLLOW_BRAKE_HOLD_TIME
      elif self.crv_lead_follow_brake_hold_remaining > 0.0:
        self.crv_lead_follow_brake_hold_remaining = max(
          0.0, self.crv_lead_follow_brake_hold_remaining - self.dt)
      raw_mpc_brake_override = self.crv_lead_follow_brake_hold_remaining > 0.0
      self.crv_lead_follow_safety_override = raw_mpc_brake_override
      lead_candidate = min(output_a_target_mpc, 0.0) if raw_mpc_brake_override else lead_follow_accel
      lead_stop = output_should_stop_mpc if raw_mpc_brake_request else False
    else:
      self.crv_lead_follow_i = 0.0
      self.crv_lead_follow_accel = self.a_cruise
      self.crv_lead_follow_time_gap = 0.0
      self.crv_lead_follow_time_error = 0.0
      self.crv_lead_follow_bias = 0.0
      self.crv_lead_follow_predictive_brake = 0.0
      self.crv_lead_follow_brake_hold_remaining = 0.0
      self.crv_lead_follow_safety_override = False
      self.crv_lead_gap_protection = False
      lead_candidate = output_a_target_mpc
      lead_stop = output_should_stop_mpc

    candidates = [(lead_candidate, self.mpc.source, lead_stop),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if is_e2e:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target = smooth_min([candidate[0] for candidate in candidates])
    plan_source = min(candidates, key=lambda c: c[0])[1]

    # Candidate arbitration can otherwise expose small solver/source changes
    # directly at the actuator boundary. Filter the normal target continuously;
    # explicit stop/safety confidence blends back to the raw target so safety
    # commands are not delayed by comfort smoothing.
    output_alpha = self.dt / (CRV_LEAD_FOLLOW_OUTPUT_TAU + self.dt)
    filtered_output = a_prev + output_alpha * (output_a_target - a_prev)
    safety_bypass = float(output_should_stop_mpc or self.crv_lead_follow_safety_override)
    output_a_target = ((1.0 - safety_bypass) * filtered_output
                       + safety_bypass * output_a_target)

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

    if self.is_crv_5g and v_ego < STOPPED_LEAD_SPEED and stopped_close_lead:
      # Remember that the ego vehicle stopped with an insufficient following
      # gap. A moving lead must open that gap before restart acceleration.
      self.crv_post_restart_gap_hold = True
      self.crv_post_restart_gap_armed = True
    elif self.crv_post_restart_gap_armed and not tracked_leads:
      # Lead loss is handled by the existing L1/L2 protections.
      self.crv_post_restart_gap_hold = False
      self.crv_post_restart_gap_armed = False

    if self.crv_post_restart_gap_armed and tracked_leads:
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      release_distance = get_safe_obstacle_distance(v_ego, get_T_FOLLOW(sm['selfdriveState'].personality)) \
        - get_stopped_equivalence_factor(max(closest_lead.vLead, 0.0))
      if closest_lead.dRel >= release_distance + CRV_POST_RESTART_GAP_RELEASE_MARGIN:
        self.crv_post_restart_gap_hold = False
      else:
        self.crv_post_restart_gap_hold = True

    if self.crv_post_restart_gap_hold and tracked_leads:
      output_a_target = min(output_a_target, 0.0)

    if self.is_crv_5g and tracked_leads:
      # Normal style control uses continuous confidence terms from the driver's
      # observed time-gap envelope. The old stopping-distance expression
      # includes emergency braking distance; using it as an everyday boundary
      # made stable leads look perpetually too close and produced coast-heavy
      # behavior. Emergency stopping, TTC, close-closing, and restart
      # protection remain separate hard safety paths.
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      t_follow = get_T_FOLLOW(sm['selfdriveState'].personality)
      # The target is the full personality following distance. The close-side
      # tolerance belongs in the response softness, not in the target itself.
      desired_lead_distance = max(2.0, v_ego * t_follow)
      gap_pressure = sigmoid((desired_lead_distance - closest_lead.dRel) /
                             max(CRV_LEAD_FOLLOW_GAP_HYSTERESIS, 0.1))
      closing_pressure = sigmoid((-closest_lead.vRel - 0.2) / 0.4)
      normal_follow_pressure = max(gap_pressure, 0.75 * closing_pressure)
      self.crv_lead_gap_protection = normal_follow_pressure > 0.5
      # Blend toward a gentle coast-down continuously. A negative candidate
      # remains untouched; only positive/near-zero normal following is shaped.
      normal_coast_limit = ((1.0 - normal_follow_pressure) * ACCEL_MAX
                            + normal_follow_pressure * CRV_LEAD_FOLLOW_COAST_DOWN_ACCEL)
      output_a_target = min(output_a_target, normal_coast_limit)

    close_closing_stop = self.is_crv_5g and crv_close_closing_should_stop(
      v_ego, (sm['radarState'].leadOne, sm['radarState'].leadTwo))
    if close_closing_stop:
      output_a_target = min(output_a_target, 0.0)

    # A closing lead must not be allowed to consume the entire following gap
    # while the TTC path is reacting. This is only a closing-lead guard: a
    # stable lead may remain inside the normal driver's time-gap envelope.
    if self.is_crv_5g and tracked_leads:
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      closing_time_gap = closest_lead.dRel / max(v_ego, 0.5)
      if closest_lead.vRel < -0.1 and closing_time_gap < CRV_LEAD_FOLLOW_MIN_CLOSING_TIME_GAP:
        output_a_target = min(output_a_target, -0.5)

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
