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
# Keep ordinary speed changes in a comfortable S-curve envelope.  The
# acceleration target still forms the plateau; the planner's jerk state shapes
# the ramp in and ramp out instead of asking the plant to absorb a step.
J_CRUISE_VALS = [0.8, 0.7, 0.6, 0.5]
CRV_ACCEL_RESPONSE_WN = 0.9
CRV_ACCEL_EMERGENCY_RESPONSE_WN = 1.2
CRV_ACCEL_MAX_SNAP = 4.0
CRV_ACCEL_EMERGENCY_MAX_SNAP = 20.0
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
CRV_CRUISE_SPEED_I_GAIN = 0.025
CRV_CRUISE_SPEED_I_LIMIT = 0.03
# A one-second speed-error response overshoots the CR-V after actuator delay;
# this slower reference is still fast enough for normal set-speed changes.
CRV_CRUISE_SPEED_GAIN = 0.12
# Dampen only transient acceleration.  Keeping grade compensation outside this
# term lets the steady-state command remain exactly the road-load balance while
# the transient reference returns toward zero acceleration before the speed
# target is crossed.
CRV_CRUISE_ACCEL_DAMPING = 0.15
CRV_GRADE_ACCEL_GAIN = 9.81 / 5.65
# Mild MPC deceleration is part of ordinary lead-speed regulation. Only a
# clearly meaningful raw brake request (or an explicit stop decision) bypasses
# the direct two-goal feedback law.
CRV_LEAD_FOLLOW_BRAKE_OVERRIDE = -0.20
CRV_LEAD_FOLLOW_COAST_DEADBAND = 0.0
CRV_LEAD_FOLLOW_GAP_HYSTERESIS = 0.5
CRV_POST_RESTART_GAP_RELEASE_MARGIN = 1.0
CRV_LEAD_FOLLOW_BRAKE_HOLD_TIME = 0.5
# Engage predictive braking well before the closing lead reaches the old 4 s TTC
# boundary; the plant/actuator delay otherwise allows an unsafe time-gap
# collapse even though the override technically fires.
# The safety fallback is jerk-limited below, so it can begin before the
# following gap is consumed without creating the former raw-MPC brake step.
CRV_LEAD_FOLLOW_SAFETY_TTC = 4.8
# Direct feedback on the two coupled goals: reach the lead velocity while
# arriving at its desired following gap.  The intercept term below supplies
# the trajectory, and the plant/actuator dynamics provide the response shape.
CRV_LEAD_FOLLOW_COUPLED_GAP_GAIN = 0.12
# The gap/relative-velocity pair is a second-order follower. Set its damping
# term to the critical value instead of leaving the response underdamped.
CRV_LEAD_FOLLOW_COUPLED_VELOCITY_GAIN = 2.0 * math.sqrt(CRV_LEAD_FOLLOW_COUPLED_GAP_GAIN)
# Lead acceleration is a noisy derivative of the tracker speed.  Normal
# following should satisfy the gap and velocity goals directly; the raw lead
# deceleration is retained as an urgency signal for the closing/safety path
# instead of being injected into every ordinary command.
CRV_LEAD_FOLLOW_LEAD_ACCEL_GAIN = 0.0
CRV_LEAD_FOLLOW_LEAD_DECEL_PRESSURE_START = 0.8
CRV_LEAD_FOLLOW_LEAD_DECEL_PRESSURE_RANGE = 1.2
# Normal following uses a comfort jerk envelope.  The rate rises continuously
# as a closing lead consumes the available TTC margin; the independent safety
# path still owns emergency braking beyond this envelope.
# Ordinary lead following should use the same occupant-comfort envelope as a
# normal speed transition.  Closing pressure raises this continuously toward
# the emergency envelope; a stable lead never gets the old 1.5 m/s^3 step.
CRV_LEAD_FOLLOW_COMFORT_BRAKE_JERK = 0.8
CRV_LEAD_FOLLOW_EMERGENCY_BRAKE_JERK = 5.0
CRV_LEAD_FOLLOW_COMFORT_TTC = 5.0
CRV_LEAD_FOLLOW_ACCEL_JERK = 1.5
CRV_LEAD_FOLLOW_MIN_CLOSING_TIME_GAP = 1.25
# Reference-drive p10 time gap is 1.38 s against a 2.05 s target.
CRV_LEAD_FOLLOW_CLOSE_TOLERANCE = 0.65
CRV_LEAD_FOLLOW_FAR_TOLERANCE = 0.50
CRV_LEAD_FOLLOW_CLOSE_RECOVERY_TAU = 10.0
CRV_LEAD_FOLLOW_FAR_RECOVERY_TAU = 5.0
CRV_LEAD_FOLLOW_TIME_ERROR_GAIN = 0.10
CRV_LEAD_FOLLOW_BIAS_LIMIT = 2.0
CRV_LEAD_FOLLOW_COAST_DOWN_ACCEL = -0.15
CRV_FOLLOW_MAX_MANEUVER_TIME = 12.0
CRV_FOLLOW_MIN_CLOSING_SPEED = 0.5
CRV_FOLLOW_GAP_GAIN = 0.04
CRV_FOLLOW_SPEED_GAIN = 0.45
CRV_FOLLOW_SMOOTHSTEP_PEAK_JERK = 10.0 / math.sqrt(3.0)

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


def smooth_deadzone(value, width):
  """Reduce tiny signed corrections continuously into coasting."""
  return float(np.sign(value) * (np.hypot(value, width) - width))


def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, throttle_authority,
                     speed_error_bias=0.0, measured_accel=0.0):
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
  if CP.carFingerprint != CAR.HONDA_CRV_5G:
    target_accel = np.clip(CRV_CRUISE_SPEED_GAIN * (v_cruise - v_ego) + speed_error_bias + grade_accel,
                           A_CRUISE_MIN, max_accel)
    j_cruise = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
    return float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))
  # The reference is net acceleration. Road-load effort belongs to the
  # acceleration tracking loop, not to this speed/gap reference.
  target_accel = np.clip(CRV_CRUISE_SPEED_GAIN * (v_cruise - v_ego)
                         - 0.6 * measured_accel + speed_error_bias,
                         A_CRUISE_MIN, max_accel)
  return float(target_accel)


def update_accel_scurve(target_accel, current_accel, jerk_prev, dt, max_jerk,
                        response_wn=CRV_ACCEL_RESPONSE_WN,
                        max_snap=CRV_ACCEL_MAX_SNAP):
  """Track an acceleration target with continuous jerk and bounded snap."""
  accel_error = target_accel - current_accel
  snap = np.clip(
    response_wn ** 2 * accel_error - 2.0 * response_wn * jerk_prev,
    -max_snap, max_snap)
  jerk = np.clip(jerk_prev + snap * dt, -max_jerk, max_jerk)
  accel = np.clip(current_accel + jerk * dt, ACCEL_MIN, ACCEL_MAX)
  return float(accel), float(jerk)


def crv_follow_maneuver(elapsed, duration, closing_speed, target_gap):
  """One-sided, jerk-continuous speed match that spends the excess gap.

  The fifth-order relative-speed smoothstep integrates to exactly half the
  initial closing speed over the maneuver. With duration = 2 * excess_gap /
  closing_speed, position and speed therefore reach their targets together.
  Its acceleration and jerk are zero at both boundaries.
  """
  s = float(np.clip(elapsed / duration, 0.0, 1.0))
  relative_speed_fraction = 1.0 - 10.0 * s ** 3 + 15.0 * s ** 4 - 6.0 * s ** 5
  distance_fraction = 0.5 - s + 2.5 * s ** 4 - 3.0 * s ** 5 + s ** 6
  decel = -closing_speed / duration * 30.0 * s ** 2 * (1.0 - s) ** 2
  return (closing_speed * relative_speed_fraction,
          target_gap + closing_speed * duration * distance_fraction,
          decel)


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
    self.crv_accel_jerk = 0.0
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
    self.crv_lead_follow_jerk_limit = CRV_LEAD_FOLLOW_COMFORT_BRAKE_JERK
    self.crv_lead_follow_urgency = 0.0
    self.crv_follow_elapsed = 0.0
    self.crv_follow_duration = 0.0
    self.crv_follow_closing_speed = 0.0
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
      self.crv_accel_jerk = 0.0
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
      self.crv_lead_follow_jerk_limit = CRV_LEAD_FOLLOW_COMFORT_BRAKE_JERK
      self.crv_lead_follow_urgency = 0.0
      self.crv_follow_elapsed = 0.0
      self.crv_follow_duration = 0.0
      self.crv_follow_closing_speed = 0.0
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
    use_crv_cruise_speed_i = self.is_crv_5g and not is_e2e \
      and self.throttle_authority > 0.0 and v_ego >= CRV_CRUISE_SPEED_I_MIN_SPEED
    if use_crv_cruise_speed_i:
      speed_error = v_cruise - v_ego
      self.crv_cruise_speed_i = float(np.clip(
        self.crv_cruise_speed_i + CRV_CRUISE_SPEED_I_GAIN * self.dt * speed_error * self.throttle_authority,
        -CRV_CRUISE_SPEED_I_LIMIT, CRV_CRUISE_SPEED_I_LIMIT))
      self.crv_cruise_speed_error_prev = speed_error
    else:
      self.crv_cruise_speed_i = 0.0
      self.crv_cruise_speed_error_prev = 0.0

    self.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.throttle_authority, self.crv_cruise_speed_i, sm['carState'].aEgo)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    tracked_leads = [lead for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo) if lead.present]
    lead_follow_active = self.is_crv_5g and bool(tracked_leads) and not is_e2e
    if lead_follow_active:
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      t_follow = get_T_FOLLOW(sm['selfdriveState'].personality)
      self.crv_lead_follow_speed = closest_lead.vLead
      self.crv_lead_follow_gap = closest_lead.dRel
      self.crv_lead_follow_time_gap = self.crv_lead_follow_gap / max(v_ego, 1.0)
      self.crv_lead_follow_time_error = self.crv_lead_follow_time_gap - t_follow
      desired_lead_gap = max(2.0, closest_lead.vLead * t_follow)
      closing_speed = max(v_ego - closest_lead.vLead, 0.0)
      excess_gap = closest_lead.dRel - desired_lead_gap
      # One continuously recomputed gap/velocity controller. A timed maneuver
      # becomes stale when the lead or the actuator response changes.
      self.crv_follow_duration = 0.0
      lead_follow_accel = (CRV_FOLLOW_SPEED_GAIN * (closest_lead.vLead - v_ego)
                           + CRV_FOLLOW_GAP_GAIN * excess_gap
                           - 0.4 * sm['carState'].aEgo)
      braking_pressure = float(np.clip(-lead_follow_accel / abs(ACCEL_MIN), 0.0, 1.0))
      self.crv_lead_follow_jerk_limit = (CRV_LEAD_FOLLOW_COMFORT_BRAKE_JERK
        + braking_pressure * (CRV_LEAD_FOLLOW_EMERGENCY_BRAKE_JERK - CRV_LEAD_FOLLOW_COMFORT_BRAKE_JERK))
      predicted_min_gap = closest_lead.dRel - closing_speed * action_t \
        - closing_speed ** 2 / (2.0 * abs(ACCEL_MIN))
      minimum_terminal_gap = max(2.0, v_ego)
      lead_follow_accel += float(np.clip(closest_lead.aLeadK, -2.0, 2.0))
      lead_follow_accel = min(lead_follow_accel, self.a_cruise)

      # The time-gap floor is a reachability constraint: when a smooth match
      # cannot preserve one second under jerk-limited braking, request the
      # necessary deceleration from the same acceleration trajectory.
      safety_gap = max(2.0, v_ego)
      uncommitted_brake_fraction = 1.0 - np.clip(-a_prev / abs(ACCEL_MIN), 0.0, 1.0)
      safety_margin = (closest_lead.dRel - safety_gap
                       - closing_speed * action_t * uncommitted_brake_fraction)
      safety_accel = -closing_speed ** 2 / (2.0 * max(safety_margin, 0.1))
      safety_required = predicted_min_gap < minimum_terminal_gap
      self.crv_lead_follow_safety_override = bool(safety_required and closing_speed > 0.1)
      lead_follow_accel = min(lead_follow_accel, safety_accel) if self.crv_lead_follow_safety_override else lead_follow_accel
      lead_candidate = float(np.clip(lead_follow_accel, ACCEL_MIN, ACCEL_MAX))
      self.crv_lead_follow_accel = lead_candidate
      self.crv_lead_follow_predictive_brake = min(lead_candidate, 0.0)
      self.crv_lead_follow_urgency = float(self.crv_lead_follow_safety_override)
      lead_stop = output_should_stop_mpc and closest_lead.vLead < STOPPED_LEAD_SPEED
    else:
      self.crv_follow_duration = 0.0
      self.crv_lead_follow_i = 0.0
      self.crv_lead_follow_accel = self.a_cruise
      self.crv_lead_follow_time_gap = 0.0
      self.crv_lead_follow_time_error = 0.0
      self.crv_lead_follow_bias = 0.0
      self.crv_lead_follow_predictive_brake = 0.0
      self.crv_lead_follow_brake_hold_remaining = 0.0
      self.crv_lead_follow_safety_override = False
      self.crv_lead_follow_jerk_limit = CRV_LEAD_FOLLOW_COMFORT_BRAKE_JERK
      self.crv_lead_follow_urgency = 0.0
      self.crv_lead_gap_protection = False
      lead_candidate = output_a_target_mpc
      lead_stop = output_should_stop_mpc

    candidates = [(lead_candidate, self.mpc.source, lead_stop)]
    if not lead_follow_active:
      candidates.append((self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop))
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

    if self.is_crv_5g and tracked_leads and not lead_follow_active:
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
    if self.is_crv_5g and tracked_leads and not lead_follow_active:
      closest_lead = min(tracked_leads, key=lambda lead: lead.dRel)
      closing_time_gap = closest_lead.dRel / max(v_ego, 0.5)
      if closest_lead.vRel < -0.1 and closing_time_gap < CRV_LEAD_FOLLOW_MIN_CLOSING_TIME_GAP:
        output_a_target = min(output_a_target, -0.5)

    self.mpc.source = plan_source
    # A negative acceleration on a downhill grade is road-load balance, not a
    # request to hold the car stopped when the driver has set a moving speed.
    self.output_should_stop = (any(should_stop for _, _, should_stop in candidates)
                               or stopped_lead_loss_hold_active or close_closing_stop) \
      and (not self.is_crv_5g or is_e2e or bool(tracked_leads) or stopped_lead_loss_hold_active or v_cruise <= 0.1)
    output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)
    normal_jerk_limit = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
    emergency_response = bool(self.crv_lead_follow_safety_override)
    response_pressure = max(float(self.crv_lead_follow_urgency),
                            float(emergency_response))
    jerk_limit = max(normal_jerk_limit,
                     self.crv_lead_follow_jerk_limit,
                     CRV_LEAD_FOLLOW_EMERGENCY_BRAKE_JERK
                     if emergency_response else 0.0)
    if not self.is_crv_5g:
      self.output_a_target = output_a_target
    elif lead_follow_active:
      # The terminal maneuver already specifies a jerk-continuous acceleration
      # curve. A second response filter would delay the entire brake phase.
      self.output_a_target = float(np.clip(output_a_target,
                                          a_prev - jerk_limit * self.dt,
                                          a_prev + jerk_limit * self.dt))
      self.crv_accel_jerk = (self.output_a_target - a_prev) / self.dt
    else:
      self.output_a_target, self.crv_accel_jerk = update_accel_scurve(
        output_a_target, a_prev, self.crv_accel_jerk, self.dt, jerk_limit,
        CRV_ACCEL_RESPONSE_WN + response_pressure *
        (CRV_ACCEL_EMERGENCY_RESPONSE_WN - CRV_ACCEL_RESPONSE_WN),
        CRV_ACCEL_MAX_SNAP + response_pressure *
        (CRV_ACCEL_EMERGENCY_MAX_SNAP - CRV_ACCEL_MAX_SNAP))

    # The restart-gap hold is a safety state, not a soft target.  Keep the
    # jerk trajectory continuous, but do not allow its stored momentum to
    # produce a positive command before the lead has opened the gap.
    if self.crv_post_restart_gap_hold:
      self.output_a_target = min(self.output_a_target, 0.0)

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
