"""Conservative longitudinal output guards for the fifth-generation CR-V."""

from dataclasses import dataclass
import math

from opendbc.sunnypilot.car.honda.longitudinal_tuning import CRV_LONGITUDINAL_TUNE


@dataclass
class CrvGuardState:
  enabled: bool = True
  stop_latch_active: bool = False
  closing_lead_guard_active: bool = False
  low_speed_limit_active: bool = False
  guarded_lead_index: int = -1
  d_rel: float = 0.0
  v_rel: float = 0.0
  model_prob: float = 0.0
  time_gap: float = 0.0
  accel_ceiling: float = math.inf
  unguarded_a_target: float = 0.0
  guarded_a_target: float = 0.0


class CrvLongitudinalGuard:
  def __init__(self, dt: float):
    self.dt = dt
    self.stop_latch = False
    self.stop_release_time = 0.0
    self.stop_release_lead = -1
    self.closing_guard = False
    self.closing_clear_time = 0.0
    self.state = CrvGuardState()

  def reset(self) -> None:
    self.stop_latch = False
    self.stop_release_time = 0.0
    self.stop_release_lead = -1
    self.closing_guard = False
    self.closing_clear_time = 0.0

  @staticmethod
  def _lead_values(lead, index: int, v_ego: float):
    present = bool(lead.present)
    d_rel = float(lead.dRel)
    v_rel = float(lead.vRel)
    model_prob = float(lead.modelProb)
    if not present or not math.isfinite(d_rel) or d_rel <= 0.0 or model_prob < CRV_LONGITUDINAL_TUNE.lead_probability:
      return None
    return index, d_rel, v_rel, model_prob, d_rel / max(v_ego, 0.1)

  def update(self, radar_state, v_ego: float, a_target: float, should_stop: bool,
             stopping_state: bool = False, reset: bool = False) -> tuple[float, bool]:
    if reset:
      self.reset()
      self.state = CrvGuardState(unguarded_a_target=a_target, guarded_a_target=a_target)
      return a_target, should_stop

    leads = [values for index, lead in enumerate((radar_state.leadOne, radar_state.leadTwo))
             if (values := self._lead_values(lead, index, v_ego)) is not None]
    nearest = min(leads, key=lambda lead: lead[1], default=None)

    if (should_stop or stopping_state) and nearest is not None and nearest[1] <= CRV_LONGITUDINAL_TUNE.close_lead_distance:
      self.stop_latch = True

    if self.stop_latch:
      release_lead = nearest if (
        nearest is not None and nearest[1] >= CRV_LONGITUDINAL_TUNE.close_lead_distance and
        nearest[2] >= CRV_LONGITUDINAL_TUNE.stop_release_v_rel
      ) else None
      if release_lead is not None:
        if self.stop_release_lead != release_lead[0]:
          self.stop_release_lead = release_lead[0]
          self.stop_release_time = 0.0
        self.stop_release_time += self.dt
        if self.stop_release_time + 1e-9 >= CRV_LONGITUDINAL_TUNE.guard_release_time:
          self.stop_latch = False
          self.stop_release_time = 0.0
          self.stop_release_lead = -1
      else:
        self.stop_release_time = 0.0
        self.stop_release_lead = -1

    closing_leads = [lead for lead in leads
                     if lead[2] <= CRV_LONGITUDINAL_TUNE.closing_v_rel and
                     lead[4] <= CRV_LONGITUDINAL_TUNE.closing_time_gap]
    closing_lead = min(closing_leads, key=lambda lead: (lead[4], lead[1]), default=None)
    if closing_lead is not None:
      self.closing_guard = True
      self.closing_clear_time = 0.0
    elif self.closing_guard:
      self.closing_clear_time += self.dt
      if self.closing_clear_time + 1e-9 >= CRV_LONGITUDINAL_TUNE.guard_release_time:
        self.closing_guard = False
        self.closing_clear_time = 0.0

    accel_ceiling = math.inf
    low_speed_limit = v_ego < CRV_LONGITUDINAL_TUNE.low_speed_max
    if low_speed_limit:
      accel_ceiling = CRV_LONGITUDINAL_TUNE.launch_accel_max
      if nearest is not None:
        if nearest[1] <= CRV_LONGITUDINAL_TUNE.close_lead_distance:
          accel_ceiling = 0.0
        elif nearest[1] < CRV_LONGITUDINAL_TUNE.full_launch_distance:
          distance_range = CRV_LONGITUDINAL_TUNE.full_launch_distance - CRV_LONGITUDINAL_TUNE.close_lead_distance
          accel_ceiling = CRV_LONGITUDINAL_TUNE.launch_accel_max * (nearest[1] - CRV_LONGITUDINAL_TUNE.close_lead_distance) / distance_range

    if self.stop_latch or self.closing_guard:
      accel_ceiling = min(accel_ceiling, 0.0)

    guarded_a_target = min(a_target, accel_ceiling)
    guarded_lead = closing_lead or nearest
    self.state = CrvGuardState(
      stop_latch_active=self.stop_latch,
      closing_lead_guard_active=self.closing_guard,
      low_speed_limit_active=low_speed_limit,
      guarded_lead_index=guarded_lead[0] if guarded_lead is not None else -1,
      d_rel=guarded_lead[1] if guarded_lead is not None else 0.0,
      v_rel=guarded_lead[2] if guarded_lead is not None else 0.0,
      model_prob=guarded_lead[3] if guarded_lead is not None else 0.0,
      time_gap=guarded_lead[4] if guarded_lead is not None else 0.0,
      accel_ceiling=accel_ceiling,
      unguarded_a_target=a_target,
      guarded_a_target=guarded_a_target,
    )
    return guarded_a_target, should_stop or self.stop_latch
