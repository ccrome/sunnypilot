"""Distance-based stopping reference with a finite, gentle terminal taper.

v = k*d**(3/4) gives a = -3*v**2/(4*d). Its local time reference is
v(t) = v0*(1-t/T)**3, T = 4*d/v0: velocity, acceleration and jerk reach
zero together. Replanning remaining distance avoids a time schedule that
silently accumulates actuator tracking error. Safety remains independent.
"""
import numpy as np

CRV_WHEEL_SPEED_CUTOFF = .58
CRV_STOP_MIN_BRAKE_ACCEL = .3477
CRV_STOP_BRAKE_SPEED_SCALE = 1.2660


def minimum_stop_deceleration(v, coast_accel=0.):
  """Receiving ECU brake floor / road load, identified independently of tests."""
  return max(-coast_accel, CRV_STOP_MIN_BRAKE_ACCEL * np.exp(-(v / CRV_STOP_BRAKE_SPEED_SCALE) ** 2))


class TerminalStop:
  INACTIVE = 0
  BRAKING = 1
  HOLDING = 2

  def __init__(self):
    self.phase = self.INACTIVE
    self.confirmation = 0.
    self.elapsed = 0.
    self.duration = 0.
    self.coefficients = np.zeros(6)


  def update(self, v, gap, stationary, moving, standstill, dt, enabled, coast_accel=0., predicted=False):
    if not enabled or moving:
      self.phase = self.INACTIVE
      self.confirmation = 0.
    self.confirmation = self.confirmation + dt if stationary and enabled else 0.
    if self.phase == self.INACTIVE and enabled and (self.confirmation >= .5 or predicted) and v > 0.:
      self.duration = 1.
      self.coefficients = np.array([v, 0., 0., 0., 0., 0.])
      self.elapsed = 0.
      self.phase = self.BRAKING
    if self.phase != self.INACTIVE:
      reference_speed = self.reference(self.elapsed)[0]
      confidence = max(0., v ** 2 - CRV_WHEEL_SPEED_CUTOFF ** 2) / (v ** 2 + CRV_WHEEL_SPEED_CUTOFF ** 2)
      initial_speed = max(0., reference_speed + confidence * (v - reference_speed))
      remaining = max(gap - 3.75, .1)
      # With unavoidable minimum deceleration b, v² = C*d**(3/2) + 2*b*d
      # lands at d=0 rather than stopping short when the ECU/road-load floor
      # takes over. The +b/2 term follows from that spatial energy equation.
      floor = minimum_stop_deceleration(initial_speed, coast_accel)
      acceleration = min(-.75 * initial_speed ** 2 / remaining + .5 * floor, -floor)
      # Cubic speed is nonnegative and monotone for the complete published
      # horizon, including a stop dominated by minimum brake/grade authority.
      self.duration = max(dt, 3. * initial_speed / max(-acceleration, .01))
      self.coefficients = np.array([initial_speed, -3. * initial_speed, 3. * initial_speed, -initial_speed, 0., 0.])
      self.elapsed = 0.
      self.elapsed += dt
      self.phase = self.HOLDING if standstill else self.BRAKING
    return self.reference(self.elapsed)

  def reference(self, elapsed):
    if self.phase != self.BRAKING:
      return 0., 0., 0.
    s = min(elapsed / self.duration, 1.)
    return tuple(float(np.polynomial.polynomial.polyval(s,
      np.polynomial.polynomial.polyder(self.coefficients, order)) / self.duration ** order) for order in range(3))

  def snap(self):
    s = min(self.elapsed / max(self.duration, .01), 1.)
    return float(np.polynomial.polynomial.polyval(s,
      np.polynomial.polynomial.polyder(self.coefficients, 3)) / max(self.duration, .01) ** 3) * (self.phase == self.BRAKING)


def vision_motion_confidence(model, lead_index, stopped_speed=.5, restart_gap_open=False):
  """Zero must be plausible AND the upper confidence bound must be low.

  Use absolute model motion, not vEgo + vision-relative velocity: the latter
  inherits ego/model velocity disagreement. Ambiguous observations retain
  ordinary following. Genuine motion releases a committed stop immediately.
  """
  if lead_index >= len(model.leadsV3):
    return False, False, False, 0.
  lead = model.leadsV3[lead_index]
  if not len(lead.v) or not len(lead.vStd) or lead.prob < .5:
    return False, False, False, 0.
  velocity, sigma = float(lead.v[0]), max(0., float(lead.vStd[0]))
  stationary = abs(velocity) <= 2. * sigma + 1e-6 and velocity + 2. * sigma <= stopped_speed
  # Finish a committed stop rather than chase a nearly-stopped lead's tiny
  # residual roll. At rest, an opened gap permits a slow genuine restart.
  lower_speed = velocity - 2. * sigma
  moving = lower_speed > stopped_speed or (restart_gap_open and lower_speed > 0.)
  speeds, deviations, times = np.asarray(lead.v), np.asarray(lead.vStd), np.asarray(lead.t)
  if len(speeds) < 2 or not (len(speeds) == len(deviations) == len(times) == len(lead.x)):
    return stationary, moving, False, 0.
  # Future intent is a prediction, not a safety measurement. A stopped mean
  # with zero inside its uncertainty can plan the approach before the lead
  # physically reaches rest; present-motion confidence still owns restart.
  future_stationary = (np.abs(speeds) <= 2. * deviations + 1e-6) & (np.abs(speeds) <= stopped_speed)
  stops = np.flatnonzero(future_stationary & (times > 0.))
  tail_deceleration = (speeds[-1] - speeds[-2]) / max(float(times[-1] - times[-2]), .01)
  stopping_tail = future_stationary[-1] and speeds[-1] > 0. and tail_deceleration < -1e-4
  forecast = bool((len(stops) and np.all(future_stationary[stops[0]:])) or stopping_tail)
  stop_index = int(stops[0]) if len(stops) else len(times) - 1
  travel = max(0., float(np.max(np.asarray(lead.x)[stop_index:])) - float(lead.x[0])) if forecast else 0.
  travel += float(stopping_tail) * max(0., float(speeds[-1])) ** 2 / (2. * max(-tail_deceleration, 1e-4))
  return stationary, moving, forecast, travel
