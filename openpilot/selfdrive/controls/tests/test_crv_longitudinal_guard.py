import unittest
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.crv_longitudinal_guard import CrvLongitudinalGuard


def lead(d_rel=0.0, v_rel=0.0, model_prob=0.0, present=False):
  return SimpleNamespace(dRel=d_rel, vRel=v_rel, modelProb=model_prob, present=present)


def radar(lead_one=None, lead_two=None):
  return SimpleNamespace(leadOne=lead_one or lead(), leadTwo=lead_two or lead())


class TestCrvLongitudinalGuard(unittest.TestCase):
  def test_low_speed_distance_ceiling(self):
    for distance, ceiling in ((6.0, 0.0), (8.0, 0.4), (10.0, 0.8)):
      with self.subTest(distance=distance):
        guard = CrvLongitudinalGuard(0.05)
        accel, should_stop = guard.update(radar(lead(distance, 0.0, 0.9, True)), 2.0, 1.2, False)
        self.assertAlmostEqual(accel, ceiling)
        self.assertFalse(should_stop)

  def test_stop_latch_requires_continuous_departure(self):
    guard = CrvLongitudinalGuard(0.05)
    close = radar(lead(3.0, 0.0, 0.9, True))
    self.assertTrue(guard.update(close, 0.0, -1.0, True)[1])
    self.assertTrue(guard.state.stop_latch_active)

    departing = radar(lead(6.1, 0.3, 0.9, True))
    for _ in range(9):
      self.assertTrue(guard.update(departing, 0.0, 0.5, False)[1])
    self.assertFalse(guard.update(departing, 0.0, 0.5, False)[1])
    self.assertFalse(guard.state.stop_latch_active)

  def test_stop_latch_starts_while_controller_is_stopping(self):
    guard = CrvLongitudinalGuard(0.05)
    _, should_stop = guard.update(radar(lead(3.0, 0.0, 0.9, True)), 0.0, 0.1, False, stopping_state=True)
    self.assertTrue(should_stop)
    self.assertTrue(guard.state.stop_latch_active)

  def test_stop_latch_survives_dropout_and_lead_switch(self):
    guard = CrvLongitudinalGuard(0.05)
    guard.update(radar(lead(3.0, 0.0, 0.9, True)), 0.0, -1.0, True)
    for _ in range(20):
      accel, should_stop = guard.update(radar(), 0.0, 1.0, False)
      self.assertTrue(should_stop)
      self.assertLessEqual(accel, 0.0)
    for _ in range(9):
      guard.update(radar(lead_two=lead(6.2, 0.3, 0.9, True)), 0.0, 1.0, False)
    self.assertTrue(guard.state.stop_latch_active)

  def test_stop_latch_uses_nearest_lead_for_release(self):
    guard = CrvLongitudinalGuard(0.05)
    guard.update(radar(lead(3.0, 0.0, 0.9, True)), 0.0, -1.0, True)
    split_leads = radar(lead(4.0, 0.0, 0.9, True), lead(8.0, 1.0, 0.9, True))
    for _ in range(20):
      _, should_stop = guard.update(split_leads, 0.0, 1.0, False)
      self.assertTrue(should_stop)
    self.assertTrue(guard.state.stop_latch_active)

  def test_closing_guard_activates_immediately_and_debounces_release(self):
    guard = CrvLongitudinalGuard(0.05)
    closing = radar(lead(20.0, -2.0, 0.9, True))
    accel, _ = guard.update(closing, 12.0, 1.0, False)
    self.assertEqual(accel, 0.0)
    self.assertTrue(guard.state.closing_lead_guard_active)

    for _ in range(9):
      accel, _ = guard.update(radar(), 12.0, 1.0, False)
      self.assertEqual(accel, 0.0)
    accel, _ = guard.update(radar(), 12.0, 1.0, False)
    self.assertEqual(accel, 1.0)
    self.assertFalse(guard.state.closing_lead_guard_active)

  def test_guard_never_weakens_braking_and_reset_clears_state(self):
    guard = CrvLongitudinalGuard(0.05)
    closing = radar(lead(5.0, -2.0, 0.9, True))
    accel, _ = guard.update(closing, 2.0, -1.2, False)
    self.assertEqual(accel, -1.2)
    guard.update(radar(), 2.0, 0.5, False, reset=True)
    self.assertFalse(guard.state.stop_latch_active)
    self.assertFalse(guard.state.closing_lead_guard_active)

  def test_recorded_route_1e_stop_release_stays_latched(self):
    """Regression for route 1e around 2808 s; values are anonymized rlog samples."""
    guard = CrvLongitudinalGuard(0.05)

    # The controller was stopping at 2.36 m before the planner requested release.
    accel, should_stop = guard.update(
      radar(lead(2.3598, 0.2465, 0.9997, True)), 0.0, 0.0093, True, stopping_state=True,
    )
    self.assertEqual(accel, 0.0)
    self.assertTrue(should_stop)

    # Half a second later the old build requested positive acceleration at 3.03 m.
    accel, should_stop = guard.update(
      radar(lead(3.0278, 0.3852, 0.9995, True)), 0.0, 0.2098, False,
    )
    self.assertEqual(accel, 0.0)
    self.assertTrue(should_stop)
    self.assertTrue(guard.state.stop_latch_active)

  def test_recorded_route_22_positive_closing_command_is_blocked(self):
    """Regression for route 22 around 904 s, where the old planner accelerated into a closing lead."""
    guard = CrvLongitudinalGuard(0.05)
    accel, should_stop = guard.update(
      radar(lead(8.0045, -8.6007, 0.5954, True)), 14.8469, 0.7765, False,
    )
    self.assertEqual(accel, 0.0)
    self.assertFalse(should_stop)
    self.assertTrue(guard.state.closing_lead_guard_active)

  def test_recorded_braking_commands_are_never_weakened(self):
    """Route-derived approach samples retain planner braking even when a guard also activates."""
    samples = (
      ("route 1e at 1974 s", 32.0821, 44.6612, -4.4550, 0.9955, -1.0519),
      ("route 1e at 2681 s", 0.9005, 6.5726, -0.7838, 0.9985, -0.3627),
      ("route 22 at 2493 s", 29.6027, 26.6188, -4.4799, 0.9967, -1.2554),
      ("route 23 at 244 s", 26.2720, 27.9743, -4.3095, 0.9932, -1.5093),
    )
    for sample, v_ego, d_rel, v_rel, probability, a_target in samples:
      with self.subTest(sample=sample):
        guard = CrvLongitudinalGuard(0.05)
        accel, _ = guard.update(radar(lead(d_rel, v_rel, probability, True)), v_ego, a_target, False)
        self.assertAlmostEqual(accel, a_target)


if __name__ == "__main__":
  unittest.main()
