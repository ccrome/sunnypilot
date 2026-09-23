"""Ablations that identify the first unstable L11 stage."""

from __future__ import annotations

import numpy as np

from opendbc.car.honda.values import CAR
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


def run_ablation(candidate_mode: str, initial_speed: float = 29.0) -> np.ndarray:
  plant = Plant(
    lead_relevancy=True,
    speed=initial_speed,
    distance_lead=80.0,
    e2e=False,
    car_fingerprint=CAR.HONDA_CRV_5G,
  )
  rows = []
  while plant.current_time < 45.0:
    t = plant.current_time
    v_lead = initial_speed if t < 12.0 else initial_speed - 1.5 if t < 22.0 else initial_speed
    lead = {"v_lead": v_lead, "d_rel": plant.distance_lead - plant.distance,
            "v_rel": v_lead - plant.speed, "prob": 1.0, "present": True}
    lead_one = dict(lead)
    lead_two = dict(lead)

    if candidate_mode == "noisy_candidates":
      # Keep the physical lead unchanged while alternating the competing
      # candidate obstacle estimates. This isolates source arbitration.
      phase = int(t / 1.0) % 2
      lead_one["d_rel"] += 8.0 if phase else 0.0
      lead_two["d_rel"] += 8.0 if not phase else 0.0
    elif candidate_mode == "small_noise":
      # Near-tied candidates should not make the MPC switch sources every
      # cycle. This is the deterministic regression for source hysteresis.
      phase = int(t / 1.0) % 2
      lead_one["d_rel"] += 0.25 if phase else 0.0
      lead_two["d_rel"] += 0.25 if not phase else 0.0
    elif candidate_mode == "fixed_lead_one":
      lead_two["present"] = False
      lead_two["prob"] = 0.0

    plant.step(
      v_lead=v_lead,
      prob_lead=1.0,
      v_cruise=initial_speed * 3.6,
      lead_one=lead_one,
      lead_two=lead_two,
    )
    rows.append((t, plant.speed, plant.acceleration, plant.planner.output_a_target,
                 plant.distance_lead - plant.distance, str(plant.planner.mpc.source)))

  return np.asarray(rows, dtype=object)


class TestCrvTrackedLeadAttackPoint(OpenpilotTestCase):
  def test_stable_candidates_are_less_switchy_than_noisy_candidates(self):
    stable = run_ablation("stable_candidates")
    noisy = run_ablation("noisy_candidates")
    stable_sources = stable[:, 5].astype(str)
    noisy_sources = noisy[:, 5].astype(str)
    stable_transitions = np.count_nonzero(stable_sources[1:] != stable_sources[:-1])
    noisy_transitions = np.count_nonzero(noisy_sources[1:] != noisy_sources[:-1])
    assert stable_transitions <= 2
    assert noisy_transitions >= 10

  def test_candidate_arbitration_does_not_change_the_physical_lead_request(self):
    stable = run_ablation("stable_candidates")
    noisy = run_ablation("noisy_candidates")
    assert abs(np.ptp(noisy[:, 3].astype(float)) - np.ptp(stable[:, 3].astype(float))) < 0.1

  def test_fixed_lead_candidate_removes_candidate_arbitration(self):
    rows = run_ablation("fixed_lead_one")
    sources = rows[:, 5].astype(str)
    assert np.count_nonzero(sources[1:] != sources[:-1]) <= 2

  def test_near_tied_lead_candidates_do_not_churn_sources(self):
    rows = run_ablation("small_noise")
    sources = rows[:, 5].astype(str)
    assert np.count_nonzero(sources[1:] != sources[:-1]) <= 2
