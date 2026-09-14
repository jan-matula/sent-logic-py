import random
import unittest

from sent_logic.sat import (
  Clause,
  Clauses,
  eval_clauses,
  solve_brute_force,
)
from sent_logic.sat.solver_dpll import solve

# ==============================================================================
# Solver tests
# ==============================================================================


# Random CNF generation
# ------------------------------------------------------------------------------


def random_clause(num_vars: int, clause_len: int, rng: random.Random) -> Clause:
  vars_chosen = rng.sample(range(1, num_vars + 1), k=min(clause_len, num_vars))
  return [var if rng.random() < 0.5 else -var for var in vars_chosen]


def random_cnf(
  num_vars: int, num_clauses: int, clause_len: int, rng: random.Random
) -> Clauses:
  return [random_clause(num_vars, clause_len, rng) for _ in range(num_clauses)]


# Test cases
# ------------------------------------------------------------------------------


class TestSolverEdge(unittest.TestCase):
  def test_no_clauses(self):
    result = solve([])
    self.assertTrue(result.sat)

  def test_empty_clause(self):
    result = solve([[]])
    self.assertFalse(result.sat)

  def test_single_unit_clause(self):
    result = solve([[1]])
    self.assertTrue(result.sat)
    assert result.vln is not None
    self.assertEqual(result.vln.get(1), 1)

  def test_direct_contradiction(self):
    result = solve([[1], [-1]])
    self.assertFalse(result.sat)

  def test_conflict_via_propagation(self):
    result = solve([[1, 2], [-1, 3], [-2, -3], [1], [2]])
    self.assertFalse(result.sat)


class TestSolverRandom(unittest.TestCase):
  num_vars: int = 10
  num_clauses: int = 20
  clause_len: int = 3
  num_trails: int = 500
  seed: int = 472

  def runTest(self):
    rng = random.Random(self.seed)

    for trial_index in range(self.num_trails):
      with self.subTest(trial=trial_index):
        clauses = random_cnf(
          self.num_vars, self.num_clauses, self.clause_len, rng
        )
        expected = solve_brute_force(clauses)
        result = solve(clauses)

        if not expected.sat:
          self.assertFalse(
            result.sat,
            "Solver reported SAT but instance is UNSAT.\n"
            + f"clauses={clauses}\nsolver assignment={result.vln}",
          )
        else:
          self.assertTrue(
            result.sat,
            "Solver reported UNSAT but instance is SAT.\n"
            + f"clauses={clauses}\nexample satisfying assignment={expected.vln}",
          )
          assert result.vln is not None
          self.assertTrue(
            eval_clauses(clauses, result.vln),
            "Solver's assignment does not satisfy all clauses.\n"
            + f"clauses={clauses}\nsolver assignment={result.vln}",
          )


if __name__ == "__main__":
  _ = unittest.main()
