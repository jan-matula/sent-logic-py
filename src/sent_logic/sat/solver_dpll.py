from collections import defaultdict
from dataclasses import dataclass

from . import Clauses, Lit, SolverResult, Var

# Public interface
# ------------------------------------------------------------------------------


def solve(clauses: Clauses) -> SolverResult:
  """
  Decides whether a CNF is satisfiable. This function uses a DPLL-based solver
  (without CDCL).
  """
  return _Solver(clauses).solve()


# Implementation
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AssignmentInfo:
  lit: Lit
  is_deduced: bool


class _ConflictException(Exception):
  pass


class _Solver:
  clauses: Clauses

  total_vars: int
  assigned_vars: int
  lit_scores: defaultdict[Lit, float]
  var_order: list[Var]

  assign_table: dict[Var, Lit]
  assign_stack: list[_AssignmentInfo]
  update_queue: list[Lit]

  watched_table: defaultdict[Lit, set[int]]

  def __init__(self, clauses: Clauses):
    self.clauses = [list(clause) for clause in clauses]
    self.total_vars = 0
    self.assigned_vars = 0
    self.lit_scores = defaultdict(float)
    self.var_order = []
    self.assign_table = {}
    self.assign_stack = []
    self.update_queue = []
    self.watched_table = defaultdict(set)

  # Preprocessing
  # ----------------------------------------------------------------------------

  def init(self):
    """
    Scans the clauses before solving is started. This method:
    - Counts the number of variables
    - Computes Jeroslow-Wang scores for literals
    - Detects empty clauses and unit clauses
    - Selects watched literals for each clause
    """
    self.init_vars()
    self.init_watched()

  def init_vars(self):
    for clause in self.clauses:
      weight = 2.0 ** (-len(clause))
      for lit in clause:
        if lit == 0:
          raise ValueError("Literals must be non-zero.")
        var = abs(lit)
        self.total_vars = max(self.total_vars, var)
        self.lit_scores[lit] += weight

    def var_priority(var: Var) -> float:
      return max(self.lit_scores[var], self.lit_scores[-var])

    self.var_order = sorted(
      range(1, self.total_vars + 1), key=var_priority, reverse=True
    )

  def init_watched(self):
    for clause_index, clause in enumerate(self.clauses):
      if len(clause) == 0:
        raise _ConflictException
      elif len(clause) == 1:
        self.assign_deduced(clause[0])
        continue

      self.watched_table[clause[0]].add(clause_index)
      self.watched_table[clause[1]].add(clause_index)

  # Guessing heuristic
  # ----------------------------------------------------------------------------

  def choose_guess(self) -> tuple[Var, Lit]:
    for var in self.var_order:
      if var not in self.assign_table:
        lit = var if self.lit_scores[var] >= self.lit_scores[-var] else -var
        return var, lit

    # Should be unreachable.
    assert False

  # Assignments
  # ----------------------------------------------------------------------------

  def assign(self, lit: Lit, is_deduced: bool = True):
    var = abs(lit)
    self.assigned_vars += 1
    self.assign_table[var] = lit
    self.assign_stack.append(_AssignmentInfo(lit, is_deduced))
    self.update_queue.append(lit)

  def assign_deduced(self, lit: Lit):
    var = abs(lit)
    if var not in self.assign_table:
      self.assigned_vars += 1
      self.assign_table[var] = lit
      self.assign_stack.append(_AssignmentInfo(lit, is_deduced=True))
      self.update_queue.append(lit)
    elif self.assign_table[var] != lit:
      raise _ConflictException

  # Propagation
  # ----------------------------------------------------------------------------

  def propagate(self):
    while len(self.update_queue) > 0:
      # Watched literals
      assigned_lit = self.update_queue.pop(0)
      clause_indices = list(self.watched_table[-assigned_lit])
      for clause_index in clause_indices:
        # A watched literal of a clause has been assigned as false.
        self.check_clause(-assigned_lit, clause_index)

  def check_clause(self, watched_lit: Lit, clause_index: int):
    clause = self.clauses[clause_index]
    other_watched_lit = 0

    for lit in clause:
      var = abs(lit)
      val = self.assign_table.get(var)

      if val == lit:
        return
      if lit != watched_lit and clause_index in self.watched_table[lit]:
        other_watched_lit = lit
        continue
      if val == None:
        self.watched_table[watched_lit].remove(clause_index)
        self.watched_table[lit].add(clause_index)
        return

    # We have a unit clause!
    if other_watched_lit != 0:
      self.assign_deduced(other_watched_lit)

  # Backtracking
  # ----------------------------------------------------------------------------

  def backtrack(self) -> bool:
    self.update_queue.clear()

    while self.assign_stack:
      assignment = self.assign_stack.pop()
      lit = assignment.lit
      var = abs(lit)
      del self.assign_table[var]
      self.assigned_vars -= 1
      if not assignment.is_deduced:
        self.assign(-lit)
        return True

    # Cannot backtrack
    return False

  # Main loop
  # ----------------------------------------------------------------------------

  def solve(self) -> SolverResult:
    try:
      self.init()
    except _ConflictException:
      return SolverResult(False, None)

    while True:
      try:
        self.propagate()
        assert self.assigned_vars <= self.total_vars
        if self.assigned_vars == self.total_vars:
          return SolverResult(True, _Solver.into_valuation(self.assign_table))

        # Branching
        var, lit = self.choose_guess()
        self.assigned_vars += 1
        self.assign_table[var] = lit
        self.assign_stack.append(_AssignmentInfo(lit, is_deduced=False))
        self.update_queue.append(lit)

      except _ConflictException:
        if not self.backtrack():
          return SolverResult(False, None)

  @staticmethod
  def into_valuation(assign_table: dict[Var, Lit]) -> dict[Var, bool]:
    return {v: assign_table[v] > 0 for v in assign_table}
