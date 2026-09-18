import heapq
from collections import defaultdict
from dataclasses import dataclass

from . import Clause, Clauses, Lit, SolverResult, Var

# Public interface
# ------------------------------------------------------------------------------


def solve(clauses: Clauses) -> SolverResult:
  """
  Decides whether a CNF is satisfiable. This function uses a CDCL-based solver.
  """
  return _Solver(clauses).solve()


# Implementation
# ------------------------------------------------------------------------------


type _Level = int
type _ClauseIndex = int


@dataclass(frozen=True, slots=True)
class _AssignmentInfo:
  lit: Lit
  level: _Level
  reason: _ClauseIndex | None


class _ConflictException(Exception):
  reason: _ClauseIndex

  def __init__(self, reason: _ClauseIndex):
    super().__init__()
    self.reason = reason


@dataclass(frozen=True, slots=True)
class _AnalysisResult:
  learned_clause: Clause
  uip_lit: Lit
  snd_lit: Lit | None
  backtrack_level: _Level


class _Solver:
  clauses: Clauses
  total_vars: int
  assigned_vars: int

  var_scores: dict[Var, float]
  bump_amount: float = 1.0
  decay_factor: float = 0.95
  var_order: list[tuple[float, Var]]  # max-heap
  save_table: dict[Var, Lit]

  dec_level: _Level
  assign_table: dict[Var, Lit]
  level_table: dict[Var, _Level]
  reason_table: dict[Var, _ClauseIndex | None]
  assign_stack: list[_AssignmentInfo]
  update_queue: list[Lit]

  watched_table: defaultdict[Lit, set[_ClauseIndex]]

  def __init__(self, clauses: Clauses):
    self.clauses = [list(clause) for clause in clauses]
    self.total_vars = 0
    self.assigned_vars = 0

    self.var_scores = {}
    self.var_order = []
    self.save_table = {}

    self.dec_level = 0
    self.assign_table = {}
    self.level_table = {}
    self.reason_table = {}
    self.assign_stack = []
    self.update_queue = []

    self.watched_table = defaultdict(set)

  # Preprocessing
  # ----------------------------------------------------------------------------

  def init(self):
    """
    Scans the clauses before solving is started. This method:
    - Validates the clauses
    - Counts the number of variables
    - Computes initial variable scores
    - Detects empty clauses and unit clauses
    - Selects watched literals for each clause
    """
    self.init_vars()
    self.init_vsids()
    self.init_watched()

  def init_vars(self):
    for clause in self.clauses:
      for lit in clause:
        if lit == 0:
          raise ValueError("Literals must be non-zero.")
        var = abs(lit)
        self.total_vars = max(self.total_vars, var)

  def init_vsids(self):
    self.var_scores = {var: 0.0 for var in range(1, self.total_vars + 1)}
    for clause in self.clauses:
      for lit in clause:
        var = abs(lit)
        self.var_scores[var] += 1.0

    self.bump_amount = 1.0
    self.var_order = [(-score, var) for var, score in self.var_scores.items()]
    heapq.heapify(self.var_order)

  def init_watched(self):
    for clause_index, clause in enumerate(self.clauses):
      match len(clause):
        case 0:
          raise _ConflictException(clause_index)
        case 1:
          self.assign_deduced(clause[0], reason=clause_index)
          continue
        case _:
          self.watched_table[clause[0]].add(clause_index)
          self.watched_table[clause[1]].add(clause_index)

  # Guessing heuristic
  # ----------------------------------------------------------------------------

  def bump_var(self, var: Var):
    self.var_scores[var] += self.bump_amount
    heapq.heappush(self.var_order, (-self.var_scores[var], var))

  def decay(self):
    self.bump_amount /= self.decay_factor
    if self.bump_amount > 1e100:
      self.rescale()

  def rescale(self):
    for var in self.var_scores:
      self.var_scores[var] /= self.bump_amount
    self.var_order = [(-score, var) for var, score in self.var_scores.items()]
    heapq.heapify(self.var_order)
    self.bump_amount = 1.0

  def choose_guess(self) -> Lit:
    while self.var_order:
      _, var = heapq.heappop(self.var_order)
      if var not in self.assign_table:
        return self.save_table.get(var, var)

    # Should be unreachable.
    assert False

  # Assignments
  # ----------------------------------------------------------------------------

  def assign_guessed(self, lit: Lit):
    var = abs(lit)
    self.assigned_vars += 1
    self.assign_table[var] = lit
    self.level_table[var] = self.dec_level
    self.reason_table[var] = None
    self.assign_stack.append(
      _AssignmentInfo(lit, level=self.dec_level, reason=None)
    )
    self.update_queue.append(lit)

  def assign_deduced(self, lit: Lit, reason: _ClauseIndex):
    var = abs(lit)
    if var not in self.assign_table:
      self.assigned_vars += 1
      self.assign_table[var] = lit
      self.level_table[var] = self.dec_level
      self.reason_table[var] = reason
      self.assign_stack.append(
        _AssignmentInfo(lit, level=self.dec_level, reason=reason)
      )
      self.update_queue.append(lit)
    elif self.assign_table[var] != lit:
      raise _ConflictException(reason)

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

  def check_clause(self, watched_lit: Lit, clause_index: _ClauseIndex):
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
      self.assign_deduced(other_watched_lit, clause_index)

  # Conflict analysis
  # ----------------------------------------------------------------------------

  def analyze(self, clause_index: _ClauseIndex) -> _AnalysisResult:
    clause = self.clauses[clause_index]
    learned: Clause = []

    candidates: set[Var] = set()
    level_candidates = 0
    for l in clause:
      var = abs(l)
      self.bump_var(var)
      candidates.add(var)
      if self.level_table[var] == self.dec_level:
        level_candidates += 1
      else:
        learned.append(l)

    i = len(self.assign_stack) - 1
    while level_candidates > 1:
      # Find next current-level candidate
      while True:
        lit = self.assign_stack[i].lit
        var = abs(lit)
        if var in candidates:
          reason = self.assign_stack[i].reason
          break
        i -= 1

      if reason == None:
        candidates.discard(var)
        i -= 1
        continue

      # Resolve with the reason clause
      clause = self.clauses[reason]
      for l in clause:
        v = abs(l)
        if v in candidates:
          continue
        self.bump_var(v)
        candidates.add(v)
        if self.level_table[v] == self.dec_level:
          level_candidates += 1
        else:
          learned.append(l)

      candidates.discard(var)
      level_candidates -= 1
      i -= 1

    # Find the unique current-level variable.
    uip_var = next(
      v for v in candidates if self.level_table[v] == self.dec_level
    )
    uip_lit = -self.assign_table[uip_var]
    learned.append(uip_lit)

    # Find second literal and backtrack level.
    snd_lit = None
    backtrack_level = 0
    for l in learned:
      v = abs(l)
      if v != uip_var and self.level_table[v] > backtrack_level:
        backtrack_level = self.level_table[v]
        snd_lit = l
    return _AnalysisResult(learned, uip_lit, snd_lit, backtrack_level)

  # Backtracking
  # ----------------------------------------------------------------------------

  def backtrack_to(self, level: _Level):
    self.update_queue.clear()
    while self.assign_stack and self.assign_stack[-1].level > level:
      info = self.assign_stack.pop()
      var = abs(info.lit)
      if info.reason is None:
        heapq.heappush(self.var_order, (-self.var_scores[var], var))
      self.save_table[var] = info.lit
      del self.assign_table[var]
      del self.level_table[var]
      del self.reason_table[var]
      self.assigned_vars -= 1
    self.dec_level = level

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
        lit = self.choose_guess()
        self.dec_level += 1
        self.assign_guessed(lit)

      except _ConflictException as e:
        if self.dec_level == 0:
          return SolverResult(False, None)

        self.decay()
        analysis = self.analyze(e.reason)
        self.backtrack_to(analysis.backtrack_level)

        clause_index = len(self.clauses)
        self.clauses.append(analysis.learned_clause)
        if analysis.backtrack_level > 0:
          assert analysis.snd_lit != None
          clause_index = len(self.clauses)
          self.clauses.append(analysis.learned_clause)
          self.watched_table[analysis.uip_lit].add(clause_index)
          self.watched_table[analysis.snd_lit].add(clause_index)

        self.assign_deduced(analysis.uip_lit, reason=clause_index)
        self.update_queue.append(analysis.uip_lit)

  @staticmethod
  def into_valuation(assign_table: dict[Var, Lit]) -> dict[Var, bool]:
    return {v: assign_table[v] > 0 for v in assign_table}
