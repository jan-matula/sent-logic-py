from dataclasses import dataclass

from .. import (
  And,
  Atom,
  Cond,
  Conn,
  ISent,
  IValuation,
  IVar,
  Not,
  Or,
  Xor,
  fold,
  fresh_index,
  valuations,
)

# CNF representation
# ------------------------------------------------------------------------------


type Var = int
"""
Variables are represented by positive integers. We emphasize that zero does not
represent a variable.
"""

type Lit = int
"""
Literals are variables and negations of variables. Positive literals are
represented by positive integers and negative literals by negative integers. For
example, `~#4` (a negative literal) is represented as `-4` and `#3` (a positive
literal) is represented as `+3`. Zero does not represent a literal.
"""

type Clause = list[Lit]

type Clauses = list[Clause]
"""
CNF is represented as a list of clauses. Clauses are represented as lists of
literals. Literals are represented by non-zero integers. For example, the list
`[[-1, 2], [2, -3, 4], [1]]` represents the sentence `(~#1 | #2) & (#2 | ~#3 | #4) & (1)`.
The empty CNF is always true and the empty clause is always false. Therefore,
`[]` is trivially satisfiable but `[[-1, 2], []]` is unsatisfiable.
"""


def nvars(clauses: Clauses) -> int:
  """
  Returns the least number `n` such that the variables occurring in the clauses
  are among `{1, 2, ..., n}`.
  """
  return max(abs(lit) for clause in clauses for lit in clause)


# Basic semantics
# ------------------------------------------------------------------------------


def eval_lit(lit: Lit, vln: IValuation) -> bool:
  """
  Evaluates a literal relative to a valuation.
  """
  return vln[abs(lit)] == (lit > 0)


def eval_clauses(clauses: Clauses, vln: IValuation) -> bool:
  """
  Evaluates a CNF relative to a valuation.
  """
  return all(any(eval_lit(lit, vln) for lit in clause) for clause in clauses)


@dataclass(frozen=True, slots=True)
class SolverResult:
  """
  Returned by functions checking whether a CNF is satisfiable (`solve_brute_force`,
  `solver_dpll.solve`, and `solver_cdcl.solve`). When the CNF is satisfiable,
  the `sat` field should be true and `vln` should be a satisfying valuation.
  Otherwise, `sat` should be false and `vln` should be set to `None`.
  """

  sat: bool
  vln: IValuation | None


def solve_brute_force(clauses: Clauses) -> SolverResult:
  """
  Decides whether a CNF is satisfiable. This function iterates through all
  valuations of the variables and checks whether they satisfy the clauses. This
  function is used for testing more sophisticated solutions for the same task.
  See the documentation on `SolverResult`.
  """
  n = nvars(clauses)
  for vln in valuations(n):
    if eval_clauses(clauses, vln):
      return SolverResult(True, vln)
  return SolverResult(False, None)


# Conversion into CNF
# ------------------------------------------------------------------------------


class _CNFConverter:
  sent: ISent
  _fresh_index: int
  clauses: list[list[int]]

  def __init__(self, sent: ISent) -> None:
    self.sent = sent
    self._fresh_index = fresh_index(sent)
    self.clauses = []

  def fresh_index(self) -> int:
    i = self._fresh_index
    self._fresh_index += 1
    return i

  def step(self, conn: Conn[IVar, int]) -> int:
    match conn:
      case Atom(IVar(i)):
        return i
      case Not(i):
        return -i
      case And(i, j):
        k = self.fresh_index()
        # k <=> i & j
        # (i & j => k) & (k => i) & (k => j)
        # (~i | ~j | k) & (~k | i) & (~k | j)
        self.clauses.extend([[-i, -j, k], [-k, i], [-k, j]])
        return k
      case Xor(i, j):
        k = self.fresh_index()
        # k <=> i ^ j
        # (i & ~j => k) & (~i & j => k) & (k => i | j) & (k => ~i | ~j)
        # (~i | j | k) & (i | ~j | k) & (~k | i | j) & (~k | ~i | ~j)
        self.clauses.extend([[-i, j, k], [i, -j, k], [-k, i, j], [-k, -i, -j]])
        return k
      case Or(i, j):
        k = self.fresh_index()
        # k <=> i | j
        # (i => k) & (j => k) & (k => i | j)
        # (~i | k) & (~j | k) & (~k | i | j)
        self.clauses.extend([[-i, k], [-j, k], [-k, i, j]])
        return k
      case Cond(i, j):
        k = self.fresh_index()
        # k <=> (i => j)
        # (k & i => j) & (~i => k) & (j => k)
        # (~k | ~i | j) & (i | k) & (~j | k)
        self.clauses.extend([[-k, -i, j], [i, k], [-j, k]])
        return k

  def get_clauses(self) -> list[list[int]]:
    k = fold(self.sent, self.step)
    self.clauses.append([k])
    return self.clauses


def into_clauses(sent: ISent) -> Clauses:
  """
  Convert a sentence into an equisatisfiable CNF.
  """
  return _CNFConverter(sent).get_clauses()
