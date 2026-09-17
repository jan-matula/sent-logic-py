from dataclasses import dataclass

from .. import (
  And,
  Atom,
  Cond,
  Conn,
  ISent,
  ISentNNF,
  IValuation,
  IVar,
  Not,
  Or,
  Xor,
  fold,
  fresh_index,
  into_nnf,
  valuations_,
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


# Basic syntactical operations
# ------------------------------------------------------------------------------


def cnf_or(cnf1: Clauses, cnf2: Clauses) -> Clauses:
  """
  Returns a CNF equivalent to the disjunction of a pair of CNFs. Note that the
  resulting CNF has `n * m` clauses when the arguments have `n` and `m` clauses
  respectively.
  """
  return [clause1 + clause2 for clause1 in cnf1 for clause2 in cnf2]


def into_cnf(sent: ISent) -> Clauses:
  """
  Converts an I-sentence into an equivalent CNF. Note that the size of the
  resulting CNF can be exponential in the size of sentence. This function should
  therefore be used mainly for small sentences.
  """
  sent_nnf: ISentNNF = into_nnf(sent)
  return _nnf_into_cnf(sent_nnf)


def _nnf_into_cnf(sent: ISentNNF) -> Clauses:
  # fmt: off
  match sent:
    case Atom(IVar(v)):      return [[v]]
    case Not(Atom(IVar(v))): return [[-v]]
    case And(l, r):          return _nnf_into_cnf(l) + _nnf_into_cnf(r)
    case Or(l, r):           return cnf_or(_nnf_into_cnf(l), _nnf_into_cnf(r))
  # fmt: on


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
  for vln in valuations_(n):
    if eval_clauses(clauses, vln):
      return SolverResult(True, vln)
  return SolverResult(False, None)


# Conversion into equisatisfiable CNF
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
  Convert a sentence into an equisatisfiable CNF. The size of the CNF and the
  number of variables in the CNF are linear in the size of the sentence. Note
  that CNF will contain variables not appearing the original sentence. This
  means that one cannot, for example, obtain equisatisfiable CNF for `sent1 &
  sent2` by combining the respective equisatisfiable CNFs for `sent1` and `sent2`.
  """
  return _CNFConverter(sent).get_clauses()
