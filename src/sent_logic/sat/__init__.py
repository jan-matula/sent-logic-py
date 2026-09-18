from dataclasses import dataclass

from .. import (
  And,
  Atom,
  Cond,
  ISent,
  ISentNNF,
  IValuation,
  IVar,
  Not,
  Or,
  Xor,
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


def nvars(*cnfs: Clauses) -> int:
  """
  Returns the least number `n` such that the variables occurring in the clauses
  are among `{1, 2, ..., n}`.
  """
  return max(abs(lit) for cnf in cnfs for clause in cnf for lit in clause)


# Basic syntactical operations
# ------------------------------------------------------------------------------


def cnf_or(cnf1: Clauses, cnf2: Clauses) -> Clauses:
  """
  Returns a CNF equivalent to the disjunction of a pair of CNFs. Note that the
  resulting CNF has `n * m` clauses when the arguments have `n` and `m` clauses
  respectively.
  """

  return [
    clause1 + clause2
    for clause1 in cnf1
    for clause2 in cnf2
    if all(-lit not in clause2 for lit in clause1)
  ]


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


def eval_cnf(clauses: Clauses, vln: IValuation) -> bool:
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
    if eval_cnf(clauses, vln):
      return SolverResult(True, vln)
  return SolverResult(False, None)


# ==============================================================================
# Conversion into equisatisfiable CNF
# ==============================================================================


# Internal DAG representation
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _NLit:
  var: int
  neg: bool = False

  def negate(self) -> "_NLit":
    return _NLit(self.var, not self.neg)

  def as_int(self) -> int:
    return -self.var if self.neg else self.var


@dataclass(frozen=True)
class _NAnd:
  args: tuple["_Node", ...]


@dataclass(frozen=True)
class _NOr:
  args: tuple["_Node", ...]


@dataclass(frozen=True)
class _NXor:
  left: "_Node"
  right: "_Node"


type _Node = _NLit | _NAnd | _NOr | _NXor


# CNF converter
# ------------------------------------------------------------------------------


class _CNFConverter:
  sent: ISent
  _fresh_index: int
  clauses: Clauses
  _memo: dict[tuple, _Node]
  _vars: dict[_Node, int]

  def __init__(self, sent: ISent) -> None:
    self.sent = sent
    self._fresh_index = fresh_index(sent)
    self.clauses = []
    self._memo = {}
    self._vars = {}

  def fresh_index(self) -> int:
    i = self._fresh_index
    self._fresh_index += 1
    return i

  # Interning / normalization
  # ----------------------------------------------------------------------------

  def _intern(self, key: tuple, node: _Node) -> _Node:
    old = self._memo.get(key)
    if old is not None:
      return old
    self._memo[key] = node
    return node

  def atom(self, i: int) -> _NLit:
    return _NLit(i)

  def make_and(self, args: list[_Node]) -> _Node:
    # Flatten nested conjunctions.
    flat: list[_Node] = []
    for x in args:
      if isinstance(x, _NAnd):
        flat.extend(x.args)
      else:
        flat.append(x)

    # Remove duplicated operands.
    flat = list(set(flat))
    assert flat

    if len(flat) == 1:
      return flat[0]

    # Sort operands.
    flat.sort(key=repr)
    operands_t = tuple(flat)

    return self._intern(("and", operands_t), _NAnd(operands_t))

  def make_or(self, args: list[_Node]) -> _Node:
    # Flatten nested disjunctions.
    flat: list[_Node] = []
    for x in args:
      if isinstance(x, _NOr):
        flat.extend(x.args)
      else:
        flat.append(x)

    # Remove duplicated operands.
    flat = list(set(flat))
    assert flat

    if len(flat) == 1:
      return flat[0]

    flat.sort(key=repr)
    operands_t = tuple(flat)

    return self._intern(("or", operands_t), _NOr(operands_t))

  def make_xor(self, a: _Node, b: _Node) -> _Node:
    if repr(a) > repr(b):
      a, b = b, a

    return self._intern(("xor", a, b), _NXor(a, b))

  # NNF conversion
  # ----------------------------------------------------------------------------

  def to_nnf(self, conn: ISent, polarity: bool = True) -> _Node:
    match conn:
      case Atom(IVar(i)):
        return self.atom(i) if polarity else _NLit(i, True)

      case Not(i):
        return self.to_nnf(i, not polarity)

      case And(i, j):
        if polarity:
          return self.make_and([self.to_nnf(i, True), self.to_nnf(j, True)])
        else:
          return self.make_or([self.to_nnf(i, False), self.to_nnf(j, False)])

      case Or(i, j):
        if polarity:
          return self.make_or([self.to_nnf(i, True), self.to_nnf(j, True)])
        else:
          return self.make_and([self.to_nnf(i, False), self.to_nnf(j, False)])

      case Cond(i, j):
        if polarity:
          return self.make_or([self.to_nnf(i, False), self.to_nnf(j, True)])
        else:
          return self.make_and([self.to_nnf(i, True), self.to_nnf(j, False)])

      case Xor(i, j):
        if polarity:
          return self.make_or(
            [
              self.make_and([self.to_nnf(i, True), self.to_nnf(j, False)]),
              self.make_and([self.to_nnf(i, False), self.to_nnf(j, True)]),
            ]
          )
        else:
          return self.make_or(
            [
              self.make_and([self.to_nnf(i, True), self.to_nnf(j, True)]),
              self.make_and([self.to_nnf(i, False), self.to_nnf(j, False)]),
            ]
          )

  # Tseitin encoding
  # ----------------------------------------------------------------------------

  def encode(self, node: _Node) -> _NLit:
    if isinstance(node, _NLit):
      return node

    old = self._vars.get(node)
    if old is not None:
      return _NLit(old)

    k = self.fresh_index()
    self._vars[node] = k

    if isinstance(node, _NAnd):
      xs = [self.encode(x) for x in node.args]

      for x in xs:
        self.clauses.append([-k, x.as_int()])

      self.clauses.append([k] + [-x.as_int() for x in xs])

    elif isinstance(node, _NOr):
      xs = [self.encode(x) for x in node.args]

      for x in xs:
        self.clauses.append([-x.as_int(), k])

      self.clauses.append([-k] + [x.as_int() for x in xs])

    elif isinstance(node, _NXor):
      a = self.encode(node.left)
      b = self.encode(node.right)

      self.clauses.extend(
        [
          [-a.as_int(), b.as_int(), k],
          [a.as_int(), -b.as_int(), k],
          [-k, a.as_int(), b.as_int()],
          [-k, -a.as_int(), -b.as_int()],
        ]
      )

    return _NLit(k)

  def get_clauses(self) -> Clauses:
    root = self.to_nnf(self.sent, True)
    root_lit = self.encode(root)
    self.clauses.append([root_lit.as_int()])
    return self.clauses


def into_equisat_cnf(sent: ISent) -> Clauses:
  """
  Convert a sentence into an equisatisfiable CNF. The size of the CNF and the
  number of variables in the CNF are linear in the size of the sentence. Note
  that CNF will contain variables not appearing the original sentence. This
  means that one cannot, for example, obtain equisatisfiable CNF for `sent1 &
  sent2` by combining the respective equisatisfiable CNFs for `sent1` and `sent2`.
  """
  return _CNFConverter(sent).get_clauses()
