import itertools
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

# ==============================================================================
# Tree representation
# ==============================================================================


# The types corresponding to connectives have type parameters. The motivation:
# We can define types corresponding to different fragments of sentential logic.
# The definition of these types reflects the recursive definition of fragments.
#
# For example, the type of sentences containing only negation and conjunction
# can be defined as follows:
#
# ```
# type SentNA[A] = (
#   Atom[A]
#   | Not[SentNA[A]]
#   | And[SentNA[A], SentNA[A]]
# )
# ```
#
# Sentences in conjunctive normal form can be defined as follows:
#
# ```
# type Lit[A] = Atom[A] | Not[Atom[A]]
# type Clause[A] = Lit[A] | Or[Clause[A], Lit[A]]
# type SentCNF[A] = Clause[A] | And[SentCNF[A], Clause[A]]
# ```
#
# Each of these types is a subtype of `Sent[A]`. To ensure that these subtyping
# relations hold, the dataclasses corresponding to the connectives must be
# immutable (the `frozen=True` argument). For example, making the dataclass
# `Not`  immutable ensures that `Not[A]` is a subtype of `Not[B]` when `A` is
# a subtype of `B`.


# Connectives
# ------------------------------------------------------------------------------

# What needs to be modified when a connective is added:
#
# - `type Conn[A, O]`
# - `def map_operands`
#
# - `type Sent[A]`
# - `def into_nao`
#
# - `def atoms`
# - `def map_atoms`
# - `def subst`
# - `def subst_simult`
#
# - `def eval_step`
#
# - Much of the display, lexer, and parser code.


@dataclass(frozen=True, slots=True)
class Atom[A]:
  atom: A

  def __str__(self: "Sent[NVar | IVar]"):
    return display(self)


@dataclass(frozen=True, slots=True)
class Not[O]:
  operand: O

  def __str__(self: "Sent[NVar | IVar]"):
    return display(self)


@dataclass(frozen=True, slots=True)
class And[L, R]:
  left: L
  right: R

  def __str__(self: "Sent[NVar | IVar]"):
    return display(self)


@dataclass(frozen=True, slots=True)
class Xor[L, R]:
  left: L
  right: R

  def __str__(self: "Sent[NVar | IVar]"):
    return display(self)


@dataclass(frozen=True, slots=True)
class Or[L, R]:
  left: L
  right: R

  def __str__(self: "Sent[NVar | IVar]"):
    return display(self)


@dataclass(frozen=True, slots=True)
class Cond[L, R]:
  left: L
  right: R

  def __str__(self: "Sent[NVar | IVar]"):
    return display(self)


# Sentences
# ------------------------------------------------------------------------------


type Conn[A, O] = (
  Atom[A] | Not[O] | And[O, O] | Xor[O, O] | Or[O, O] | Cond[O, O]
)
"""
An object of type `Conn[A, O]` is either an atom or a connective with objects of
type `O` in the position of operands. `Sent[A]` is `Conn[A, Sent[A]]`.

See the documentation of `fold` for an illustration of how `Conn` can be used.
"""


type Sent[A] = (
  Atom[A]
  | Not[Sent[A]]
  | And[Sent[A], Sent[A]]
  | Xor[Sent[A], Sent[A]]
  | Or[Sent[A], Sent[A]]
  | Cond[Sent[A], Sent[A]]
)
"""
Sentences with atoms of type `A`.
"""


# General operations on sentences
# ------------------------------------------------------------------------------


def map_operands[A, Oa, Ob](
  conn: Conn[A, Oa], f: Callable[[Oa], Ob]
) -> Conn[A, Ob]:
  """
  Calls a function `f` on the objects in the position of operands. For example,
  `map_operands(And(a, b), f)` returns `And(f(a), f(b))`. Note that `f` is not
  called on atoms meaning that `map_operands(Atom(a))` returns `Atom(a)`.
  """
  # fmt: off
  match conn:
    case Atom(a):    return conn
    case Not(o):     return Not(f(o))
    case And(a, b):  return And(f(a), f(b))
    case Xor(a, b):  return Xor(f(a), f(b))
    case Or(a, b):   return Or(f(a), f(b))
    case Cond(a, b): return Cond(f(a), f(b))
  # fmt: on


def fold[A, R](sent: Sent[A], f: Callable[[Conn[A, R]], R]) -> R:
  """
  The fold operation on sentences.

  We explain this function using an example. Suppose we want to compute the size
  of a sentence. We can define a function that computes the size of a sentence
  based on the size of its subsentences and the top-level connective:
  ```
  def size_step[A](conn: Conn[A, int]) -> int:
    match conn:
      case Atom(_):    return 1
      case Not(o):     return o + 1
      case And(a, b):  return a + b + 1
      case Xor(a, b):  return a + b + 1
      case Or(a, b):   return a + b + 1
      case Cond(a, b): return a + b + 1
  ```
  The we can compute the size of a sentence `sent` using:
  ```
  def size[A](sent: Sent[A]) -> int:
    return fold(sent, size_step)
  ```
  Suppose that `sent` is `Cond(And(Atom(a), Atom(b)), Atom(a))`. Then `fold`
  computes the size of `sent` as follows:
  ```
  size(Cond(And(Atom(a), Atom(b)), Atom(a))) == …
  • size(And(Atom(a), Atom(b))) == …
    • size(Atom(a)) == size_step(Atom(a)) == 1
    • size(Atom(b)) == size_step(Atom(b)) == 1
    … == size_step(And(1, 1)) == 3
    • size(Atom(a)) == size_step(Atom(a)) == 1
    … == size_step(Cond(3, 1)) == 5
  ```
  """
  return f(map_operands(sent, partial(fold, f=f)))


def atoms[A](sent: Sent[A]) -> Generator[A, None, None]:
  """
  Yields the atoms of a sentence from the leftmost to the rightmost. Atoms that
  occur multiple times in the sentence will be yielded multiple times. For a
  function that disregards the relative position and multiplicity of atom
  occurences, see the function `unique_atoms`.
  """
  match sent:
    case Atom(atom):
      yield atom
    case Not(o):
      yield from atoms(o)
    case And(l, r) | Xor(l, r) | Or(l, r) | Cond(l, r):
      yield from atoms(l)
      yield from atoms(r)


def unique_atoms[A](sent: Sent[A]) -> set[A]:
  """
  Returns the set of atoms in a sentence.
  """
  return {a for a in atoms(sent)}


def map_atoms[A, B](sent: Sent[A], f: Callable[[A], B]) -> Sent[B]:
  """
  Transforms every atom in a sentence with a function `f`. Iterates through
  the atoms from the leftmost to the rightmost (this may be significant when
  calling `f` produces side effects).

  For example, `And(Not(Atom(x)), Atom(y))` is mapped to the sentence
  `And(Not(Atom(f(x))), Atom(f(y)))`. `f(x)` is executed first, `f(y)` second.
  """
  # fmt: off
  match sent:
    case Atom(a):    return Atom(f(a))
    case Not(o):     return Not(map_atoms(o, f))
    case And(l, r):  return And(map_atoms(l, f), map_atoms(r, f))
    case Xor(l, r):  return Xor(map_atoms(l, f), map_atoms(r, f))
    case Or(l, r):   return Or(map_atoms(l, f), map_atoms(r, f))
    case Cond(l, r): return Cond(map_atoms(l, f), map_atoms(r, f))
  # fmt: on


def subst[A](sent: Sent[A], x: A, t: Sent[A]) -> Sent[A]:
  """
  Substitutes `t` for every occurence of the atom `x` in `sent`. For
  simultaneous substitution for multiple atoms, see the function `subst_simult`.
  """
  # fmt: off
  match sent:
    case Atom(a):    return t if a == x else sent
    case Not(o):     return Not(subst(o, x, t))
    case And(l, r):  return And(subst(l, x, t), subst(r, x, t))
    case Xor(l, r):  return Xor(subst(l, x, t), subst(r, x, t))
    case Or(l, r):   return Or(subst(l, x, t), subst(r, x, t))
    case Cond(l, r): return Cond(subst(l, x, t), subst(r, x, t))
  # fmt: on


def subst_simult[A](sent: Sent[A], asg: Mapping[A, Sent[A]]) -> Sent[A]:
  """
  Accepts an assignment `asg` mapping atoms `x₁, x₂, ..., xₙ` to sentences
  `t₁, t₂, ..., tₙ` respectively and simultanously substitutes sentences `tᵢ`
  for all occurences of the atoms `xᵢ` in the sentence `sent`. For substitution
  of a single sentence for a single variable, see the function `subst`.

  Note that simultaneous substitution can compute results different from
  sequential substitution. For example, simultanous substitution
  `{x: Atom(y), y: Atom(x)}` into the sentence `Or(Atom(x), Atom(y))` yields
  `Or(Atom(y), Atom(x))`, but `{x: Atom(y)}` followed by `{y: Atom(x)}` yield
  `Or(Atom(x), Atom(x))`.
  """
  # fmt: off
  match sent:
    case Atom(a):    return asg.get(a, sent)
    case Not(o):     return Not(subst_simult(o, asg))
    case And(l, r):  return And(subst_simult(l, asg), subst_simult(r, asg))
    case Xor(l, r):  return Xor(subst_simult(l, asg), subst_simult(r, asg))
    case Or(l, r):   return Or(subst_simult(l, asg), subst_simult(r, asg))
    case Cond(l, r): return Cond(subst_simult(l, asg), subst_simult(r, asg))
  # fmt: on


# Fragments
# ------------------------------------------------------------------------------


type SentNAO[A] = (
  Atom[A]
  | Not[SentNAO[A]]
  | And[SentNAO[A], SentNAO[A]]
  | Or[SentNAO[A], SentNAO[A]]
)
"""
Sentences composed only from atoms, negations, conjunctions, and disjunctions.
"""


def into_nao[A](sent: Sent[A]) -> SentNAO[A]:
  """
  Returns an equivalent NAO sentence. The paraphrases of sentences whose
  principal connective is XOR leads to the duplication of subsentences. This
  means that the paraphrasing a sentence containing the XOR connective can lead
  to an exponential increase of size.
  """
  # fmt: off
  match sent:
    case Atom(_):    return sent
    case Not(o):     return Not(into_nao(o))
    case And(l, r):  return And(into_nao(l), into_nao(r))

    case Xor(l, r):
      l1, r1 = into_nao(l), into_nao(r)
      return Or(And(l1, Not(r1)), And(Not(l1), r1))

    case Or(l, r):   return Or(into_nao(l), into_nao(r))
    case Cond(l, r): return Or(Not(into_nao(l)), into_nao(r))
  # fmt: on


type Lit[A] = Atom[A] | Not[Atom[A]]
"""
Literals are sentences that are either atoms or negations of atoms.
"""

type SentNNF[A] = (
  Lit[A] | And[SentNNF[A], SentNNF[A]] | Or[SentNNF[A], SentNNF[A]]
)
"""
Sentences are in negation normal form (NNF) when they are composed only from
atoms, negations of atoms, conjunctions, and disjunctions. That is, the sentence
can contain only negations, conjunctions, and disjunctions as connectives and
negations can attach only to atoms.
"""


def into_nnf[A](sent: SentNAO[A]) -> SentNNF[A]:
  """
  Returns an equivalent NNF sentence. The size of the paraphrase is equal to the
  size of the original NAO sentence.
  """
  # fmt: off
  match sent:
    case Atom(_):      return sent
    case Not(Atom(_)): return cast(Lit[A], sent)
    case Not(o):       return _into_nnf_aux(o)
    case And(l, r):    return And(into_nnf(l), into_nnf(r))
    case Or(l, r):     return Or(into_nnf(l), into_nnf(r))
  # fmt: on


def _into_nnf_aux[A](sent: SentNAO[A]) -> SentNNF[A]:
  """
  Returns an NNF sentence equivalent to the negation of the argument.
  """
  # fmt: off
  match sent:
    case Atom(_):   return Not(sent)
    case Not(o):    return into_nnf(o)
    case And(l, r): return Or(_into_nnf_aux(l), _into_nnf_aux(r))
    case Or(l, r):  return And(_into_nnf_aux(l), _into_nnf_aux(r))
  # fmt: on


# Sentences with indexed and named variables.
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IVar:
  index: int

  def __post_init__(self):
    assert self.index >= 1


type ISent = Sent[IVar]
"""
Sentences whose atoms are variables indexed by integers (I-sentences for short).
Variable indices must be positive.

Sentences with indexed variables should be constructed so that the set of
variable indices used is of the form `{1, 2, ..., n}` for some `n`. While
functions working with I-sentences should be correct for I-sentences deviating
from this principle, with some functions, deviations will lead to inefficiencies
(especially inefficiencies in memory allocation).
"""

type ISentNAO = SentNAO[IVar]
type ILit = Lit[IVar]
type ISentNNF = SentNNF[IVar]


def nindices(*sents: ISent) -> int:
  """
  Returns the least number `n` such that the set of variable indices occuring in
  the I-sentence is contained in `{1, 2, ..., n}`. This can be used as an
  alternative to `variables` specifically for I-sentences.
  """
  result = 0
  for sent in sents:
    for v in atoms(sent):
      result = max(result, v.index)
  return result


def fresh_index(*sents: ISent) -> int:
  """
  Returns a variable index not occuring in the sentence.
  """
  return nindices(*sents) + 1


@dataclass(frozen=True, slots=True)
class NVar:
  name: str

  def __post_init__(self):
    assert NVar.validate_name(self.name)

  @staticmethod
  def validate_name(name: str) -> bool:
    """
    Returns `True` if the string is a valid variable name. For the restrictions
    concerning valid variable names, see the documentation for `NSent`.
    """
    return len(name) > 0 and all(ch.isalnum() or ch == "_" for ch in name)


type NSent = Sent[NVar]
"""
Sentences whose atoms are named variables. A valid name consists of alphanumeric
characters together with the underscore character `_`.
"""


def names(*sents: NSent) -> set[str]:
  """
  Returns the set of variable names occuring in a sentence whose atoms are named
  variables.
  """
  return {v.name for sent in sents for v in atoms(sent)}


def variables(*sents: Sent[NVar | IVar]) -> set[str | int]:
  """
  Returns the set of variable names and indices in a sentence whose atoms are
  named and indexed variables.
  """
  return {
    (v.name if isinstance(v, NVar) else v.index)
    for sent in sents
    for v in atoms(sent)
  }


def variable(v: int | str) -> IVar | NVar:
  return IVar(v) if isinstance(v, int) else NVar(v)


# Converting between indexed and named variables.
# ------------------------------------------------------------------------------


class NameMap:
  """
  Stores a bidirectional correspondence between variable names and variable
  indices.
  """

  _index_to_name: dict[int, str]
  _name_to_index: dict[str, int]
  _fresh_index: int

  def __init__(self, fresh_index: int = 1):
    assert fresh_index >= 1
    self._index_to_name = {}
    self._name_to_index = {}
    self._fresh_index = fresh_index

  def get_name(self, index: int) -> str | None:
    """
    Returns `None` if the variable index is not mapped to any variable name.
    """
    assert index >= 0
    return self._index_to_name.get(index)

  def get_index(self, name: str) -> int | None:
    """
    Returns `None` if the variable name is not mapped to any variable index.
    """
    assert NVar.validate_name(name)
    return self._name_to_index.get(name)

  def map_index(self, index: int) -> str | int:
    """
    If the variable index is not mapped to any variable name, then this function
    simply returns the variable index.
    """
    assert index >= 0
    name = self.get_name(index)
    return name if name else index

  def map_name(self, name: str) -> int:
    """
    If the variable name is not mapped to any variable index, then this function
    creates a new mapping between that variable name and a new variable index.
    Otherwise, this function is like `get_index`.
    """
    assert NVar.validate_name(name)

    if name in self._name_to_index:
      return self._name_to_index[name]

    index = self._fresh_index
    self._index_to_name[index] = name
    self._name_to_index[name] = index
    self._fresh_index += 1
    return index


def into_named(
  indexed_sent: Sent[IVar | NVar], name_map: NameMap
) -> Sent[NVar | IVar]:
  """
  Convert a sentence with indexed variables into a sentence with named
  variables. This function uses the index-to-name mappings from `name_map`. The
  indexed variables without index-to-name mappings are simply not converted into
  named variables. This function then returns a sentence with mixed variables:
  some variables will be named, some will be indexed. (This functions also works
  when the argument contains named variables; these are ignored.)
  """
  return map_atoms(
    indexed_sent,
    lambda v: (
      variable(name_map.map_index(v.index)) if isinstance(v, IVar) else v
    ),
  )


def into_indexed(named_sent: Sent[IVar | NVar]) -> tuple[ISent, NameMap]:
  """
  Convert a sentence with named variables into a sentence with indexed
  variables. Returns the converted sentence paired with a `NameMap` storing the
  name-to-index mappings used.
  """
  # If the argument contained indexed variables, this function could identify
  # the indexed variables with named variables.
  fresh_index = (
    max(v.index for v in atoms(named_sent) if isinstance(v, IVar)) + 1
  )
  name_map = NameMap(fresh_index)
  indexed_sent = map_atoms(
    named_sent,
    lambda v: IVar(name_map.map_name(v.name)) if isinstance(v, NVar) else v,
  )
  return indexed_sent, name_map


# ==============================================================================
# Semantics
# ==============================================================================


# For the sake of simplicity, the semantics are implemented to support sentences
# with indexed variables only.


type Valuation = dict[str | int, bool]
"""
A valuation. See the documentation for `eval_sent`.
"""

type IValuation = dict[int, bool]


def valuations(vars: set[str | int]) -> Generator[Valuation]:
  """
  Generate all valuations on variables from the set `vars` (a set of variable
  names of type `str` and indices of type `int`).
  """
  var_list = list(vars)
  n = len(var_list)
  for bits in itertools.product([False, True], repeat=n):
    yield {var_list[i]: bits[i] for i in range(n)}


def valuations_(nvars: int) -> Generator[IValuation]:
  """
  Generate all valuations on variables `#1, #2, ..., #n` where `#n` has index
  `nvars` (that is, generate all valuations on the first `nvars` variables).
  """
  assert nvars >= 1
  for bits in itertools.product([False, True], repeat=nvars):
    yield {i: bits[i - 1] for i in range(1, nvars + 1)}


def eval_step[A](conn: Conn[NVar | IVar, bool], vln: Valuation) -> bool:
  """
  A step in the evaluation of a sentence. Evaluates an atom or a connective
  whose operands are evaluated (that is, connective which have boolean values
  in the position of the operands). See the documentations for `Conn` and
  `eval_sent`.
  """
  # fmt: off
  match conn:
    case Atom(v):    return vln[v.name] if isinstance(v, NVar) else vln[v.index]
    case Not(o):     return not o
    case And(a, b):  return a and b
    case Xor(a, b):  return a != b
    case Or(a, b):   return a or b
    case Cond(a, b): return not a or b
  # fmt: on


# Python has already has a function named `eval`.
def eval_sent(sent: Sent, vln: Valuation) -> bool:
  """
  Evaluates an sentence `sent` using a valuation `vln` of the variables. The
  value of the argument `vln` should be a valuation: That is, a `dict` with
  keys of type `str | name` representing variable names and indices and values
  of type `bool` representing the values for the corresponding variables.
  """
  return fold(sent, partial(eval_step, vln=vln))


def eval_sent_(sent: ISent, vln: IValuation) -> bool:
  """
  Like `eval_sent` but specifically for I-sentences and I-valuations.
  """
  # `eval_sent` does not add keys to `vln` which means that we can pretend that
  # `IValuation <: Valuation` in this context.
  return eval_sent(sent, cast(Valuation, vln))


def check_sat(sent: Sent) -> bool:
  """
  Check whether a sentence is satisfiable (evaluates to `True` on some
  valuation).
  """
  vars = variables(sent)
  return any(eval_sent(sent, vln) for vln in valuations(vars))


def check_sat_(sent: ISent) -> bool:
  """
  Like `check_sat` but specifically for I-sentences.
  """
  n = nindices(sent)
  return any(eval_sent_(sent, vln) for vln in valuations_(n))


def check_valid(sent: Sent) -> bool:
  """
  Check whether a sentence is a validity (evaluate to `True` on every
  valuation).
  """
  vars = variables(sent)
  return all(eval_sent(sent, vln) for vln in valuations(vars))


def check_valid_(sent: ISent) -> bool:
  """
  Like `check_valid` but specifically for I-sentences.
  """
  n = nindices(sent)
  return all(eval_sent_(sent, vln) for vln in valuations_(n))


def check_equiv(a: Sent, b: Sent) -> bool:
  """
  Check whether a pair of sentences are equivalent (evaluate to the same value
  on every valuation).
  """
  vars = variables(a).union(variables(b))
  return all(eval_sent(a, vln) == eval_sent(b, vln) for vln in valuations(vars))


def check_equiv_(a: ISent, b: ISent) -> bool:
  """
  Like `check_equiv` but specifically for I-sentences.
  """
  n = max(nindices(a), nindices(b))
  return all(eval_sent_(a, vln) == eval_sent_(b, vln) for vln in valuations_(n))


def check_equisat(a: Sent, b: Sent) -> bool:
  """
  Check whether a pair of sentences are equisatisfiable (both satisfiable or
  both unsatisfiable). Note that equisatisfiability is weaker than equivalence.
  """
  return check_sat(a) == check_sat(b)


def check_equisat_(a: ISent, b: ISent) -> bool:
  """
  Like `check_equisat` but specifically for I-sentences.
  """
  return check_sat_(a) == check_sat_(b)


# ==============================================================================
# Writing and reading sentences
# ==============================================================================


# Display
# ------------------------------------------------------------------------------


def precedence[A, O](conn: Conn[A, O]) -> int:
  """
  The precedence of a connective. Connectives that bind more tightly have higher
  precedence.
  """
  # fmt: off
  match conn:
    case Atom(_):    return 6
    case Not(_):     return 5
    case And(_, _):  return 4
    case Xor(_, _):  return 3
    case Or(_, _):   return 2
    case Cond(_, _): return 1
  # fmt: on


def display_step(conn: Conn[NVar | IVar, tuple[str, int]]) -> tuple[str, int]:
  """
  The return value is a pair consisting of the display string and the
  precedence of the connective. Uses the display strings and precedences of the
  principal connectives of subsentences to construct the display string of a
  sentence. The precedences are used for parenthesization.
  """
  p = precedence(conn)
  # fmt: off
  match conn:
    case Atom(v): s = "#" + str(v.index) if isinstance(v, IVar) else v.name

    case Not((o, po)):           s = "~" + _par(o, po, p)
    case And((a, pa), (b, pb)):  s = _par(a, pa, p + 1) + " & "  + _par(b, pb, p)
    case Xor((a, pa), (b, pb)):  s = _par(a, pa, p + 1) + " ^ "  + _par(b, pb, p)
    case Or((a, pa), (b, pb)):   s = _par(a, pa, p + 1) + " | "  + _par(b, pb, p)
    case Cond((a, pa), (b, pb)): s = _par(a, pa, p + 1) + " => " + _par(b, pb, p)
  return s, p
  # fmt: on


def _par(s: str, ps: int, p: int) -> str:
  return s if ps >= p else "(" + s + ")"


def display(sent: Sent[NVar | IVar]) -> str:
  """
  Returns a string representation of a sentence. See the documentation of
  `parse` for a description of the syntax.
  """
  return fold(sent, display_step)[0]


# Lexing and parsing
# ------------------------------------------------------------------------------


class ParseError(Exception):
  pass


type Lexeme = IVar | NVar | Literal["(", ")", "~", "&", "^", "|", "=>"]
type Lexemes = Generator[Lexeme, None, None]


def _lexemes(s: str) -> Lexemes:
  """
  Divides a sequence of characters into a sentence into a sequence of lexemes. A
  lexeme is either an indexed variable, a named variable, or one of the symbols
  `( ) ~ & ^ | =>`. Insignificant whitespace is discarded.

  For example, the string `"#1 | (#1 => #2)"` is divided into
  `IVar(1), "|", "(", IVar(1), "=>", IVar(2), ")"`.
  """
  i = 0
  while i < len(s):
    # Whitespace
    if s[i].isspace():
      i += 1
      continue

    # Indexed variables
    if s[i] == "#":
      i += 1
      index = ""
      while i < len(s) and s[i].isalnum():
        index += s[i]
        i += 1
      if not index or not index.isnumeric():
        raise ParseError("Expected variable index after `#`.")
      yield IVar(int(index))
      continue

    # Named variables
    if s[i].isalnum() or s[i] == "_":
      name = s[i]
      i += 1
      while i < len(s) and (s[i].isalnum() or s[i] == "_"):
        name += s[i]
        i += 1
      yield NVar(name)
      continue

    # Parentheses and one-character operators.
    if s[i] in ("(", ")", "~", "&", "^", "|"):
      yield cast(Literal["(", ")", "~", "&", "^", "|"], s[i])
      i += 1
      continue

    # The conditional operator
    if s[i:].startswith("=>"):
      i += 2
      yield "=>"
      continue

    raise ParseError


class _Parser:
  lexeme: Lexeme | None
  lexemes: Lexemes

  def __init__(self, lexemes: Lexemes):
    self.lexeme = next(lexemes)
    self.lexemes = lexemes

  def advance(self):
    self.lexeme = next(self.lexemes, None)

  def expect(self, l: Lexeme):
    if not self.lexeme == l:
      raise ParseError(f"Expected `{l}`; found `{self.lexeme}`.")
    self.advance()

  def parse(self, p: int = 0) -> Sent[IVar | NVar]:
    """
    This function tries to consume lexemes until it the sequence of lexemes
    consumed correspends to a sentence.
    """
    # Conditionals
    if p <= 1:
      sent1 = self.parse(p=2)
      if self.lexeme == "=>":
        self.advance()
        sent2 = self.parse(p=1)
        return Cond(sent1, sent2)
      return sent1

    # Disjunctions
    if p <= 2:
      sent1 = self.parse(p=3)
      if self.lexeme == "|":
        self.advance()
        sent2 = self.parse(p=2)
        return Or(sent1, sent2)
      return sent1

    # Exclusive disjunctions
    if p <= 3:
      sent1 = self.parse(p=4)
      if self.lexeme == "^":
        self.advance()
        sent2 = self.parse(p=3)
        return Xor(sent1, sent2)
      return sent1

    # Conjunctions
    if p <= 4:
      sent1 = self.parse(p=5)
      if self.lexeme == "&":
        self.advance()
        sent2 = self.parse(p=4)
        return And(sent1, sent2)
      return sent1

    # Negations
    if self.lexeme == "~":
      self.advance()
      sent = self.parse(p=5)
      return Not(sent)

    # Parentheses
    if self.lexeme == "(":
      self.advance()
      sent = self.parse()
      self.expect(")")
      return sent

    # Variables
    if isinstance(self.lexeme, (IVar, NVar)):
      sent = self.lexeme
      self.advance()
      return Atom(sent)

    raise ParseError(
      "Unexpected end of input."
      if not self.lexeme
      else f"Unexpected lexeme: `{self.lexeme}`."
    )

  def parse_all(self) -> Sent[IVar | NVar]:
    """
    Same as `_Parser.parse`, but checks that the lexemes have been exhausted.
    For example, the former function would parse `p & q (p & q)` with the result
    `And(Atom(NVar("p")), Atom(NVar("p")))` instead of throwing an error.
    """
    sent = self.parse()
    if self.lexeme:
      raise ParseError("Expected end of input.")
    return sent


def parse(s: str) -> Sent[IVar | NVar]:
  """
  Read a sentence from a string.

  ## Syntax
  The `display` and `parse` functions work with the following syntax for
  sentences:

  ```
  <var> := "#1", "#2", "#3", ...  # indexed variables
        |  ...                    # named variables

  <sent_6> := "(" <sent_1> ")" | <var>
  <sent_5> := "~" <sent_5> | <sent_6>

  <sent_4> := <sent_5> ("&"  <sent_5>)*
  <sent_3> := <sent_4> ("^"  <sent_4>)*
  <sent_2> := <sent_3> ("|"  <sent_3>)*
  <sent_1> := <sent_2> ("=>" <sent_2>)*
  ```
  """
  return _Parser(_lexemes(s)).parse_all()
