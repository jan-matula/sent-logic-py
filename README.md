## Module outline

(For documentation, see the docstrings in the source files themselves.)

The `sent_logic` module implements:

- The tree representation for the syntax of classical sentential logic
- Basic transformations on the tree representation (folding, substitution, etc.)
- Transformation into negation normal form
- Conversion between sentences with named versus indexed variables
- Semantical functions (evaluation, checking satisfiability, etc.)
- Functions for displaying and parsing sentences

The `sent_logic.sat` module then defines the flat representation of CNF
sentences and implements:

- Transformation into an equivalent CNF sentence and some convenience functions
  for generating clauses.
- Semantical functions for the flat representation (evaluation, checking
  satisfiability by brute force)
- Transformation of a sentence (tree representation) into an equisatisfiable
  CNF sentence (flat representation)

`sent_logic.sat.solver_dpll` implements a more efficient SAT solver based on
the DPLL algorithm. More specifically: This solver uses unit propagation with
watched literals and guessing with chronological backtracking. The guessing
heuristic is static Jeroslow-Wang. It does not use pure literal elimination,
which is part of the original DPLL algorithm. It also does not implement clause
learning.

Finally, `sent_logic.sat.solver_cdcl` implements a SAT solver based on the CDCL
algorithm. More specifically: This solver also uses watched literals for unit
propagation, but it also implements clause learning and non-chronological
backtracking. The heuristic used in guessing is VSIDS with phase saving.

## How to install

To install the `sent_logic` library, download the source code from Github and
run `python -m pip install .` inside the `sent-logic-py` directory (not
`sent-logic-py/src/sent_logic`). Alternatively, run the following commands.

```sh
git clone https://github.com/jan-matula/sent-logic-py.git
cd sent-logic-py
python -m pip install .
```

## How to use the solvers

The solvers reside in modules `sent_logic.sat.solver_dpll` and
`sent_logic.sat.solver_cdcl` as functions named `solve`. The input to `solve`
should be of type `list[list[int]]`. For example, the CNF `(v₁ | ~v₂ | ~v₃) &
(~v₁ | v₂) & (~v₁ | v₃)` would be encoded as the list `[[1, -2, -3], [-1, 2],
[-1, 3]]` (this CNF is equivalent to `v₁ <=> (v₂ & v₃)`). The output from
`solve` is an object of type `SolverResult` with the following fields:

- `sat: bool` which should be true if the input clauses are satisfiable.
- `vln: dict[int, bool] | None` which should contain a satisfying assignment for
  the input clauses whenever `sat` is set to true.

For example:

```python
from sent_logic.sat import eval_clauses
from sent_logic.sat.solver_cdcl import solve

clauses = [[1, -2, -3], [-1, 2], [-1, 3]]
result = solve(clauses)
#> SolverResult(sat=True, vln={1: True, 2: True, 3: True})
eval_clauses(clauses, result.vln)
#> True
```

## Tests

Tests for the solvers are located at `test/sent_logic/sat/test_solver.py`.
Firstly, the solvers are tested on small instances. We solve these instances
using the solver and using the brute-force method (something that would not be
feasible with large instances) and compare the results. Secondly, the solvers
are tested on large instances. In these second tests, we only check that
purported satisfying assignments produced by the solver actually satisfy the
clauses. (The solvers produce refutations when the clauses are unsatisfiable, so
we do not have a way of feasibly checking the result in these cases.)

To run the tests, install the library (see above) and run:

```sh
python -m unittest discover -s test/sent_logic/sat
```

Note that, by default, the CDCL solver is used as the one being tested. To test
the DPLL solver, one has to alter the source code of the test (see the file
itself).
