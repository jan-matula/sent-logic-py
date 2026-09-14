# Module outline

The `sent_logic` module implements:

- The tree representation for the syntax of classical sentential logic
- Basic transformations on the tree representation (folding, substitution, etc.)
- Transformation into negation normal form
- Conversion between sentences with named versus indexed variables
- Semantical functions (evaluation, checking satisfiability, etc.)
- Functions for displaying and parsing sentences

The `sent_logic.sat` module then defines the flat representation of CNF
sentences and implements:

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

# Tests

Tests for the solvers are located at `test/sent_logic/test_sat.py`. Firstly, the
solvers are tested on small instances. We solve these instances using the solver
and using the brute-force method (something that would not be feasible with
large instances) and compare the results. Secondly, the solvers are tested on
large instances. In these second tests, we only check that purported satisfying
assignments produced by the solver actually satisfy the clauses.

To run the tests:

```
pip install .
python -m unittest discover -s test/sent_logic
```
