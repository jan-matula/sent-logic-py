from collections.abc import Iterable
from itertools import combinations

from sent_logic.sat import Clauses, Var
from sent_logic.sat.solver_cdcl import solve as solve_cdcl

type Grid = list[list[int]]
"""
A grid for Sudoku: A 9x9 matrix integers 0–9 as entries. Zero signifies an empty
cell.
"""


def encode_assignment(i: int, j: int, n: int) -> Var:
  """
  Assigns variable indices to tuples `(i, j, n)` where `i` and `j` give a
  position on a Sudoku grid and `n` is an integer 1–9. The corresponding atomic
  sentence signifies that the cell at `(i, j)` contains `n`.
  """
  assert 0 <= i < 9 and 0 <= j < 9 and 1 <= n <= 9
  return 1 + i + 9 * j + 9 * 9 * (n - 1)


def decode_assignment(v: Var) -> tuple[int, int, int]:
  """
  The inverse to `encode_assignment`.
  """
  v = v - 1
  return (v % 9, (v // 9) % 9, (v // 9 // 9) % 9 + 1)


def alternatives(vs: Iterable[Var]) -> Clauses:
  """
  Encodes the exclusive disjunction of the `vs` into clauses.
  """
  vs = list(vs)
  return [vs] + [[-v1, -v2] for v1, v2 in combinations(vs, 2)]


def encode_sudoku_rules() -> Clauses:
  """
  Encodes the general rules of Sudoku into clauses.
  """
  clauses = []

  # Every cell should contain one number:
  for i in range(9):
    for j in range(9):
      clauses += alternatives(encode_assignment(i, j, n) for n in range(1, 10))

  # Every row and column should contain every number:
  for i in range(9):
    for n in range(1, 10):
      clauses += alternatives(encode_assignment(i, j, n) for j in range(9))
      clauses += alternatives(encode_assignment(j, i, n) for j in range(9))

  # Every 3x3 should contain every number:
  for i in range(3):
    for j in range(3):
      for n in range(1, 10):
        clauses += alternatives(
          encode_assignment(i1 + 3 * i, j1 + 3 * j, n)
          for i1 in range(3)
          for j1 in range(3)
        )

  return clauses


def encode_sudoku(grid: Grid) -> Clauses:
  """
  Encode a Sudoku problem into clauses.
  """
  clauses = encode_sudoku_rules()
  clauses.extend(
    [encode_assignment(i, j, grid[i][j])]
    for i in range(9)
    for j in range(9)
    if grid[i][j] != 0
  )
  return clauses


def solve_sudoku_cdcl(grid: Grid) -> Grid | None:
  """
  Solves a Sudoku puzzle using a SAT solver.
  """
  clauses = encode_sudoku(grid)
  result = solve_cdcl(clauses)

  if not result.sat:
    return None

  solved_grid = [[0 for _ in range(9)] for _ in range(9)]

  assert result.vln is not None
  for var, val in result.vln.items():
    if val:
      i, j, n = decode_assignment(var)
      solved_grid[i][j] = n

  return solved_grid


if __name__ == "__main__":
  sudoku: Grid = [
    [9, 2, 0, 3, 0, 6, 0, 0, 0],
    [0, 0, 1, 0, 2, 4, 6, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 4, 0, 0, 0, 7, 0, 0, 0],
    [1, 0, 3, 4, 0, 2, 7, 0, 6],
    [0, 0, 0, 1, 0, 0, 0, 8, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 7, 2, 8, 0, 9, 0, 0],
    [0, 0, 0, 6, 0, 1, 0, 3, 7],
  ]

  solved = solve_sudoku_cdcl(sudoku)
  assert solved is not None
  for row in solved:
    print(*row)
