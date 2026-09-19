"""Core grid data structure used by every algorithm in VECTOR.

The Grid is a pure data structure: it knows nothing about how it was
generated (see environment.py) and nothing about how it is searched
(see algorithms/). This separation lets algorithms depend on a single,
stable interface regardless of whether the grid came from random
generation, a hand-authored map, or a dynamically changing environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

Position = tuple[int, int]  # (row, col)


class CellType(Enum):
    """The structural role of a cell, independent of its movement cost."""

    TRAVERSABLE = "."
    BLOCKED = "#"
    START = "S"
    GOAL = "E"


@dataclass
class Grid:
    """A 2D grid of cells with independent structural type and terrain cost.

    Attributes:
        width: Number of columns.
        height: Number of rows.
        cell_types: height x width array of CellType values.
        costs: height x width array of movement costs (float). A cell's
            cost is only meaningful when the cell is not BLOCKED.
        start: The start position, if set.
        goal: The goal position, if set.
    """

    width: int
    height: int
    cell_types: list[list[CellType]] = field(default_factory=list)
    costs: list[list[float]] = field(default_factory=list)
    start: Position | None = None
    goal: Position | None = None

    def __post_init__(self) -> None:
        if not self.cell_types:
            self.cell_types = [
                [CellType.TRAVERSABLE for _ in range(self.width)]
                for _ in range(self.height)
            ]
        if not self.costs:
            self.costs = [[1.0 for _ in range(self.width)] for _ in range(self.height)]

        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        if len(self.cell_types) != self.height or any(
            len(row) != self.width for row in self.cell_types
        ):
            raise ValueError("cell_types dimensions do not match width/height")
        if len(self.costs) != self.height or any(
            len(row) != self.width for row in self.costs
        ):
            raise ValueError("costs dimensions do not match width/height")

    def in_bounds(self, pos: Position) -> bool:
        """Return True if pos lies within the grid."""
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width

    def is_traversable(self, pos: Position) -> bool:
        """Return True if pos is in bounds and not blocked."""
        return self.in_bounds(pos) and self.cell_types[pos[0]][pos[1]] != CellType.BLOCKED

    def cost_at(self, pos: Position) -> float:
        """Return the movement cost of entering pos.

        Raises:
            ValueError: if pos is out of bounds or blocked.
        """
        if not self.is_traversable(pos):
            raise ValueError(f"Cannot query cost of non-traversable position {pos}")
        return self.costs[pos[0]][pos[1]]

    def set_cell(self, pos: Position, cell_type: CellType, cost: float = 1.0) -> None:
        """Set the type and cost of a single cell."""
        row, col = pos
        self.cell_types[row][col] = cell_type
        self.costs[row][col] = cost
        if cell_type == CellType.START:
            self.start = pos
        elif cell_type == CellType.GOAL:
            self.goal = pos

    def neighbors(self, pos: Position) -> Iterator[Position]:
        """Yield traversable 4-connected neighbors of pos, in fixed order.

        Fixed order (up, right, down, left) keeps algorithm behavior
        deterministic across runs, which matters for reproducible
        benchmarking and for tie-breaking in equal-cost paths.
        """
        row, col = pos
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for d_row, d_col in deltas:
            candidate = (row + d_row, col + d_col)
            if self.is_traversable(candidate):
                yield candidate

    def to_ascii(self) -> str:
        """Render the grid as an ASCII string, e.g. for logging or debugging.

        Traversable cells are rendered by cost tier so weighted terrain
        is visible: cost 1 -> '.', cost 3 -> '~', cost 6 -> '^', any
        other cost -> '.'. Non-traversable/special cells use their
        CellType symbol directly.
        """
        cost_to_symbol = {1.0: ".", 3.0: "~", 6.0: "^"}
        lines = []
        for row_idx, row in enumerate(self.cell_types):
            chars = []
            for col_idx, cell in enumerate(row):
                if cell == CellType.TRAVERSABLE:
                    chars.append(cost_to_symbol.get(self.costs[row_idx][col_idx], "."))
                else:
                    chars.append(cell.value)
            lines.append("".join(chars))
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.to_ascii()
