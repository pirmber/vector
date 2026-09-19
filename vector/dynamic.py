"""Dynamic environment support (optional layer, Phase 9).

This module does not modify Grid, algorithms, or benchmark — it wraps
them. A DynamicSimulation holds a Grid and a currently-planned path,
lets obstacles be toggled mid-execution, and re-invokes a normal
algorithm's search() to re-plan when the current path becomes invalid.
Nothing about this changes the static core; it's an outer loop around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from vector.algorithms import SearchResult
from vector.grid import CellType, Grid, Position

SearchFn = Callable[[Grid, Position, Position], SearchResult]


@dataclass
class DynamicSimulation:
    """Simulates an agent following a path while obstacles change.

    Attributes:
        grid: The live environment (mutated as obstacles change).
        search_fn: The algorithm's search() function used to (re)plan,
            e.g. `lambda g, s, t: astar.search(g, s, t)`.
        agent_pos: The agent's current position.
        goal: The goal position.
        current_path: The currently planned path from agent_pos to goal.
        replan_count: Number of times re-planning was triggered.
        history: Log of events for later inspection/printing.
    """

    grid: Grid
    search_fn: SearchFn
    agent_pos: Position
    goal: Position
    current_path: list[Position] = field(default_factory=list)
    replan_count: int = 0
    history: list[str] = field(default_factory=list)

    def plan(self) -> SearchResult:
        """Plan (or re-plan) a path from the agent's current position to the goal."""
        result = self.search_fn(self.grid, self.agent_pos, self.goal)
        self.current_path = result.path if result.success else []
        self.history.append(
            f"PLAN from {self.agent_pos}: success={result.success} "
            f"cost={result.total_cost} explored={result.nodes_explored}"
        )
        return result

    def toggle_obstacle(self, pos: Position) -> None:
        """Flip a cell between BLOCKED and TRAVERSABLE, at runtime.

        Refuses to toggle the agent's own current position or the goal,
        since that would trivially invalidate the simulation rather than
        exercise re-planning.
        """
        if pos in (self.agent_pos, self.goal):
            raise ValueError("cannot toggle the agent's own position or the goal")

        current_type = self.grid.cell_types[pos[0]][pos[1]]
        if current_type == CellType.BLOCKED:
            self.grid.set_cell(pos, CellType.TRAVERSABLE)
            self.history.append(f"OBSTACLE CLEARED at {pos}")
        else:
            self.grid.set_cell(pos, CellType.BLOCKED)
            self.history.append(f"OBSTACLE ADDED at {pos}")

    def path_is_valid(self) -> bool:
        """Return True if every remaining step of current_path is still traversable."""
        if not self.current_path:
            return False
        return all(self.grid.is_traversable(pos) for pos in self.current_path)

    def step(self) -> bool:
        """Advance the agent one cell along current_path, re-planning first if invalid.

        Returns:
            True if the agent reached the goal this step, False otherwise.

        Raises:
            RuntimeError: if no valid path exists even after re-planning.
        """
        if not self.path_is_valid():
            result = self.plan()
            self.replan_count += 1
            if not result.success:
                raise RuntimeError(f"no path exists from {self.agent_pos} to {self.goal}")

        # current_path[0] is agent_pos itself; advance to the next cell.
        next_pos = self.current_path[1]
        self.agent_pos = next_pos
        self.current_path = self.current_path[1:]
        self.history.append(f"MOVE to {self.agent_pos}")

        return self.agent_pos == self.goal
