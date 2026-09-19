"""Visualization for VECTOR.

Deliberately has zero knowledge of algorithm internals — it only
consumes a Grid and a SearchResult (plus, for animation, an ordered
list of explored positions), so any current or future algorithm works
here without changes to this module.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap

from vector.algorithms import SearchResult
from vector.grid import CellType, Grid, Position

# Fixed color encoding so every plot in the project is visually consistent.
COLOR_BLOCKED = "#2b2b2b"
COLOR_TRAVERSABLE_LOW = "#f5f5f5"   # cost 1
COLOR_TRAVERSABLE_MED = "#8fc1e3"   # cost 3
COLOR_TRAVERSABLE_HIGH = "#3a6ea5"  # cost 6
COLOR_EXPLORED = "#ffd166"
COLOR_PATH = "#ef476f"
COLOR_START = "#06d6a0"
COLOR_GOAL = "#118ab2"


def _base_rgb_grid(grid: Grid) -> np.ndarray:
    """Build an (height, width, 3) RGB array reflecting terrain/obstacles."""
    cost_to_color = {1.0: COLOR_TRAVERSABLE_LOW, 3.0: COLOR_TRAVERSABLE_MED, 6.0: COLOR_TRAVERSABLE_HIGH}

    def hex_to_rgb(h: str) -> tuple[float, float, float]:
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    rgb = np.zeros((grid.height, grid.width, 3))
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.cell_types[r][c] == CellType.BLOCKED:
                rgb[r, c] = hex_to_rgb(COLOR_BLOCKED)
            else:
                color = cost_to_color.get(grid.costs[r][c], COLOR_TRAVERSABLE_LOW)
                rgb[r, c] = hex_to_rgb(color)
    return rgb


def plot_result(grid: Grid, result: SearchResult, explored_order: Sequence[Position] | None = None, ax=None, show: bool = True):
    """Render a static plot of the grid with explored nodes and final path.

    Args:
        grid: The environment that was searched.
        result: The SearchResult to visualize.
        explored_order: Optional ordered sequence of explored positions.
            If omitted, only the final path is highlighted (explored
            region is left as terrain).
        ax: Optional matplotlib Axes to draw on. A new figure is created
            if omitted.
        show: If True, calls plt.show() before returning.

    Returns:
        The matplotlib Axes used.
    """
    from matplotlib.colors import to_rgb

    rgb = _base_rgb_grid(grid)

    if explored_order:
        explored_rgb = to_rgb(COLOR_EXPLORED)
        for pos in explored_order:
            rgb[pos[0], pos[1]] = explored_rgb

    if result.success:
        path_rgb = to_rgb(COLOR_PATH)
        for pos in result.path:
            rgb[pos[0], pos[1]] = path_rgb

    if grid.start is not None:
        rgb[grid.start[0], grid.start[1]] = to_rgb(COLOR_START)
    if grid.goal is not None:
        rgb[grid.goal[0], grid.goal[1]] = to_rgb(COLOR_GOAL)

    if ax is None:
        _, ax = plt.subplots(figsize=(grid.width / 6, grid.height / 6))

    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"{result.algorithm_name} — cost={result.total_cost:.1f}, explored={result.nodes_explored}"
    ax.set_title(title, fontsize=10)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def save_result_plot(grid: Grid, result: SearchResult, path: str, explored_order: Sequence[Position] | None = None) -> None:
    """Render and save a static plot to disk (no interactive display)."""
    ax = plot_result(grid, result, explored_order=explored_order, show=False)
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def animate_search(grid: Grid, result: SearchResult, explored_order: Sequence[Position], path: str, interval_ms: int = 40) -> None:
    """Save a GIF/MP4 animating frontier expansion, then the final path.

    Args:
        grid: The environment that was searched.
        result: The SearchResult being animated.
        explored_order: Ordered sequence of positions as they were
            explored (popped from the frontier), used to animate
            expansion frame by frame.
        path: Output file path. Extension determines format (.gif
            requires pillow, .mp4 requires ffmpeg).
        interval_ms: Delay between frames in milliseconds.
    """
    base_rgb = _base_rgb_grid(grid)
    from matplotlib.colors import to_rgb

    explored_rgb = to_rgb(COLOR_EXPLORED)
    path_rgb = to_rgb(COLOR_PATH)
    start_rgb = to_rgb(COLOR_START)
    goal_rgb = to_rgb(COLOR_GOAL)

    fig, ax = plt.subplots(figsize=(grid.width / 6, grid.height / 6))
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(base_rgb.copy(), interpolation="nearest")

    n_explore_frames = len(explored_order)
    n_path_frames = len(result.path) if result.success else 0
    total_frames = n_explore_frames + n_path_frames

    def update(frame_idx: int):
        frame = base_rgb.copy()
        if frame_idx < n_explore_frames:
            for pos in explored_order[: frame_idx + 1]:
                frame[pos[0], pos[1]] = explored_rgb
        else:
            for pos in explored_order:
                frame[pos[0], pos[1]] = explored_rgb
            path_idx = frame_idx - n_explore_frames + 1
            for pos in result.path[:path_idx]:
                frame[pos[0], pos[1]] = path_rgb

        if grid.start is not None:
            frame[grid.start[0], grid.start[1]] = start_rgb
        if grid.goal is not None:
            frame[grid.goal[0], grid.goal[1]] = goal_rgb

        im.set_data(frame)
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=max(total_frames, 1), interval=interval_ms, blit=True)

    if path.endswith(".gif"):
        anim.save(path, writer="pillow", fps=max(1, 1000 // interval_ms))
    else:
        anim.save(path, fps=max(1, 1000 // interval_ms))

    plt.close(fig)
