"""Demo: agent follows a plan, an obstacle appears mid-path, it re-plans.

Run with:
    PYTHONPATH=. python examples/run_dynamic_demo.py
"""

from __future__ import annotations

from vector.algorithms import astar
from vector.dynamic import DynamicSimulation
from vector.environment import generate_random_environment


def main() -> None:
    grid = generate_random_environment(width=15, height=8, obstacle_density=0.15, seed=5)
    print(grid.to_ascii())
    print()

    sim = DynamicSimulation(
        grid=grid,
        search_fn=lambda g, s, t: astar.search(g, s, t),
        agent_pos=grid.start,
        goal=grid.goal,
    )
    sim.plan()
    print(f"Initial plan: {sim.current_path}")
    print()

    steps_taken = 0
    obstacle_injected = False

    while True:
        # Once the agent is a few steps in, drop an obstacle directly on
        # its planned route (if still ahead of it) to force a re-plan.
        if not obstacle_injected and steps_taken == 3 and len(sim.current_path) > 3:
            block_pos = sim.current_path[3]
            print(f"--- Injecting obstacle at {block_pos} ---")
            sim.toggle_obstacle(block_pos)
            obstacle_injected = True

        reached_goal = sim.step()
        steps_taken += 1
        if reached_goal:
            break
        if steps_taken > grid.width * grid.height:
            print("Safety stop: too many steps.")
            break

    print()
    print(f"Reached goal in {steps_taken} moves, with {sim.replan_count} re-plan(s).")
    print()
    print("Event log:")
    for line in sim.history:
        print(f"  {line}")


if __name__ == "__main__":
    main()
