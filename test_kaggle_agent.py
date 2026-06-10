"""
Local validation of kaggle_agent.MyAgent: stubs the official
agents.agent.Agent base class and drives the agent against the local
engine the way the official runner does (agent issues its own RESETs).

Usage: python test_kaggle_agent.py [game_prefix,...]
"""

import sys
import time
import types

import numpy as np

# Stub the official ARC-AGI-3-Agents package before importing kaggle_agent
agents_pkg = types.ModuleType("agents")
agent_mod = types.ModuleType("agents.agent")


class _StubAgent:
    def __init__(self, game_id=""):
        self.game_id = game_id
        self.action_counter = 0


agent_mod.Agent = _StubAgent
agents_pkg.agent = agent_mod
sys.modules["agents"] = agents_pkg
sys.modules["agents.agent"] = agent_mod

import logging
logging.disable(logging.CRITICAL)

import arc_agi
from arcengine.enums import GameState

import kaggle_agent
from kaggle_agent import MyAgent


def play(game_id, max_actions):
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    fd = env.reset()
    agent = MyAgent(game_id=game_id)

    levels = {}
    actions_this_level = 0
    current_level = 0

    while agent.action_counter < max_actions:
        if agent.is_done([], fd):
            break
        action = agent.choose_action([], fd)
        data = None
        if action.name == "ACTION6":
            ad = action.action_data
            data = {"x": int(ad.x), "y": int(ad.y)}
        fd = env.step(action, data=data)
        agent.action_counter += 1
        actions_this_level += 1
        if fd is None:
            break
        if fd.levels_completed > current_level:
            levels[current_level] = actions_this_level
            current_level = fd.levels_completed
            actions_this_level = 0
        if fd.state is GameState.WIN:
            break
    return levels, agent.action_counter


def rhae(levels, baselines):
    n = len(baselines)
    total = 0.0
    for i in range(n):
        a = levels.get(i, 0)
        if a > 0 and a <= 5 * baselines[i]:
            total += (i + 1) * min(1.0, baselines[i] / a) ** 2
    return total / sum(range(1, n + 1))


def main():
    prefixes = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    arc = arc_agi.Arcade()
    envs = sorted(arc.get_environments(), key=lambda e: e.game_id)
    if prefixes:
        envs = [e for e in envs if any(e.game_id.startswith(p) for p in prefixes)]

    total = 0.0
    t0 = time.time()
    for env_info in envs:
        baselines = env_info.baseline_actions or []
        cap = 5 * sum(baselines)
        # fresh module state per game (RUN_START is global)
        kaggle_agent.RUN_START = time.time()
        try:
            levels, n_actions = play(env_info.game_id, cap)
            score = rhae(levels, baselines)
        except Exception as e:
            levels, n_actions, score = {}, 0, 0.0
            print(f"  ERROR {env_info.game_id}: {type(e).__name__}: {e}")
        total += score
        solved = ",".join(f"L{k}:{v}" for k, v in levels.items()) or "-"
        print(f"{env_info.game_id[:15]:<16} rhae={score*100:5.2f}%  "
              f"actions={n_actions:>5}  solved=[{solved}]")
    print("-" * 50)
    print(f"OVERALL: {total / len(envs) * 100:.3f}%  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
