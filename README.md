# ARC-AGI-3 Agent — Frugal Go-Explore Graph Explorer

An AI agent for [ARC-AGI-3](https://arcprize.org/arc-agi/3), the interactive reasoning
benchmark behind [ARC Prize 2026](https://arcprize.org/competitions/2026/arc-agi-3).
Unlike the original ARC's static grid puzzles, ARC-AGI-3 tasks are **playable games**:
the agent observes a 64×64 grid of 16 colors, takes actions (move / rotate / click /
undo), and must discover the rules and win condition entirely through exploration —
no instructions, no reward signal beyond level completion.

Frontier LLMs (GPT-5, Claude, Gemini) score **under 1%** on this benchmark.
The competition requires fully offline agents: no API calls, no internet during
evaluation.

## Why no neural network?

The scoring metric (RHAE — Relative Human Action Efficiency) is brutal about wasted
actions:

```
level_score = min(1, human_actions / agent_actions)²   →  0 beyond 5× human
```

Halving the action count quadruples the score. An online-learning CNN (the
1st-place preview approach) burns 200–300 actions per level before its predictions
are useful — which alone blows the 5× cutoff on most levels. This agent instead uses
a **training-free effect model**: Laplace-smoothed running tallies of which action
types, clicked colors, and screen regions actually change the game state. It is
useful from the very first observation and costs nothing to "train."

## Architecture

```
frame ──▶ stable state hash ──▶ state graph ──▶ action selection
            (frozen UI mask)      (nodes = states,    │
                                   edges = actions)   │
              ┌────────────────────────────────────────┘
              ▼
   1. untested candidate with highest effect-model priority, else
   2. BFS to nearest frontier node, replay path (return-then-explore), else
   3. expand candidate tiers (object pixels → coarse grid)
```

Key design decisions, each tied to the scoring math:

| Component | What it does | Why |
|-----------|-------------|-----|
| **Frozen UI mask** | Volatile border pixels (step counters, timers) are detected early, then the mask is frozen for the level | Identical play states must hash the same or the graph corrupts; freezing prevents mid-level hash drift |
| **Effect model** | Tallies P(frame changes) per action type / clicked color / screen cell, with an exploration bonus for untried action types | Try the most promising action *first* — the main lever on action efficiency |
| **Component-snapped clicks** | Click candidates come from connected-component segmentation, deduped by (color, region), not blind grid scans | A 64×64 click space is 4096 actions; objects cut it to ~16 |
| **Return-then-explore** | When a state is exhausted, BFS over known edges to the nearest state with untested actions and replay the path | Go-Explore-style frontier navigation beats random walks; the deterministic engine makes replay exact |
| **Death attribution** | A killing action is recorded *before* the recovery reset: per-level hard ban + cross-level soft penalty | Never repeat a fatal mistake; deaths cost a reset action and restart progress |
| **Cross-level transfer** | The effect model persists across levels within a game | Mechanics usually carry over, and later levels carry more scoring weight — transferred knowledge compounds exactly where the points are |

## Results

| Agent | Local RHAE (25 public games) |
|-------|------------------------------|
| Random baseline | 0.000% |
| Naive graph explorer | 0.236% |
| StochasticGoose CNN (1st-place preview approach) | 0.05% on Kaggle |
| **FrugalExplorer v3** | **0.364%** |

Current Kaggle leaderboard: 0.08% (earlier explorer version; v3 submission
pending). For context, the live leaderboard leader is at 1.21% — the entire
field is compressed under 1.3%, and every solved-within-budget level moves
rank significantly. Highlights: two levels solved *faster than the human
baseline* (tn36 L0 at 0.5x, lp85 L0 at 0.8x).

The single biggest win came from reverse-engineering the engine's UI overlay:
a step counter drawn down column 0 ticks every few actions — too slowly for
volatility detection — so identical play states hashed differently, silently
corrupting the state graph in every game. Masking the engine UI strip from
hashes tripled the score in one change.

## Repo structure

```
core/
  frugal_explorer.py   # the agent: graph + effect model + frontier navigation
  effect_model.py      # training-free action-effect statistics
  game_runner.py       # play loop, RHAE scoring, death-continuation
  frame_parser.py      # connected components, frame diffs, object tracking
  explorer_agent.py    # earlier naive graph explorer (baseline)
harness.py             # 25-game local sweep mirroring the leaderboard metric
runs/                  # timestamped results for every harness run
submission.ipynb       # Kaggle submission notebook
environment_files/     # public games (downloadable via the arc-agi SDK)
```

## Running locally

```bash
pip install arc-agi numpy scipy
# register a free API key at https://three.arcprize.org, then:
python harness.py                      # full 25-game sweep (~45s)
python harness.py --agent random       # baseline comparison
python harness.py --games cn04,lp85    # debug specific games
python run_agent.py cn04-2fe56bfb      # single game with per-level detail
```

Every harness run saves a JSON snapshot to `runs/`, so score progress is
tracked across agent versions.

## Status

Active development for ARC Prize 2026 Milestone 1 (June 30, 2026). See
`project-brief.md` for the research log, competitive analysis, and build plan.

## License

[MIT-0](LICENSE) (MIT No Attribution), per ARC Prize 2026 open-source
requirements. Game sources under `environment_files/` are MIT-licensed by the
ARC Prize Foundation and retain their original notices.
