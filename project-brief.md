# ARC-AGI-3 — Interactive Game-Solving Agent

## Target: 0.65%+ RHAE (Top 5 on live Kaggle leaderboard — recalibrated 2026-06-09)

> Original target of 20% was based on preview-competition scores. The full benchmark is far harder:
> the live leaderboard leader (Tufa Labs) sits at **1.21%**, and top-5 is ~**0.64%**. See "Live
> Leaderboard Research" below.

## What Is This?

An AI agent that solves ARC-AGI-3 interactive game environments. Unlike original ARC (static grid→grid), ARC-AGI-3 tasks are **playable games** — the agent observes 64x64 pixel grids (16 colors), takes actions (move/rotate/click/undo), and must achieve a win condition within a step budget.

Each task is a Python game with 6+ levels of increasing difficulty. There are no instructions — the agent must **discover the rules** by exploring, then solve efficiently.

## Why This Project?

- **For target roles (Uber/Google/Meta):** Demonstrates reasoning under uncertainty, world-model building, LLM-as-agent, and systems engineering
- **Competition:** ARC Prize 2026 offers $850K+ in prizes; Milestone #1 deadline is **June 30, 2026** ($35K top prizes)
- **Differentiator:** Current frontier models score <1%. Even 20% would be a top leaderboard position

## Competition Landscape (as of June 2026)

| Approach | Score | Offline? | Key Insight |
|----------|-------|----------|-------------|
| Frontier LLMs raw (Claude/GPT/Gemini) | 0.25-0.37% | No | Pure reasoning fails without exploration |
| StochasticGoose (CNN+RL, Tufa Labs) | 12.58% | **Yes** | Learn which actions change state, train CNN online |
| Blind Squirrel (Graph+ResNet18) | 6.71% | **Yes** | State graph + learned value model for action ranking |
| Graph-Based Exploration (3rd place) | ~10% | **Yes** | State graph + priority-based action selection, no NN |
| Executable World Models (arxiv) | 32.58% | No | Coding agent builds Python world model (requires LLM API) |
| **Our ExplorerAgent (current)** | **0.08%** | **Yes** | Basic graph exploration, no learning |

**Key takeaway (SUPERSEDED — see below):** Preview scores (10-12%) did NOT transfer to the full benchmark. The entire live leaderboard is under 1.3%.

---

## Live Leaderboard Research (2026-06-09)

### Actual standings (Kaggle CLI, live)

| Rank | Team | Score | Note |
|------|------|-------|------|
| 1 | Tufa Labs | **1.21** | StochasticGoose authors (Dries Smit) |
| 2 | Redfield Rentals | 0.68 | |
| 3-5 | Barada Sahu / Kevin E R MILLE / SVG | 0.66-0.65 | |
| ~10 | various | ~0.59 | |
| ~800 | **us** | **0.08** | v3 explorer; v6 CNN pending |

- **Top-5 bar ≈ 0.64-0.66. #1 = 1.21.** Milestone #1 ($25K/$10K/$2.5K) awarded June 30.
- Best *public* notebooks score **0.35-0.46** — forking the meta gets most of the way to top-10.

### Top public notebooks analyzed (code downloaded to /tmp/arc3-research)

1. **[0.46] Persistent Memory BFS** (`nihilisticneuralnet`) — multi-algorithm search (beam, IDA*, bounded BFS, A*, MCTS) run against the *imported game engine source* in a background thread, plus CNN with prioritized experience replay (PER), cross-level Dijkstra solution transfer, and expert-demo injection from solver solutions into the RL buffer.
2. **[0.35] FORGE v16 trigger-aware BFS** (`aadigupta1601`) — engine-source BFS with hidden-field probing ("trigger-aware" state hashing: probes internal game variables that change without pixel changes), counter-A*, sprite permutation for click games, IDDFS, solution transfer with object-relative offsets.
3. **FORGE v21 / Ash's agent** (`ashvinsingh`, updated June 8, most-voted active notebook) — **dropped torch AND engine introspection entirely.** Training-free, frugal Go-Explore graph explorer with a tally-based "effect model" (Laplace-smoothed per-action / per-color / spatial-heatmap change rates), frozen per-level UI mask for stable state hashing, connected-component candidate generation, return-then-explore via reset+replay, death attribution, and cross-level persistence of the effect model.

### Critical strategic findings

1. **Engine introspection (importing game source and brute-force searching it) is a trap.**
   Ash's v21 notebook states the official rerun is sandboxed (engine not importable for hidden
   games), it's against the stated spirit of the competition, and **prize-eligible solutions are
   screened for it — disqualification risk.** Since our goal is milestone prize + portfolio, avoid it.
2. **The CNN is questionable under RHAE.** It burns ~200-300 actions per level learning before it
   exploits — with the 5x-human hard cutoff and squared penalty, those wasted actions zero out most
   levels. Ash (who iterated FORGE v8→v21) removed torch because "a from-scratch net never learned a
   sparse single-reward level inside one budget."
3. **Frugality is the scoring lever.** level_score = (human/agent)², capped, 0 beyond 5x human.
   Halving action count ≈ 4x level score. Efficient exploration beats more exploration.
4. **Cross-level transfer is legit and compounds where points are** (later levels weigh more):
   carry effect model + learned action vocabulary across levels; try previous level's solution
   directly, then object-relative offset transfer, then action-count multiplier.

### Revised plan (3 weeks to June 30 milestone)

1. **Pivot from CNN-first to frugal Go-Explore explorer + training-free effect model** (Ash v21
   architecture, which is also our original Step 3 done right). De-prioritize Step 4 CNN.
2. Port the best ideas: frozen UI-mask hashing, prioritized candidate clicks snapped to connected
   components, return-then-explore, death attribution, cross-level effect-model persistence,
   cross-level solution transfer.
3. **RHAE budget governor:** per-level action cap (~5x estimated human baseline), per-game time
   budget, early bail in `is_done`.
4. Build local RHAE harness over all 25 public games; iterate locally, submit every 2-3 days.
5. **Open-source the solution before June 30** (CC0/MIT-0) for milestone eligibility.

## Scoring: RHAE (Relative Human Action Efficiency)

```
Level Score = min(1.0, human_baseline_actions / AI_actions)²
Environment Score = weighted average (later levels weigh more)
Total Score = mean of all environment scores
```

- **5x cutoff:** If AI takes >5x human actions on any level → score 0 for that level
- **To get 20% per level:** Need AI_actions ≤ 2.24x human actions
- **Squared penalty:** 2x actions = 25% score; 3x actions = 11%; 5x = 4%
- **Human baseline:** Second-best of 10 testers on first exposure

## Architecture: Hybrid Graph Explorer + CNN Agent

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Controller Loop                         │
│                                                                   │
│  ┌────────────┐   ┌──────────────┐   ┌─────────────────────┐   │
│  │  Frame      │──▶│ State Graph  │──▶│  Action Selector    │   │
│  │  Parser     │   │ Explorer     │   │ (CNN-guided +       │   │
│  │ (objects,   │   │ (nodes=frames│   │  priority tiers +   │   │
│  │  segments,  │   │  edges=acts) │   │  graph frontier)    │   │
│  │  status bar)│   │              │   │                      │   │
│  └────────────┘   └──────────────┘   └─────────────────────┘   │
│        ▲                                        │                │
│        └────────────────────────────────────────┘                │
│              observe result, train CNN online                     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  CNN: Predict which (action, coord) changes the frame   │     │
│  │  Train on binary labels from exploration observations   │     │
│  │  Re-rank action priority queue with CNN predictions     │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Approach Works (Offline, No API)

1. **Graph exploration as backbone** — Systematic state tracking, undo-based backtracking, no wasted actions
2. **CNN learns online** — Trains during gameplay to predict frame-changing actions (no pre-training)
3. **Priority-based action selection** — 5 tiers based on visual salience (objects > edges > background)
4. **Budget-aware** — Bail early on hopeless games, focus budget on solvable ones
5. **T4 GPU sufficient** — Small CNN (4 layers, ~1M params) trains in milliseconds per batch

### Key Components

1. **Frame Parser** — Parse 64x64 grid into connected components, detect status bar, mask UI elements
2. **State Graph** — Directed graph of unique frames (hashed). Edges = actions. Track tested/untested per state
3. **Priority System** — 5-tier action ranking based on object morphology, color salience, change regions
4. **CNN Action Predictor** — 4-layer CNN (32→64→128→256), two heads: action type + click coordinates
5. **Online Training Loop** — Binary cross-entropy on (state, action) → frame_changed labels
6. **Budget Manager** — Allocate actions per game/level, fallback to L0-only if burning too fast

### Action Space

| Action | GameAction | Meaning |
|--------|-----------|---------|
| Up | ACTION1 | Move selected object up |
| Down | ACTION2 | Move selected object down |
| Left | ACTION3 | Move selected object left |
| Right | ACTION4 | Move selected object right |
| Special | ACTION5 | Rotate/cycle/confirm (game-dependent) |
| Click(x,y) | ACTION6 | Select object at coordinates |
| Undo | ACTION7 | Revert to previous state |

## Stack

- **Language:** Python 3.11+
- **ML:** PyTorch (CNN online learning, no pre-training)
- **Toolkit:** `pip install arc-agi` (official SDK with local engine)
- **Submission:** Kaggle notebook (T4 GPU, **NO internet** during competition rerun)
- **Testing:** pytest + RHAE evaluation harness
- **Compute:** T4 GPU (16GB VRAM) for CNN training during gameplay

## Submission Format

- Kaggle notebook submission
- **Internet is NOT allowed** during competition rerun — all approaches must be fully offline
- Agent connects to gateway server at `http://gateway:8001` using official `ARC-AGI-3-Agents` framework
- Must output `submission.parquet` during normal run (dummy file); gateway handles scoring during competition rerun
- Uses official `Agent` base class with `choose_action()`/`is_done()` interface

## Build Steps

### Step 1: SDK Setup & Basic Agent Loop
- [ ] `pip install arc-agi` and verify toolkit works
- [ ] Register API key at https://three.arcprize.org
- [ ] Write minimal agent that plays a game with random actions
- [ ] Understand the `Arcade.make()` → `env.step(action)` → frame loop
- [ ] Capture frames and log state transitions
- [ ] Verify: random agent runs cn04, captures frames, hits level completion/failure

### Step 2: Frame Parsing & State Representation
- [ ] Build frame parser: 64x64 grid → connected components (objects)
- [ ] Detect objects: color, bounding box, pixel mask, position
- [ ] Track frame diffs: what changed between frames (moved, appeared, color changed)
- [ ] Build state hash for graph deduplication (detect cycles/loops)
- [ ] Detect UI elements (step counter bar) and mask them from game state
- [ ] Verify: parse cn04 frames, correctly identify all sprites and their positions

### Step 3: Advanced Graph Explorer (Week 1 Priority)
- [ ] Upgrade frame parser: connected component segmentation, status bar detection/masking
- [ ] Implement 5-tier action priority system based on visual salience (objects > edges > background)
- [ ] Add shortest-path navigation to nearest untested state-action pair in the graph
- [ ] Proper undo-based backtracking with path memory
- [ ] Dead end / loop detection — avoid revisiting exhausted states
- [ ] Study & adapt from `github.com/dolphin-in-a-coma/arc-agi-3-just-explore` (3rd place)
- [ ] Verify: graph explorer solves cn04 Level 1 within budget, ~5-10% RHAE on public games

### Step 4: CNN Online Learning (Week 2 Priority)
- [ ] Implement 4-layer CNN (32→64→128→256, 16-channel one-hot input) with two heads:
      - Action head: predicts P(frame_changes | action_type) for ACTION1-5
      - Coordinate head: convolutional, predicts P(frame_changes | click_x, click_y) for ACTION6
- [ ] Online training loop: binary cross-entropy on (state, action) → frame_changed labels
- [ ] Experience buffer with hash-based deduplication (~200K capacity)
- [ ] Hierarchical action sampling: sigmoid probabilities, action type first then coordinates
- [ ] Model reset on level change (new level = new mechanics)
- [ ] Entropy regularization to prevent premature convergence
- [ ] Study & adapt from `github.com/DriesSmit/ARC3-solution` (1st place StochasticGoose)
- [ ] Verify: CNN agent achieves ~10-15% RHAE on public games

### Step 5: Hybrid Integration (Explorer + CNN)
- [ ] Use graph explorer for systematic state tracking + CNN for action prioritization
- [ ] CNN predictions re-rank the priority queue within graph explorer
- [ ] Exploration phase: ~100-200 actions with graph explorer to gather training data
- [ ] Exploitation phase: CNN-guided action selection once model confidence is high
- [ ] Adaptive switching: monitor CNN prediction accuracy, revert to pure exploration if poor
- [ ] Verify: hybrid agent outperforms both standalone approaches

### Step 6: Scoring Meta-Strategy & Budget Management
- [ ] L0-only fallback: bail after 40 actions on games where exploration is failing
- [ ] Per-game action budget caps (tune based on baseline_actions)
- [ ] Breadth vs depth: attempt all games with small budget before deep-diving any
- [ ] Later levels weigh more — allocate more budget to later levels if early ones solve
- [ ] Track RHAE per level in real-time, abandon games with zero progress
- [ ] Verify: meta-strategy improves overall score vs naive full-budget-per-game

### Step 7: Multi-Game Tuning & Submission
- [ ] Run agent on all 25 public games, profile failure modes per game type
- [ ] Tune CNN hyperparameters (learning rate, train frequency, buffer size)
- [ ] Tune exploration hyperparameters (priority weights, undo threshold, budget split)
- [ ] Game-type detection: click-only vs keyboard-only vs hybrid → adjust strategy
- [ ] Multiple Kaggle submissions to iterate on leaderboard score
- [ ] Verify: agent achieves >10% RHAE on leaderboard

### Step 8: Kaggle Submission
- [x] Set up Kaggle notebook environment
- [x] Package agent for notebook execution (official Agent base class, gateway integration)
- [x] First submission to leaderboard (0.08% RHAE, rank ~800)
- [ ] Submit upgraded agent with CNN + graph explorer
- [ ] Iterate: identify weak games, tune per-game strategies
- [ ] Verify: leaderboard score >10% RHAE

### Step 9: Polish & Portfolio
- [ ] Write technical README with architecture, results, approach
- [ ] Create GitHub repo with clean structure
- [ ] Record demo showing agent solving games
- [ ] Write blog post explaining the approach (Medium or personal site)
- [ ] Add to resume/portfolio targeting Uber/Google/Meta

---

## Progress Tracker

| Step | Name | Status | Started | Completed | Notes |
|------|------|--------|---------|-----------|-------|
| 1 | SDK Setup & Basic Agent Loop | COMPLETE | 2026-06-07 | 2026-06-07 | arc-agi 0.9.8, GameRunner, ExplorerAgent, RHAE scoring |
| 2 | Frame Parsing & State Representation | COMPLETE | 2026-06-07 | 2026-06-07 | FrameParser: objects, diffs, movement tracking. Tested on 10 games |
| 3 | Advanced Graph Explorer | IN PROGRESS | 2026-06-09 | — | FrugalExplorer v2: effect model, frozen mask, frontier BFS w/ drift cooldown, death attribution, per-color segmentation, tiered click probing. 0.111% local (seeded/reproducible), 4 L0s solving — efficiency is the bottleneck now |
| 4 | CNN Online Learning | ABANDONED | 2026-06-09 | 2026-06-10 | Kernel v6 scored **0.05%** — worse than naive explorer (0.08%). Confirms research: CNN burns 200-300 actions/level learning, blows the 5x cutoff. Pivoted to training-free effect model |
| 5 | Hybrid Integration | RESCOPED | — | — | New scope: explorer + effect model (done in Step 3); CNN dropped |
| 6 | Scoring Meta-Strategy | NOT STARTED | — | — | Budget management, exploit-on-first-success, efficiency tuning |
| 7 | Multi-Game Tuning | NOT STARTED | — | — | Per-game profiling and tuning |
| 8 | Kaggle Submission | IN PROGRESS | 2026-06-07 | — | Best: 0.08% (v3 explorer, rank ~800). v6 CNN: 0.05%. Next submission: FrugalExplorer once local > 0.3%. Milestone #1: June 30 |
| 9 | Polish & Portfolio | IN PROGRESS | 2026-06-09 | — | GitHub repo live (private): github.com/shloksah/arc-agi-3-agent with README. Flip public + add MIT-0 license before June 30 for milestone eligibility |

**Overall:** 2/9 steps complete | **Current:** 0.08% Kaggle / 0.111% local | **Target:** 0.65%+ (top 5)

---

## Development Log

### 2026-06-10
- **v6 CNN scored 0.05%** — worse than the v3 explorer (0.08%). CNN path confirmed dead; Step 4 closed.
- Git history rewritten to sole authorship (Shlok Sah); commit routine updated.

### 2026-06-09 (architecture pivot day)
- Researched live leaderboard via Kaggle CLI + analyzed 4 top public notebooks (code in `/tmp/arc3-research/`): top = 1.21%, top-5 ≈ 0.65%, best public notebooks 0.35-0.46%. Engine-introspection BFS identified as DQ risk; frugal Go-Explore explorer identified as the legitimate meta.
- **Built `harness.py`**: 25-game local RHAE sweep (~50s, seeded, reproducible), results tracked in `runs/`.
- **Fixed `compute_rhae`**: official weighting (all levels in denominator) + 5x cutoff.
- **GameRunner death-continuation**: GAME_OVER → death attribution → RESET → play on (was terminal; bp35 used to die at action 16).
- **Built FrugalExplorer** (`core/frugal_explorer.py` + `core/effect_model.py`): frozen border-band UI mask hashing, component-snapped clicks with tiered expansion, training-free effect model with exploration bonus, frontier BFS (return-then-explore) with drift cooldown, death attribution.
- **Found root-cause parser bug**: bg-vs-rest segmentation merged boards into one giant component, hiding embedded interactive elements (lp85: 600 clicks, 0 changes). Fixed with per-color connected components → lp85 L0 solves within cutoff.
- **Local scores**: random 0.000% / old explorer 0.236% (2 lucky solves, high variance) / FrugalExplorer 0.111% deterministic with 4 L0s solving (lp85 in-cutoff; lf52/tn36/vc33 past 5x = zero credit).
- **Key insight**: solving is no longer the bottleneck — action efficiency is. Getting the 3 zeroed solves under the cutoff ≈ triples local score; near 2x baseline ≈ 0.5%+ (top-5 range).
- **Diagnosed, parked**: m0r0 = genuinely large state space (mirror-matching game, not a hash bug); r11l = click tunnel-vision (every click changes pixels, none progress).
- **Next**: exploit-on-first-success (concentrate on same-color/region candidates once a click works), cross-level candidate pruning via effect model, then submit when local > 0.3%.

---

## Final CNN Submission (kernel v6 — closed at 0.05%)

- **Notebook:** `submission.ipynb` — adapted from official StochasticGoose sample (`kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose`)
- **Config:** T4 GPU enabled, internet disabled, competition data source linked
- **Agent:** `MyAgent(Agent)` — 4-layer CNN that learns online which actions change the frame
- **Time budget:** Bails at 8 hours (9-hour Kaggle limit, 1-hour buffer)
- **Pattern:** During competition rerun, connects to gateway at `http://gateway:8001`; during normal run, writes dummy `submission.parquet`

### How the 8-hour run works (per game)
1. Pick a game, observe the 64×64 frame
2. Try an action (random at first — CNN is untrained)
3. Compare new frame to old → label `(state, action) → changed (1) / unchanged (0)`
4. Every 5 actions, train the CNN on a batch of these labels
5. After ~200-300 actions, CNN predicts useful actions → shifts from exploration to exploitation
6. On level-up: **CNN fully resets** (fresh weights + empty buffer), relearns from scratch
7. Repeat across all games

### Known improvement opportunities (not yet implemented)
1. **Knowledge transfer across levels** — Currently CNN fully resets each level (wasteful; ~200-300 actions relearning basics). Options:
   - Keep CNN weights, only clear buffer (warm start; adapts if mechanics change)
   - Carry forward a fraction of high-confidence experiences
   - Freeze conv backbone (visual features transfer), reset only action/coord heads
2. **Graph explorer integration** (Step 5) — add systematic state tracking + undo backtracking on top of CNN
3. **Scoring meta-strategy** (Step 6) — L0-only fallback, per-game budget caps for breadth
4. **Caveat:** StochasticGoose scored 12.58% in preview but ~0.25% on full benchmark at launch — preview games were easier. Don't assume 12% transfers directly.

---

## Competitive Analysis: Realistic Target 10%+ RHAE

### What 10% Means Concretely
- Solve tutorial levels (L0) on most games + several deeper levels
- For solved levels, stay within ~3x human action count
- Later levels weigh more — even nailing early levels helps
- Breadth matters: score = mean across ALL environments, so attempting more games > perfecting few

### Our Advantages
1. **Graph + CNN hybrid** — Combines proven approaches (3rd place + 1st place preview)
2. **No pre-training needed** — CNN trains online during gameplay
3. **Budget-aware strategy** — L0-only fallback maximizes breadth
4. **Open-source references** — Both StochasticGoose and graph explorer code are available

### Our Risks
1. **CNN generalization** — Model resets per level; may not learn fast enough on hard games
2. **Novel games** — Private test set has unseen games; heuristics must generalize
3. **Time budget** — 9 hours for all games; CNN training adds overhead per game
4. **Exploration ceiling** — Without a world model, can't plan optimal paths

### Mitigation Strategy
- Graph explorer provides 5-10% floor even without CNN
- CNN layered on top for exploitation gains
- L0-only fallback for games where both approaches struggle
- Tune on all 25 public games before submitting

---

## Key Insights from Research

### From StochasticGoose (12.58% RHAE, 1st Place Preview)
- CNN architecture: 4 layers (32→64→128→256 channels), 16-channel one-hot input
- Two heads: action type (5 classes) + coordinate (64×64 spatial, convolutional)
- Learns which actions change frame state (binary classification)
- Hierarchical action sampling: sigmoid probabilities, action type first then coordinates
- Hash tables for state-action deduplication (~200K experience buffer)
- Model reset + buffer clear on level change
- Entropy regularization prevents premature convergence
- ~350 initial inefficient moves, then learns and shifts to exploitation
- **Caveat:** Score dropped from 12.58% (preview) to ~0.25% on full benchmark at launch
- **Code:** `github.com/DriesSmit/ARC3-solution`

### From "Explore It Till You Solve It" (3rd Place Preview, ~10% RHAE)
- Training-free, pure graph-based exploration — no neural networks
- Two-stage pipeline: Frame Processor → Level Graph Explorer
- Frame Processor: connected component segmentation, status bar detection, priority grouping, hashing
- 5-tier action priority based on visual salience (objects > edges > background)
- Shortest-path navigation to nearest state with untested actions at highest priority
- Undo-based backtracking for efficient graph building
- Solved median 30/52 levels across 5 runs; official: 12 levels (3rd place)
- Bug fix post-competition improved to 16 private / 14 public levels
- **Code:** `github.com/dolphin-in-a-coma/arc-agi-3-just-explore`
- **Paper:** arxiv.org/abs/2512.24156

### From Blind Squirrel (2nd Place Preview, 6.71% RHAE)
- Graph-based state exploration + learned value model
- Builds state graph, prunes actions creating loops or no state change
- On score improvement: back-labels level with distances, retrains ResNet18-based value model
- More efficient action usage than StochasticGoose (fewer actions per level) but solved fewer levels
- No public code available

### From Executable World Models Paper (32.58% RHAE — Requires LLM API)
- Agent writes 3 Python files: `world_model_engine.py`, `world_model_state_io.py`, `world_model_main_planner.py`
- Verification is critical — test model against observed transitions before trusting it
- Refactoring toward simplicity prevents overfitting to early observations
- Weakness: "tunnel vision" — agent locks into initial hypothesis too early
- **Cannot run on Kaggle** — requires GPT-5.4 API calls ($34-$620 per game)
- Could theoretically work with local quantized 7B model on T4, but code quality would be far below

### From Sensi (Curriculum-Based Test-Time Learning)
- Two-player architecture separating perception from action
- Curriculum learning via external state machine
- Database-as-control-plane makes context window steerable
- LLM-as-judge with dynamic rubrics
- 50-94x better sample efficiency, but solves only 2 game levels
- **Cannot run on Kaggle** — requires LLM API calls

### Scoring Meta-Strategies
- **L0-only strategy:** Only attempt Level 0 (tutorial) of each game, bail after ~40 actions. Trade per-game depth for breadth across more games. Since score = mean across ALL environments, breadth wins.
- **RHAE exploitation:** Squared penalty means >5x human actions ≈ 0% credit. No point burning budget past that threshold.
- **Later levels weigh more:** Level 5 = 5/15 weight vs Level 1 = 1/15. Mastering later levels matters more, but failing them costs more too.
- **Source:** Kaggle discussion thread on scoring strategy

### From Competition Rules
- 135 total environments (25 public, 55 semi-private, 55 private)
- Leaderboard uses 50% of test data; final uses other 50%
- Must open-source for prize eligibility (CC0 or MIT-0 license)
- 6+ levels per game, level 1 is tutorial
- Human median: 8.1 minutes per successful attempt
- **No internet during competition rerun** — all approaches must be offline
- 9-hour time limit for notebook execution
- Gateway server at `http://gateway:8001` during competition rerun

---

## Key Insights from cn04 (First Downloaded Task)

- **Game type:** Sprite connection puzzle — connect colored markers by positioning/rotating sprites
- **Win condition:** All color-8 (blue) and color-13 (magenta) markers must overlap with another sprite's matching marker
- **Mechanics:** Select sprite → move/rotate → markers align → connectors turn green (color 3)
- **Difficulty scaling:** More sprites, grey masking (can't see unselected sprites' colors), stacked sprites (cycle with rotate)
- **Step budgets:** 75 → 100 → 125 → 125 → 150 → 200 (increasing per level)
- **Baseline actions:** [29, 54, 85, 300, 208, 113] — human performance per level

## Design Decisions

1. **Hybrid graph + CNN** — Graph exploration as floor (~5-10%), CNN as exploitation layer (targeting 10-15%)
2. **Fully offline** — No API calls; CNN trains online during gameplay on T4 GPU
3. **Breadth over depth** — Attempt all games with budget caps; L0-only fallback for hard games
4. **Model reset per level** — Each level may have new mechanics; fresh CNN avoids stale predictions
5. **Budget-aware** — Track remaining actions; switch from exploration to exploitation based on budget
6. **Build on proven code** — Adapt StochasticGoose CNN + graph explorer rather than building from scratch

## Open Source Resources

| Resource | URL | What It Provides |
|----------|-----|-----------------|
| StochasticGoose | `github.com/DriesSmit/ARC3-solution` | Full CNN+RL agent (1st place) |
| Graph Explorer | `github.com/dolphin-in-a-coma/arc-agi-3-just-explore` | Full graph agent (3rd place) |
| Official Starter Agents | `github.com/arcprize/ARC-AGI-3-Agents` | Agent base class, gateway integration |
| Kaggle CNN notebook | `kaggle.com/code/gourabr0y555/arc-agi-3-stochastic-cnn-exploration-agent` | Ready-to-submit CNN agent |
| Kaggle Graph+Value notebook | `kaggle.com/code/nihilisticneuralnet/arc-agi-3-graph-exploration-w-value-learning` | Graph + value learning |
| Official sample submission | `kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose` | StochasticGoose for Kaggle |

## References

- [Executable World Models for ARC-AGI-3](https://arxiv.org/abs/2605.05138) — Rodionov, 32.58% RHAE (requires API)
- [Graph-Based Exploration for ARC-AGI-3](https://arxiv.org/abs/2512.24156) — 3rd place preview, ~10% RHAE
- [StochasticGoose (1st place preview)](https://github.com/DriesSmit/ARC3-solution) — CNN+RL, 12.58%
- [Sensi: Curriculum Test-Time Learning](https://arxiv.org/abs/2603.17683) — Perception/action separation
- [ARC-AGI-3 Technical Report](https://arxiv.org/abs/2603.24621) — Official benchmark paper
- [ARC-AGI Toolkit Docs](https://docs.arcprize.org/toolkit/overview) — SDK reference
- [ARC-AGI-3 Agents Repo](https://github.com/arcprize/ARC-AGI-3-Agents) — Official starter agents
- [ARC Prize Blog: 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) — Preview competition analysis
- [StochasticGoose Writeup](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db) — 1st place methodology
