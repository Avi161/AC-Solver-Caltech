# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AC-Solver is a research codebase for the paper "What Makes Math Problems Hard for Reinforcement Learning: A Case Study." It implements the Andrews-Curtis (AC) Conjecture as an RL environment and combines classical search algorithms with learned value functions to solve AC instances.

**Key Components:**
1. **AC Environment** (`ac_solver/envs/`) - Gymnasium-based environment for balanced group presentations
2. **Search Algorithms** (`ac_solver/search/`) - Greedy, BFS, and Miller-Schupp solvers
3. **RL Agent** (`ac_solver/agents/`) - PPO agent with optional RND curiosity module
4. **Value-Guided Search** (`value_search/`) - ML-trained networks to guide classical search
5. **Experiments** (`experiments/`) - Configuration-driven experiment orchestration

## Development Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode (already done, but for reference)
pip install -e .
```

**Python Version:** 3.9+ (currently using 3.9, configured in `.python-version`)

**Key Dependencies:**
- `torch` - Neural networks for agent and value functions
- `gymnasium` - RL environment API
- `wandb` - Experiment tracking and logging
- `pyyaml` - Experiment configuration
- `numpy, scipy, scikit-learn` - Data processing
- `sympy` - Mathematical computations (group theory)

**Development Tools:**
- `pytest` - Testing framework
- `ruff` - Linting
- `black` - Code formatting
- `poetry` - Dependency management

## Common Commands

### Training and Experiments

```bash
# Train PPO agent with default settings
python ac_solver/agents/ppo.py

# Train PPO with custom hyperparameters (see args.py for full list)
python ac_solver/agents/ppo.py --num-envs 4 --learning-rate 0.0003 --num-steps 128

# Run experiment suite (benchmarks with value-guided search)
python experiments/run_experiments.py

# Run experiments with custom config
python experiments/run_experiments.py --config experiments/config.yaml --max-nodes 10000

# Run iterative refinement loop
python experiments/iterative_refinement.py
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/envs/test_ac_env.py -v

# Run specific test function
pytest tests/envs/test_ac_env.py::test_trivial_presentation -v

# Run with coverage
pytest --cov=ac_solver tests/
```

### Code Quality

```bash
# Lint code (check without fixing)
ruff check ac_solver/ tests/

# Format code
black ac_solver/ tests/ experiments/ value_search/

# Lint a specific file
ruff check ac_solver/agents/ppo.py
```

## Architecture & Data Flow

### Core Training Loop (ac_solver/agents/ppo.py)

1. **Environment Setup** (`environment.py:get_env()`)
   - Creates parallel Gymnasium environments
   - Loads initial states from Miller-Schupp dataset or custom presentations
   - Tracks success metrics and history

2. **PPO Training** (`training.py:ppo_training_loop()`)
   - Collects experience from parallel environments
   - Computes advantages and policy gradient updates
   - Optional curiosity reward (RND module) for exploration
   - Logs to Weights & Biases

3. **Agent Architecture** (`ppo_agent.py`)
   - Policy head: outputs action logits
   - Value head: estimates state value
   - Observation processing: handles flattened presentation state

### Search Algorithms (ac_solver/search/)

**Greedy Search** (`greedy.py`)
- Best-first search using heuristic (reduction count)
- Fast, single path exploration
- Time limit support

**BFS** (`breadth_first.py`)
- Exhaustive exploration up to node limit
- Returns first solution found
- Configurable max nodes

**Miller-Schupp** (`miller_schupp/`)
- Dataset of ~1200 balanced presentations
- Presentation-to-action encoding utilities

### Value-Guided Search (value_search/)

**Training** (`train_value_net.py`)
- Trains MLP or Transformer networks to predict solution feasibility
- Features extracted from presentation state
- Supervised learning with solved/unsolved labels

**Search Integration** (`value_guided_search.py`)
- Value-guided greedy: uses learned scores for action selection
- Beam search: maintains K best candidates
- Solution caching for efficiency

### Experiments Framework (experiments/)

**Configuration** (`config.yaml`)
- Defines search algorithms to benchmark
- Presentation dataset and search parameters
- Time limits and node budgets

**Orchestration** (`run_experiments.py`)
- Runs algorithms in sequence
- Saves results incrementally (failures tolerated)
- Produces JSON results with solve rates and timing

**Refinement** (`iterative_refinement.py`)
- Iterative loop: train value net → search with it → collect labels
- Multi-iteration training for performance gains
- Tracks state in `experiments/refinement/`

## Key Files & Patterns

### Configuration Management

```python
# AC Environment config
from ac_solver.envs import ACEnvConfig
config = ACEnvConfig(
    initial_state=[1, 0, 2, 0],  # or list/ndarray
    horizon_length=1000,
    use_supermoves=False
)

# Training config
# See ac_solver/agents/args.py for all arguments (passed via CLI or config)
```

### Environment State Representation

- **Presentation:** flat array of integers representing group relations
  - Indices: [rel1_gen1, rel1_gen2, ..., relN_genN, 0, 0]
  - Generators: 1 = x, 2 = y, -1 = x^-1, -2 = y^-1
  - Zeros mark end of relators
- **Trivial state:** [1, 0, 2, 0] (empty presentation)

### Logging & Checkpoints

- **PPO runs:** Logged to Weights & Biases (wandb)
  - Controlled via `--use-wandb` flag
  - Track loss, rewards, success metrics

- **Value networks:** Checkpoints in `value_search/checkpoints/`
  - `history_seq.json`, `history_mlp.json` - training metrics
  - Model weights saved after training

- **Experiments:** Results in `experiments/results/<timestamp>/`
  - `config_used.yaml` - exact config for reproducibility
  - `results.json` - solve rates and timing
  - Algorithm details for debugging

## Testing Strategy

**Key test modules:**
- `tests/envs/test_ac_env.py` - Environment mechanics
- `tests/search/test_bfs.py`, `test_gs.py` - Search algorithm correctness
- `tests/agents/test_ppo.py` - Agent initialization and training
- `tests/test_solution_verification.py` - Path validation

**Running before commits:**
```bash
pytest tests/ -v --tb=short
```

## Experiment Workflows

### Standard Experiment Run

1. Review config in `experiments/config.yaml`
2. Run experiment suite: `python experiments/run_experiments.py`
3. Results saved to timestamped directory with JSON summary
4. Use notebooks in `notebooks/` for analysis

### Training a Value Network

1. Collect solved/unsolved examples (via search)
2. Run: `python value_search/train_value_net.py`
3. Model saved to `value_search/checkpoints/`
4. Use in search via `value_guided_search.py`

### Iterative Refinement

1. Initialize in `experiments/config.yaml`
2. Run: `python experiments/iterative_refinement.py`
3. State persisted in `experiments/refinement/`
4. Can resume runs

## Important Context

- **AC Conjecture:** Decades-old open problem in group theory; this repo tackles it as an RL benchmark
- **Sparse Rewards:** Agent only gets signal when reaching trivial state (success) or timeout
- **Horizon Control:** Easy to vary difficulty by adjusting max steps
- **Paper Reproducibility:** See `notebooks/` and W&B for published results
- **Computational Efficiency:** AC solver extremely fast (~ms per step) compared to typical RL domains
