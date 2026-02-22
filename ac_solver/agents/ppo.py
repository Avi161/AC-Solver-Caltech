"""
This file trains a PPO (Proximal Policy Optimization) agent on AC Environment.
It sets up the training environment, initializes the agent, and runs the PPO training loop.
Run this script directly to start training the PPO agent, as simply as 

```
python ppo.py
```

To see the entire list of command line arguments you may pass, check args.py
"""

import numpy as np
import torch
import random
from torch.optim import Adam
from ac_solver.agents.ppo_agent import Agent
from ac_solver.agents.args import parse_args
from ac_solver.agents.environment import get_env
from ac_solver.agents.training import ppo_training_loop
from ac_solver.agents.rnd import RNDModule


def train_ppo():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    (
        envs,
        initial_states,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
    ) = get_env(args)

    agent = Agent(envs, args.nodes_counts).to(device)
    optimizer = Adam(agent.parameters(), lr=args.learning_rate, eps=args.epsilon)

    # Optionally create RND module for curiosity-driven exploration
    rnd_module = None
    if args.use_rnd:
        input_dim = np.prod(envs.single_observation_space.shape)
        rnd_module = RNDModule(
            input_dim=input_dim,
            hidden_dim=args.rnd_hidden_dim,
            output_dim=args.rnd_output_dim,
            learning_rate=args.rnd_lr,
            device=device,
        )
        print(f"RND exploration enabled: coef={args.rnd_coef}, "
              f"hidden={args.rnd_hidden_dim}, output={args.rnd_output_dim}")

    ppo_training_loop(
        envs,
        args,
        device,
        optimizer,
        agent,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
        rnd_module=rnd_module,
    )

    envs.close()


if __name__ == "__main__":
    train_ppo()
