"""
ppo_training_loop function implements the training loop logic of PPO.
"""

import math
import random
import uuid
import json
import time
import datetime
import wandb
from collections import deque
from tqdm import tqdm
from os import makedirs
from os.path import join
import numpy as np
import torch
from torch import nn

from ac_solver.envs.ac_moves import ACMove
from ac_solver.envs.utils import is_presentation_trivial


def get_curr_lr(n_update, lr_decay, warmup, max_lr, min_lr, total_updates):
    """
    Calculates the current learning rate based on the update step, learning rate decay schedule,
    warmup period, and other parameters.

    Parameters:
    n_update (int): The current update step (1-indexed).
    lr_decay (str): The type of learning rate decay to apply ("linear" or "cosine").
    warmup (float): The fraction of total updates to be used for the learning rate warmup.
    max_lr (float): The maximum learning rate.
    min_lr (float): The minimum learning rate.
    total_updates (int): The total number of updates.

    Returns:
    float: The current learning rate.

    Raises:
    NotImplementedError: If an unsupported lr_decay type is provided.
    """
    # Convert to 0-indexed for internal calculations
    n_update -= 1
    total_updates -= 1

    # Calculate the end of the warmup period
    warmup_period_end = total_updates * warmup

    if warmup_period_end > 0 and n_update <= warmup_period_end:
        lrnow = max_lr * n_update / warmup_period_end
    else:
        if lr_decay == "linear":
            slope = (max_lr - min_lr) / (warmup_period_end - total_updates)
            intercept = max_lr - slope * warmup_period_end
            lrnow = slope * n_update + intercept

        elif lr_decay == "cosine":
            cosine_arg = (
                (n_update - warmup_period_end)
                / (total_updates - warmup_period_end)
                * math.pi
            )
            lrnow = min_lr + (max_lr - min_lr) * (1 + math.cos(cosine_arg)) / 2

        else:
            raise NotImplementedError(
                "Only 'linear' and 'cosine' lr-schedules are available."
            )

    return lrnow


def replay_final_state(action_list, initial_state):
    """Replay actions and return the final state array for triviality verification."""
    state = np.array(initial_state, dtype=np.int8)
    max_relator_length = len(state) // 2
    lengths = [
        int(np.count_nonzero(state[i * max_relator_length : (i + 1) * max_relator_length]))
        for i in range(2)
    ]
    for action in action_list:
        state, lengths = ACMove(
            move_id=int(action),
            presentation=state,
            max_relator_length=max_relator_length,
            lengths=lengths,
        )
    return state


def actions_to_path(action_list, initial_state):
    """
    Replay a list of 0-indexed actions on an initial state to produce
    the experiment-compatible path format: [[action_0idx, total_length_after], ...].

    This format matches what run_experiments.py and replay_path.py expect,
    enabling step-by-step verification of PPO-discovered solutions.

    Also returns a detailed_path that records the full relator states after
    each move (including the effects of free and cyclic reductions), so that
    every intermediate presentation can be retraced.

    Returns:
        (path, detailed_path) where
        - path: [[action_id, total_length], ...]
        - detailed_path: list of dicts with keys:
            "action": int, "r0": list[int], "r1": list[int],
            "lengths": [int, int]
    """
    state = np.array(initial_state, dtype=np.int8)
    max_relator_length = len(state) // 2
    lengths = [
        int(np.count_nonzero(state[i * max_relator_length : (i + 1) * max_relator_length]))
        for i in range(2)
    ]
    path = []
    detailed_path = []
    for action in action_list:
        state, lengths = ACMove(
            move_id=int(action),
            presentation=state,
            max_relator_length=max_relator_length,
            lengths=lengths,
        )
        path.append([int(action), int(sum(lengths))])
        r0 = [int(x) for x in state[:max_relator_length] if x != 0]
        r1 = [int(x) for x in state[max_relator_length:] if x != 0]
        detailed_path.append({
            "action": int(action),
            "r0": r0,
            "r1": r1,
            "lengths": [int(lengths[0]), int(lengths[1])],
        })
    return path, detailed_path


def ppo_training_loop(
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
    rnd_module=None,
):
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()[0]).to(device)  # get first observation
    next_done = torch.zeros(args.num_envs).to(device)  # get first done
    num_updates = args.total_timesteps // args.batch_size
    episodic_return = np.array([0] * args.num_envs)
    episodic_length = np.array([0] * args.num_envs)
    episode = 0
    returns_queue = deque([0], maxlen=100)
    lengths_queue = deque([0], maxlen=100)
    round1_complete = False  # whether we have already chosen each element of initial_states at least once to initiate rollout
    beta = None if args.is_loss_clip else args.beta

    run_name = f"{args.exp_name}_ppo-ffn-nodes_{args.nodes_counts}_{uuid.uuid4()}"
    out_dir = f"out/{run_name}"
    makedirs(out_dir, exist_ok=True)

    # Set up experiment results directory (compatible with run_experiments.py / replay_path.py)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    rnd_tag = "_rnd" if rnd_module is not None else ""
    results_dir = join("experiments", "results", f"{timestamp}_ppo{rnd_tag}")
    makedirs(results_dir, exist_ok=True)
    progress_path = join(results_dir, f"ppo{rnd_tag}_progress.jsonl")
    progress_fh = open(progress_path, "w")
    # Track best paths per presentation index for final details file
    best_paths = {}  # idx -> {"path": [...], "path_length": int, "time": float}
    training_start_time = time.time()
    print(f"Results will be saved to: {results_dir}/")

    if args.wandb_log:
        run = wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    print(f"total number of timesteps: {args.total_timesteps}, updates: {num_updates}")
    for update in tqdm(
        range(1, num_updates + 1), desc="Training Progress", total=num_updates
    ):

        # using different seed for each update to ensure reproducibility of paused-and-resumed runs
        random.seed(args.seed + update)
        np.random.seed(args.seed + update)
        torch.manual_seed(args.seed + update)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            lrnow = get_curr_lr(
                n_update=update,
                lr_decay=args.lr_decay,
                warmup=args.warmup_period,
                max_lr=args.learning_rate,
                min_lr=args.learning_rate * args.min_lr_frac,
                total_updates=num_updates,
            )
            optimizer.param_groups[0]["lr"] = lrnow

        # collecting and recording data
        for step in tqdm(
            range(0, args.num_steps), desc=f"Rollout Phase - {update}", leave=False
        ):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done  # contains 1 if done else 0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs
                )  # shapes: n_envs, n_envs, n_envs, (n_envs, 1)
                values[step] = value.flatten()  # num_envs
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, infos = envs.step(
                action.cpu().numpy()
            )  # step is taken on cpu
            extrinsic_reward = torch.tensor(reward).to(device).view(-1)

            # Compute and add intrinsic curiosity reward (RND)
            if rnd_module is not None:
                intrinsic_reward = rnd_module.compute_intrinsic_reward(
                    torch.Tensor(next_obs).to(device)
                )
                intrinsic_reward_tensor = (
                    torch.tensor(intrinsic_reward, dtype=torch.float32).to(device)
                )
                rewards[step] = extrinsic_reward + args.rnd_coef * intrinsic_reward_tensor
            else:
                rewards[step] = extrinsic_reward

            episodic_return = episodic_return + reward
            episodic_length = episodic_length + 1

            _record_info = np.array(
                [
                    True if done[i] or truncated[i] else False
                    for i in range(args.num_envs)
                ]
            )
            if _record_info.any():
                for i, el in enumerate(_record_info):

                    if done[i]:
                        # if done, add curr_states[i] to 'solved' cases
                        if curr_states[i] in success_record["unsolved"]:
                            success_record["unsolved"].remove(curr_states[i])
                            success_record["solved"].add(curr_states[i])

                        # Extract action list from info (gymnasium >=1.0 format)
                        # SyncVectorEnv puts keys directly in infos dict with _<key> mask
                        if "final_info" in infos:
                            # gymnasium < 1.0
                            action_list = infos["final_info"][i]["actions"]
                        else:
                            # gymnasium >= 1.0
                            action_list = list(infos["actions"][i])

                        # Record the sequence of actions in ACMoves_hist
                        is_new_best = False
                        if curr_states[i] not in ACMoves_hist:
                            ACMoves_hist[curr_states[i]] = action_list
                            is_new_best = True
                        else:
                            prev_path_length = len(ACMoves_hist[curr_states[i]])
                            new_path_length = len(action_list)
                            if new_path_length < prev_path_length:
                                ACMoves_hist[curr_states[i]] = action_list
                                is_new_best = True

                        # Save solution in experiment-compatible format
                        if is_new_best:
                            idx = curr_states[i]
                            path_with_lengths, detailed = actions_to_path(
                                action_list, initial_states[idx]
                            )
                            # Verify path actually reaches trivial state
                            final_length = path_with_lengths[-1][1] if path_with_lengths else -1
                            final_state = replay_final_state(action_list, initial_states[idx])
                            if final_length == 2 and is_presentation_trivial(final_state):
                                elapsed = time.time() - training_start_time
                                record = {
                                    "idx": idx,
                                    "solved": True,
                                    "path_length": len(path_with_lengths),
                                    "time": elapsed,
                                    "global_step": global_step,
                                    "episode": episode,
                                    "path": path_with_lengths,
                                    "detailed_path": detailed,
                                }
                                progress_fh.write(json.dumps(record) + "\n")
                                progress_fh.flush()
                                best_paths[idx] = record

                    # record+reset episode data, reset ith initial state to the next state in init_states
                    if el:
                        # record and reset episode data
                        returns_queue.append(episodic_return[i])
                        lengths_queue.append(episodic_length[i])
                        episode += 1
                        episodic_return[i], episodic_length[i] = 0, 0

                        # update next_obs to have the next initial state
                        prev_state = curr_states[i]
                        round1_complete = (
                            True
                            if round1_complete
                            or (max(states_processed) == len(initial_states) - 1)
                            else False
                        )
                        if not round1_complete:
                            curr_states[i] = max(states_processed) + 1
                        else:
                            # TODO: If states-type=all, first choose from all solved presentations then choose from unsolved presentations
                            if len(success_record["solved"]) == 0 or (
                                success_record["unsolved"]
                                and random.uniform(0, 1) > args.repeat_solved_prob
                            ):
                                curr_states[i] = random.choice(
                                    list(success_record["unsolved"])
                                )
                            else:
                                curr_states[i] = random.choice(
                                    list(success_record["solved"])
                                )
                        states_processed.add(curr_states[i])
                        next_obs[i] = initial_states[curr_states[i]]
                        envs.envs[i].reset(options={"starting_state": next_obs[i]})

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

        if (
            not args.norm_rewards
        ):  # if not normalizing rewards through a NormalizeRewards Wrapper, rescale rewards manually.
            # Access max_reward from the unwrapped ACEnv (may be behind wrappers)
            base_env = envs.envs[0].unwrapped if hasattr(envs.envs[0], 'unwrapped') else envs.envs[0]
            rewards /= base_env.max_reward
            normalized_returns = np.array(returns_queue) / base_env.max_reward
            normalized_lengths = np.array(lengths_queue) / args.horizon_length
        else:
            normalized_returns = np.array(returns_queue)
            normalized_lengths = np.array(lengths_queue)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values  # Where do we use returns?

        # flattening out the data collected from parallel environments
        b_obs = obs.reshape(
            (-1,) + envs.single_observation_space.shape
        )  # num_envs * num_steps, obs_space.shape
        b_logprobs = logprobs.reshape(-1)  # num_envs * num_steps
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value networks
        b_inds = np.arange(args.batch_size)  # indices of batch_size
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )  # .long() converts dtype to int64
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = (
                    logratio.exp()
                )  # pi(a|s) / pi_old(a|s); is a tensor of 1s for epoch=0.

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                kl_var = (
                    ratio - 1
                ) - logratio  # the random variable whose expectation gives approx kl
                with torch.no_grad():
                    approx_kl = (
                        kl_var.mean()
                    )  # mean of (pi(a|s) / pi_old(a|s) - 1 - log(pi(a|s) / pi_old(a|s)))
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                if args.is_loss_clip:  # clip loss
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:  # KL-penalty loss
                    pg_loss2 = beta * kl_var
                    pg_loss = (pg_loss1 + pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(
                    -1
                )  # value computed by NN with updated parameters
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )  # can i implement this myself?
                optimizer.step()

            if args.is_loss_clip:  # if clip loss and approx_kl > target kl, break
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            else:  # if KL-penalty loss, update beta
                beta = (
                    beta / 2
                    if approx_kl < args.target_kl / 1.5
                    else (beta * 2 if approx_kl > args.target_kl * 1.5 else beta)
                )

        # Train RND predictor on the collected observations
        rnd_loss = 0.0
        if rnd_module is not None:
            rnd_loss = rnd_module.update(b_obs)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.wandb_log:
            wandb.log(
                {
                    "charts/global_step": global_step,
                    "charts/episode": episode,
                    "charts/normalized_returns_mean": normalized_returns.mean(),
                    "charts/normalized_lengths_mean": normalized_lengths.mean(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/solved": len(success_record["solved"]),
                    "charts/unsolved": len(success_record["unsolved"]),
                    "charts/highest_solved": (
                        max(success_record["solved"])
                        if success_record["solved"]
                        else -1
                    ),
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy_loss": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/explained_variance": explained_var,
                    "losses/clipfrac": np.mean(clipfracs),
                    "debug/advantages_mean": b_advantages.mean(),
                    "debug/advantages_std": b_advantages.std(),
                    **(
                        {
                            "rnd/predictor_loss": rnd_loss,
                            "rnd/reward_running_std": np.sqrt(
                                rnd_module.reward_rms.var + 1e-8
                            ),
                        }
                        if rnd_module is not None
                        else {}
                    ),
                }
            )

        if update > 0 and update % 100 == 0:  # save a checkpoint every 100 updates
            checkpoint = {
                "critic": agent.critic.state_dict(),
                "actor": agent.actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "episode": episode,
                "config": vars(args),
                "mean_return": normalized_returns.mean(),
                "success_record": success_record,
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "approx_kl": approx_kl.item(),
                "explained_var": explained_var,
                "clipfrac": np.mean(clipfracs),
                "global_step": global_step,
                "round1_complete": round1_complete,
                "curr_states": curr_states,
                "states_processed": states_processed,
                "ACMoves_hist": ACMoves_hist,
                "supermoves": getattr(envs.envs[0].unwrapped, "supermoves", None),
                **(
                    {"rnd_state": rnd_module.state_dict()}
                    if rnd_module is not None
                    else {}
                ),
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, join(out_dir, "ckpt.pt"))

    # Save final experiment results (compatible with replay_path.py verification)
    progress_fh.close()
    total_time = time.time() - training_start_time

    # Build details list: one entry per presentation (solved or not)
    algo_name = f"ppo{rnd_tag}"
    all_results = []
    for idx in range(len(initial_states)):
        if idx in best_paths:
            all_results.append(best_paths[idx])
        else:
            all_results.append({
                "idx": idx,
                "solved": False,
                "path_length": 0,
                "time": total_time,
            })

    details_path = join(results_dir, f"{algo_name}_details.json")
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save summary
    solved_indices = sorted(best_paths.keys())
    path_lengths = [best_paths[i]["path_length"] for i in solved_indices]
    summary = {
        "timestamp": timestamp,
        "algorithm": algo_name,
        "config": {k: str(v) for k, v in vars(args).items()},
        "total_time": total_time,
        "metrics": {
            "solved": len(solved_indices),
            "total": len(initial_states),
            "avg_path_length": float(np.mean(path_lengths)) if path_lengths else 0,
            "median_path_length": float(np.median(path_lengths)) if path_lengths else 0,
            "max_path_length": int(max(path_lengths)) if path_lengths else 0,
            "solved_indices": solved_indices,
        },
    }
    summary_path = join(results_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PPO Training Complete")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Solved: {len(solved_indices)}/{len(initial_states)}")
    if path_lengths:
        print(f"  Avg path length: {np.mean(path_lengths):.1f}")
    print(f"  Results: {results_dir}/")
    print(f"    {algo_name}_progress.jsonl  — streaming solutions")
    print(f"    {algo_name}_details.json    — all results (verify with replay_path.py)")
    print(f"    results.json               — summary metrics")
    print(f"\n  Verify solutions with:")
    print(f"    python value_search/replay_path.py -r {results_dir} -a {algo_name}")
    print(f"{'='*60}")

    return
