"""rpo_continuous. Run 'tensorboard --logdir runs' to monitor the
process. Run render_rpo.py any moment after an autosave to make a
video of how it is doing. Run 'nvidia-smi' to see your CUDA device
use."""

# TODO: Resume still fails

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

from agent import Agent

ENV_ID = "dragoon"

SAVE = True
LOAD = False
LOAD_UPDATE = 488  # The number of the update to load
ANNEAL_LR = True
CHECKPOINT_PERIOD = 100  # In updates. Small value can generate
# gigabytes of data
TOTAL_TIMESTEPS = int(1e9)  # 5-10 mil is usually enough. You can
# manually
# terminate the process with CTRL+C at any time, and it will be
# saved, just be sure you saw "saved" recently.

LEARNING_RATE = 3e-4
N_ENVS = 1  # The number of parallel game environments
N_STEPS = 2048  # The number of steps to run in each environment per
# policy rollout
# value networks
GAMMA = 0.99  # The discount factor gamma
GAE_LAMBDA = 0.95  # The lambda for the general advantage estimation
N_MINIBATCHES = 32
UPDATE_EPOCHS = 10  # The K epochs to update the policy
NORM_ADV = True  # Toggles advantages normalization
CLIP_COEF = 0.2  # The surrogate clipping coefficient
CLIP_VLOSS = True  # Toggles whether to use a clipped loss for the
# value function, as per the paper
ENT_COEF = 0.0  # Coefficient of the entropy
VF_COEF = 0.5  # Coefficient of the value function
MAX_GRAD_NORM = 0.5  # The maximum norm for the gradient clipping
TARGET_KL = None  # The target KL divergence threshold
RPO_ALPHA = 0.5  # The alpha parameter for RPO
BATCH_SIZE = int(N_ENVS * N_STEPS)
MINIBATCH_SIZE = int(BATCH_SIZE // N_MINIBATCHES)


def make_env(env_id, idx, run_name, gamma):
    def thunk():
        from tripod_env import env
        env = DmControlCompatibilityV0(env)
        env = gym.wrappers.FlattenObservation(
            env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env,
            lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env,
            lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


if __name__ == "__main__":

    # Even if there is no loading, we don't want prev runs to be
    # overwritten
    path = f"runs/{ENV_ID}"
    if os.path.exists(path):
        dir_list = os.listdir(path)
        if dir_list:
            dir_list.sort(key=int)
            last_run_id = int(dir_list[-1])
            new_run_id = last_run_id + 1
            run_name = str(new_run_id)
        else:
            run_name = "1"
    else:
        run_name = "1"

    writer = SummaryWriter(f"runs/{ENV_ID}/{run_name}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, run_name, GAMMA) for i in
         range(N_ENVS)]
    )
    assert isinstance(envs.single_action_space,
        gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(device, envs, RPO_ALPHA).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE,
                           eps=1e-5)

    # ALGO Logic: Storage setup.
    obs = torch.zeros(
        (N_STEPS, N_ENVS) + envs.single_observation_space.shape).to(
        device)
    actions = torch.zeros(
        (N_STEPS, N_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((N_STEPS, N_ENVS)).to(device)
    rewards = torch.zeros((N_STEPS, N_ENVS)).to(device)
    dones = torch.zeros((N_STEPS, N_ENVS)).to(device)
    values = torch.zeros((N_STEPS, N_ENVS)).to(device)

    # Start the game
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(N_ENVS).to(device)
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE

    starting_update = 1

    if LOAD:
        path = f"runs/{ENV_ID}/{last_run_id}"
        dir_list = os.listdir(path)

        checkpoint = torch.load(f"runs/{ENV_ID}"
            f"/{last_run_id}/checkpoint_{LOAD_UPDATE}.tar")
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.train()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        obs = checkpoint['obs']
        actions = checkpoint['actions']
        logprobs = checkpoint['logprobs']
        rewards = checkpoint['rewards']
        dones = checkpoint['dones']
        values = checkpoint['values']

        starting_update = checkpoint['update'] + 1
        global_step = checkpoint['global_step'] + 1
        # print(optimizer.state_dict())
        print(f"resumed at update {starting_update}")

    for update in range(starting_update, num_updates + 1):
        print(f"update {update}")

        # Annealing the rate if instructed to do so.
        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, N_STEPS):
            global_step += 1 * N_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = \
                    agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = \
                envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(
                    f"global_step={global_step}, "
                    f"episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return",
                                  info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length",
                                  info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(N_STEPS)):
                if t == N_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues \
                        * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * \
                        GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) +
                            envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) +
                                    envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = \
                    agent.get_action_and_value(b_obs[mb_inds],
                                               b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl
                    # http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_advantages = (mb_advantages
                - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio,
                                                        1 - CLIP_COEF,
                                                        1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[
                        mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -CLIP_COEF, CLIP_COEF,
                    )
                    v_loss_clipped = (v_clipped - b_returns[
                        mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped,
                                           v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[
                        mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss \
                       + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         MAX_GRAD_NORM)
                optimizer.step()

            if TARGET_KL is not None:
                if approx_kl > TARGET_KL:
                    break

        y_pred, y_true = b_values.cpu().numpy(),\
            b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(
            y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate",
            optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss",
                          v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss",
                          pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy",
                          entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(),
                          global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs),
                          global_step)
        writer.add_scalar("losses/explained_variance", explained_var,
                          global_step)

        # Saving
        if SAVE:
            if update % CHECKPOINT_PERIOD == 0 \
                    or update == num_updates:
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),

                    'obs': obs,
                    'actions': actions,
                    'logprobs': logprobs,
                    'rewards': rewards,
                    'dones': dones,
                    'values': values,

                    'update': update,
                    'global_step': global_step,

                },
                    f"runs/{ENV_ID}/{run_name}/checkpoint"
                    f"_{update}.tar"
                )
                print("Saved")

    envs.close()
    writer.close()
