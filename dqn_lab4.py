import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports all our hyperparameters from the other file
from hyperparams import Hyperparameters as params

# stable_baselines3 have wrappers that simplifies 
# the preprocessing a lot, read more about them here:
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer


# Creates our gym environment and with all our wrappers.
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # TODO: Deinfe your network (agent)
        # Look at Section 4.1 in the paper for help: https://arxiv.org/pdf/1312.5602v1.pdf
        in_channels = env.single_observation_space.shape[0]
        n_actions  = env.single_action_space.n

        self.network = nn.Sequential(
            # conv 1: 32 @ 8×8, stride 4
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # conv 2: 64 @ 4×4, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # conv 3: 64 @ 3×3, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            # flatten
            nn.Flatten(),
            # fully-connected to 512
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            # final action‐value outputs
            nn.Linear(512, n_actions)
        )

    def forward(self, x):

        x = x.float() / 255.0
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(params.env_id, params.seed, 0, params.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # We’ll be using experience replay memory for training our DQN. 
    # It stores the transitions that the agent observes, allowing us to reuse this data later. 
    # By sampling from it randomly, the transitions that build up a batch are decorrelated. 
    # It has been shown that this greatly stabilizes and improves the DQN training procedure.
    rb = ReplayBuffer(
        params.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=True,
    )

    obs = envs.reset()
    for global_step in range(params.total_timesteps):
        # Here we get epsilon for our epislon greedy.
        epsilon = linear_schedule(params.start_e, params.end_e, params.exploration_fraction * params.total_timesteps, global_step)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)]) # TODO: sample a random action from the environment 
        else:
            state_tensor = torch.from_numpy(obs).to(device)
            q_values = q_network(state_tensor)  # TODO: get q_values from the network you defined, what should the network receive as input?
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Take a step in the environment
        next_obs, rewards, dones, infos = envs.step(actions)

        # Here we print our reward.
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                break

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        # Here we store the transitions in D
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        obs = next_obs
        # Training 
        if global_step > params.learning_starts:
            if global_step % params.train_frequency == 0:
                # Sample random minibatch of transitions from D
                data = rb.sample(params.batch_size)
                # You can get data with:
                # data.observation, data.rewards, data.dones, data.actions

                with torch.no_grad():
                    next_q = target_network(data.next_observations)
                    # Now we calculate the y_j for non-terminal phi.
                    target_max, _ = next_q.max(dim=1) # TODO: Calculate max Q
                    rewards = data.rewards.squeeze(-1)
                    dones = data.dones.squeeze(-1)
                    td_target = rewards + params.gamma * target_max * (1.0 - dones) # TODO: Calculate the td_target (y_j)

                old_val = q_network(data.observations).gather(1, data.actions.long().squeeze(-1).unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(old_val, td_target) 

                # perform our gradient decent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % params.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        params.tau * q_network_param.data + (1.0 - params.tau) * target_network_param.data
                    )

    if params.save_model:
        model_path = f"runs/{run_name}/{params.exp_name}_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
