import datetime
import os
import re

import cv2
import gymnasium as gym
import numpy as np
import torch
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

from agent import Agent
from tripod_env import env


VIDEO_NAME = f'{datetime.datetime.now()}.mp4'
LENGTH = 1000
WIDTH = 600
HEIGHT = 480

ENV_ID = "dragoon"
N_ENVS = 1


def make_env(env_id, idx, run_name, gamma=0.99):
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


def convert_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    video_writer = cv2.VideoWriter(VIDEO_NAME,
        cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (WIDTH, HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX
    dir_list = os.listdir(f"runs/{ENV_ID}")
    dir_list.sort(key=int)
    try:
        last_run_id = int(dir_list[-1])
    except IndexError:
        print('No checkpoint to load')
        exit()
    new_run_id = last_run_id + 1
    run_name = str(new_run_id)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, run_name) for i in
         range(N_ENVS)]
    )
    assert isinstance(envs.single_action_space,
        gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(device, envs).to(device)
    agent.enjoy = True

    # Start the game
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(N_ENVS).to(device)

    # Getting the latest update
    path = f"runs/{ENV_ID}/{last_run_id}"
    filenames = os.listdir(path)
    checkpoint_filenames = []
    for filename in filenames:
        if filename[:10] == "checkpoint":
            checkpoint_filenames.append(filename)
    updates = []
    for filename in checkpoint_filenames:
        updates.append(re.search(r'\d+', filename).group())
    updates.sort(key=int)
    updates = [int(el) for el in updates]
    try:
        latest_update = updates[-1]
    except IndexError:
        print('No checkpoint to load')
        exit()

    checkpoint = torch.load(f"runs/{ENV_ID}"
        f"/{last_run_id}/checkpoint_{latest_update}.tar")
    starting_update = checkpoint['update'] + 1
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    print(f"resumed at update {starting_update}")

    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtFrame.mjFRAME_SITE] = False

    for step in range(0, LENGTH):

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = \
                agent.get_action_and_value(next_obs)

        # Execute the game
        next_obs, reward, terminated, truncated, infos = \
            envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        next_obs, next_done = torch.Tensor(next_obs).to(
            device), torch.Tensor(done).to(device)

        # Video
        frame = env.physics.render(HEIGHT, WIDTH,
                                   scene_option=scene_option)
        frame = convert_rgb(frame)

        # Normalized reward
        cv2.putText(img=frame,
                    text=f'nr: {str(round(reward[0], 2))}',
                    org=(50, 50),
                    fontFace=font, fontScale=1,
                    color=(0, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_4)

        # Frame
        cv2.putText(img=frame,
                    text=f'f: {step}',
                    org=(50, 100),
                    fontFace=font, fontScale=1,
                    color=(0, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_4)
        video_writer.write(frame)

    envs.close()
    print("video made")
    video_writer.release()
