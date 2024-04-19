import random

import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import torch.nn as nn
from tqdm import tqdm
from stable_baselines3.common.base_class import BaseAlgorithm
import time

def evaluate(name: str, model: BaseAlgorithm, env: gym.Env):
    N = 100
    rewards = []
    steps = []
    for i in tqdm(range(N)):
        obs, _ = env.reset(seed=time.time_ns())
        step_reward = 0
        for step in range(1000):
            action, state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_reward += reward
            if terminated or truncated:
                rewards.append(step_reward)
                steps.append(step)
                break

    with open(f'output/{name}.txt', 'w') as f:
        json.dump({
            'rewards'       : sum(rewards),
            'average_reward': sum(rewards) / len(rewards),
            'steps'         : sum(steps),
            'average_step'  : sum(steps) / len(steps)
        }, f, ensure_ascii=False, indent=4)
