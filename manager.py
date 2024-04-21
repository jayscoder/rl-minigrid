import os.path
import json

import pybts
from tqdm import tqdm
import time
import argparse
from bt import *
from common import *
import gymnasium as gym
import minigrid
from stable_baselines3.common.base_class import BaseAlgorithm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')  # 是否开启训练
parser.add_argument('--render', action='store_true')  # 是否开启渲染
parser.add_argument('--track', action='store_true')  # 是否开启pybts监控
parser.add_argument('--debug', action='store_true')  # 是否开启DEBUG模式，debug模式下会有更多的输出内容
args = parser.parse_args()


def gen_seed():
    # return 0
    return int(time.time())


random.seed(gen_seed())
np.random.seed(gen_seed())
torch.manual_seed(gen_seed())


def make_env(env_id, render: bool = False):
    return gym.make(env_id, render_mode='human' if (args.render or render) else 'rgb_array')


class Manager:

    def __init__(self, code_file: str, env: gym.Env, version: str = 'v0', debug=args.debug, name=''):
        self.code_file = code_file
        self.base_dir = os.path.dirname(code_file)
        self.filename = code_file.split('/')[-1].split('.')[0]
        self.name = name or self.filename
        self.version = version
        self.logs_dir = os.path.join(self.base_dir, 'logs', self.name, self.version)
        self.models_dir = os.path.join(self.base_dir, 'models', self.name, self.version)
        self.env = env
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.logger = TensorboardLogger(folder=self.logs_dir, verbose=1)
        self.debug = debug or args.debug

    def run_model(self, model: BaseAlgorithm, train: bool = args.train):
        """运行一个强化学习模型"""
        model_path = os.path.join(self.models_dir, self.filename)
        os.makedirs(self.models_dir, exist_ok=True)
        model.tensorboard_log = self.logs_dir
        train = train or args.train
        if train:
            model.learn(2e5, progress_bar=True)
            model.save(model_path)
        model = model.load(model_path)
        self.evaluate_model(model=model)

    def run_policy(self, policy: BTPolicy, train: bool = False, track: bool = False, episodes: int = 100):
        """运行一个行为树策略"""
        train = train or args.train
        policy.tree.context.update({
            'models_dir': self.models_dir,
            'logs_dir'  : self.logs_dir,
            'train'     : train,
            'base_dir'  : self.base_dir,
            'filename'  : self.filename,
            'debug'     : self.debug,
            'version'   : self.version
        })
        policy.tree.setup()
        policy.reset()
        self.evaluate_policy(policy, episodes=episodes, track=track or args.track, train=train)

    def evaluate_policy(self, policy: BTPolicy, episodes: int, track: bool, train: bool):
        env = policy.env
        reward_list = []
        step_count_list = []
        terminated_list = []
        truncated_list = []
        terminated_count = 0

        board = pybts.Board(tree=policy.tree, log_dir=os.path.join(self.logs_dir, f'{self.filename}'))
        if track:
            board.clear()

        for episode in range(episodes):
            policy.reset()
            if episode % 10 == 0:
                # 每隔10轮清理一下
                board.clear()
            env.reset(seed=gen_seed())
            obs, accum_reward, terminated, truncated, info = None, 0, False, False, None
            with tqdm(total=10000, desc=f'[{episode}, train={train}]') as pbar:
                while not env.done:
                    policy.tree.context['step'] = pbar.n
                    policy.take_action()
                    if track:
                        board.track({
                            'episode'     : episode,
                            'accum_reward': accum_reward,
                            'actions'     : pybts.utility.read_queue_without_destroying(env.actions)
                        })
                    obs, reward, terminated, truncated, info = env.update()
                    # 将环境奖励存储到默认学习域中
                    policy.tree.context['reward']['default'] += reward

                    # context中的本来就是累积奖励了，所以直接加起来就行
                    accum_reward = 0
                    for s in policy.tree.context['reward']:
                        accum_reward += policy.tree.context['reward'][s]

                    pbar.update()
                    pbar.set_postfix_str(f'accum_reward: {accum_reward}, terminated={terminated_count}')

                    if terminated or truncated:
                        break

            policy.terminate()  # 最后触发一次，避免遗漏奖励
            if terminated:
                terminated_count += 1
            terminated_list.append(terminated)
            truncated_list.append(truncated)

            self.logger.record('accum_reward', accum_reward)
            self.logger.record('step_count', pbar.n)
            self.logger.dump(episode)

            reward_list.append(accum_reward)
            step_count_list.append(pbar.n)

        with open(os.path.join(self.logs_dir, f'{self.filename}.json'), 'w') as f:
            json.dump({
                'rewards'       : sum(reward_list),
                'average_reward': sum(reward_list) / len(reward_list),
                'steps'         : sum(step_count_list),
                'average_step'  : sum(step_count_list) / len(step_count_list),
                'terminated'    : sum(terminated_list),
                'truncated'     : sum(truncated_list)
            }, f, ensure_ascii=False, indent=4)

        return reward_list, step_count_list, terminated_list, truncated_list

    def evaluate_model(self, model: BaseAlgorithm):
        N = 20
        rewards = []
        steps = []
        terminated_list = []
        truncated_list = []
        for episode in range(N):
            obs, _ = self.env.reset(seed=gen_seed())
            step_reward = 0
            with tqdm(range(2600), desc=f'[{episode}]') as pbar:
                for step in pbar:
                    action, state = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    step_reward += reward

                    pbar.set_postfix_str(f'reward: {step_reward}')
                    if terminated or truncated:
                        terminated_list.append(terminated)
                        truncated_list.append(truncated)
                        break
            rewards.append(step_reward)
            steps.append(step)
        with open(os.path.join(self.logs_dir, f'{self.filename}.json'), 'w') as f:
            json.dump({
                'rewards'       : sum(rewards),
                'average_reward': sum(rewards) / len(rewards),
                'steps'         : sum(steps),
                'average_step'  : sum(steps) / len(steps),
                'terminated'    : sum(terminated_list),
                'truncated'     : sum(truncated_list)
            }, f, ensure_ascii=False, indent=4)
