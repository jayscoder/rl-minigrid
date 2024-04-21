from __future__ import annotations

import numpy as np

from bt.base import *
from py_trees.common import Status
from rl.nodes import Reward
from stable_baselines3 import PPO, SAC, HerReplayBuffer, DQN, DDPG, TD3
import gymnasium as gym
from rl import RLBaseNode
import typing
from rl.logger import TensorboardLogger
from rl.common import is_off_policy_algo
from common import *
from pybts import Composite, Switcher, Sequence, Selector, Behaviour
from common.constants import *
from rl.common import *


class RLNode(BaseBTNode, RLBaseNode, ABC):
    """

    deterministic:
        true: 确定性动作意味着对于给定的状态或观测，策略总是返回相同的动作。没有随机性或变化性涉及，每次给定相同的输入状态，输出（即动作）总是一样的。
            在实际应用中，确定性选择通常用于部署阶段，当你希望模型表现出最稳定、可预测的行为时，例如在测试或实际运行环境中。
        false: 随机性动作则意味着策略在给定的状态下可能产生多种可能的动作。这通常是通过策略输出的概率分布实现的，例如，一个使用softmax输出层的神经网络可能会对每个可能的动作分配一个概率，然后根据这个分布随机选择动作。
            随机性在训练阶段特别有用，因为它可以增加探索，即允许代理（agent）尝试和学习那些未必立即最优但可能长期更有益的动作。这有助于策略避免陷入局部最优并更全面地学习环境。
    """

    def __init__(self,
                 path: str = '',
                 algo: str = 'PPO',
                 domain: str = '',
                 save_path: str = '',  # 空代表不保存
                 save_interval: int | str = 0,
                 deterministic: bool | str = False,
                 train: bool | str = False,
                 tensorboard_log: str = '',
                 **kwargs
                 ):
        super().__init__(**kwargs)
        RLBaseNode.__init__(self)
        self.algo = algo
        self.domain = domain  # 如果scope设置成default或其他不为空的值，则认为奖励要从context.rl_reward[scope]中拿
        self.path = path
        self.save_path = save_path
        self.save_interval = save_interval
        self.deterministic = deterministic
        self.train = train
        self.tensorboard_log = tensorboard_log

    def to_data(self):
        return {
            **super().to_data(),
            **RLBaseNode.to_data(self),
            'algo'         : str(self.algo),
            'path'         : self.path,
            'domain'       : self.domain,
            'save_interval': self.save_interval,
        }

    def rl_model_args(self) -> dict:
        # 这里来实现模型的参数
        policy_kwargs = dict(
                features_extractor_class=CNNMinigridFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
        )

        attrs = {
            'policy_kwargs': policy_kwargs,
        }

        if 'SAC' in self.algo:
            attrs.update({
                'train_freq': (10, "step"),
            })

        return attrs
        # return {
        #     'policy_kwargs': policy_kwargs,
        #     # 'learning_starts': learning_starts,
        #     # 'batch_size'     : batch_size,
        #     # 'train_freq'   : (10, "step"),
        #     # 'gradient_steps' : 1,
        # }

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)

        self.path = self.converter.render(
                value=self.path,
        )
        self.domain = self.converter.render(self.domain)
        self.save_interval = self.converter.int(self.save_interval)
        self.algo = self.converter.render(self.algo).upper()
        if self.tensorboard_log != '':
            self.tensorboard_log = self.converter.render(self.tensorboard_log)

        args = self.rl_model_args()
        for key in ['batch_size', 'n_steps', 'learning_starts', 'verbose']:
            if key in self.attrs:
                args[key] = self.converter.int(self.attrs[key])

        self.setup_model(algo=self.algo, **args)

    def setup_model(self, algo: str, policy: str = 'MlpPolicy', **kwargs):
        tensorboard_logger = TensorboardLogger(folder=self.tensorboard_log, verbose=0)

        if algo == 'PPO':
            self.rl_setup_model(
                    policy=policy,
                    model_class=PPO,
                    train=True,
                    path=self.path,
                    logger=tensorboard_logger,
                    **kwargs
            )
        elif algo == 'SAC':
            self.rl_setup_model(
                    policy=policy,
                    model_class=SAC,
                    train=True,  # 在训练过程中可能会开/闭某个节点的训练，所以一开始初始化都默认开启训练
                    path=self.path,
                    logger=tensorboard_logger,
                    **kwargs
            )
        elif algo == 'SAC-HER':
            self.rl_setup_model(
                    policy=policy,
                    model_class=SAC,
                    train=True,
                    path=self.path,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy='future',
                    ),
                    logger=tensorboard_logger,
                    **kwargs
            )
        elif algo == 'TD3':
            self.rl_setup_model(
                    policy=policy,
                    model_class=TD3,
                    train=True,
                    path=self.path,
                    logger=tensorboard_logger,
                    **kwargs
            )
        elif algo == 'TD3-HER':
            self.rl_setup_model(
                    policy=policy,
                    model_class=TD3,
                    train=True,
                    path=self.path,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy='future',
                    ),
                    logger=tensorboard_logger,
                    **kwargs
            )
        else:
            raise Exception(f'Unsupported algo type {algo}')

    def rl_env(self) -> gym.Env:
        return self.env

    def rl_action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    def rl_observation_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    def rl_gen_obs(self):
        return self.env.gen_obs()

    def rl_gen_info(self) -> dict:
        return self.env.gen_info()

    def rl_gen_reward(self) -> float:
        if self.domain != '':
            return RLBaseNode.rl_gen_reward(self)
        return self.env.gen_reward()

    def rl_domain(self) -> str:
        return self.domain

    def rl_gen_done(self) -> bool:
        info = self.env.gen_info()
        return info['terminated'] or info['truncated']

    def rl_device(self) -> str:
        return self.env.options.device

    def reset(self):
        # self.rl_model.logger.record("final_reward", self.rl_reward)
        # self.rl_model.logger.record("return", self.rl_accum_reward)

        # self.rl_model.logger.record("win", self.env.game_info[self.agent.color]['win'])
        # self.rl_model.logger.record("lose", self.env.game_info[self.agent.color]['lose'])
        # self.rl_model.logger.record("draws", self.env.game_info['draws'])
        # self.rl_model.logger.record("destroyed_count", self.agent.destroyed_count)
        # self.rl_model.logger.record("missile_hit_enemy_count", self.agent.missile_hit_enemy_count)
        # self.rl_model.logger.record("missile_miss_count", self.agent.missile_miss_count),
        # self.rl_model.logger.record("missile_hit_self_count", self.agent.missile_hit_self_count),
        # self.rl_model.logger.record("missile_fired_count", self.agent.missile_fired_count),
        # self.rl_model.logger.record("missile_evade_success_count", self.agent.missile_evade_success_count)
        # self.rl_model.logger.record("aircraft_collided_count", self.agent.aircraft_collided_count)

        if self.env.episode > 0 and self.save_interval > 0 and self.env.episode % self.save_interval == 0 and self.save_path != '':
            save_path = self.converter.render(self.save_path)
            self.rl_model.save(path=save_path)

        super().reset()
        RLBaseNode.reset(self)

    def take_action(self):
        return self.rl_take_action(
                train=self.converter.bool(self.train),
                deterministic=self.converter.bool(self.deterministic),
        )

    def save_model(self, filepath: str = ''):
        if filepath == '':
            filepath = self.converter.render(self.save_path)
        else:
            filepath = self.converter.render(filepath)
        self.rl_model.save(path=filepath)


class RLComposite(RLNode, Composite):
    def __init__(self, exp_fill: bool | str = False, **kwargs):
        Composite.__init__(self, **kwargs)
        RLNode.__init__(self, **kwargs)
        self.exp_fill = exp_fill

    def rl_action_space(self) -> gym.spaces.Space:
        if is_off_policy_algo(self.algo):
            return gym.spaces.Box(low=0, high=len(self.children))
        return gym.spaces.Discrete(len(self.children))

    def rl_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
                {
                    'image' : RLNode.rl_observation_space(self),
                    'status': gym.spaces.MultiDiscrete([4] * len(self.children))
                }
        )

    def rl_gen_obs(self):
        return {
            'image' : RLNode.rl_gen_obs(self),
            'status': children_status_ids(self)
        }

    def rl_model_args(self) -> dict:
        # 这里来实现模型的参数
        policy_kwargs = dict(
                features_extractor_class=CompositeFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
        )

        attrs = {
            'policy_kwargs': policy_kwargs,
            'policy'       : 'MultiInputPolicy'
        }

        if 'SAC' in self.algo:
            attrs.update({
                'train_freq': (1, "episode"),
            })

        return attrs

    def gen_index(self) -> int:
        if is_off_policy_algo(self.algo):
            index = int(self.take_action()[0]) % len(self.children)
        else:
            index = self.take_action()
        self.put_update_message('gen_index index={}'.format(index))
        return index


class RLSwitcher(RLComposite, Switcher):
    """
    选择其中一个子节点来执行
    """

    def tick(self) -> typing.Iterator[Behaviour]:
        yield from Switcher.tick(self)


class RLSelector(RLComposite, Selector):
    """
    将某个子节点作为开头来执行，剩下的按照Selector的规则来
    """

    def tick(self) -> typing.Iterator[Behaviour]:
        yield from Selector.tick(self)


class RLSequence(RLComposite, Sequence):
    """
    将某个子节点作为开头来执行，剩下的按照Sequence的规则来
    """

    def tick(self) -> typing.Iterator[Behaviour]:
        yield from Sequence.tick(self)


class RLCondition(RLNode, pybts.Condition):
    """
    条件节点：用RL来判断是否需要执行
    """

    def rl_action_space(self) -> gym.spaces.Space:
        if is_off_policy_algo(self.algo):
            return gym.spaces.Box(low=0, high=2, shape=(1,))
        return gym.spaces.Discrete(2)

    def update(self) -> Status:
        action = self.take_action()[0]
        if action >= 1:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class RLIntValue(RLNode):
    """
    强化学习int值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: int | str, low: int | str = 0, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        self.key = self.converter.render(self.key)
        self.high = self.converter.int(self.high)
        self.low = self.converter.int(self.low)

        super().setup(**kwargs)

        assert self.high > self.low, "RLIntValue high must > low"

    def to_data(self):
        return {
            **super().to_data(),
            "key" : self.key,
            "high": self.high,
            "low" : self.low,
        }

    def rl_action_space(self) -> gym.spaces.Space:
        self.high = self.converter.int(self.high)
        self.low = self.converter.int(self.low)
        return gym.spaces.Discrete(self.high - self.low + 1)

    def update(self) -> Status:
        action = self.take_action()
        self.context[self.key] = int(action)
        return Status.SUCCESS


class RLFloatValue(RLNode):
    """
    强化学习float值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: float | str, low: float | str = 0, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        assert self.high > self.low, "RLFloatValue high must > low"

    def to_data(self):
        return {
            **super().to_data(),
            "key" : self.key,
            "high": self.high,
            "low" : self.low,
        }

    def rl_action_space(self) -> gym.spaces.Space:
        self.high = self.converter.float(self.high)
        self.low = self.converter.float(self.low)
        return gym.spaces.Box(low=self.low, high=self.high, shape=(1,))

    def update(self) -> Status:
        action = self.take_action()[0]
        self.context[self.converter.render(self.key)] = float(action)
        return Status.SUCCESS


class RLFloatArrayValue(RLNode):
    """
    强化学习float值生成，将生成的值保存到context[key]里
    """

    def __init__(self, key: str, high: float | str, low: float | str = 0, length: int | str = 1, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.high = high
        self.low = low
        self.length = int(length)

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        assert self.high > self.low, "RLFloatValue high must > low"

    def to_data(self):
        return {
            **super().to_data(),
            "key"   : self.key,
            "high"  : self.high,
            "low"   : self.low,
            'length': self.length
        }

    def rl_action_space(self) -> gym.spaces.Space:
        self.high = self.converter.float(self.high)
        self.low = self.converter.float(self.low)
        self.length = self.converter.float(self.length)
        return gym.spaces.Box(low=self.low, high=self.high, shape=(self.length,))

    def update(self) -> Status:
        action = self.take_action()
        self.context[self.converter.render(self.key)] = action.tolist()
        return Status.SUCCESS


class RLAction(RLNode):
    def __init__(
            self,
            allow_actions: str = 'left,right,forward,pickup,drop,toggle',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.allow_actions = allow_actions

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.allow_actions = self.allow_actions.split(',')
        self.allow_actions = list(map(lambda x: ACTIONS_MAP[x], self.allow_actions))

    def to_data(self):
        return {
            **super().to_data(),
            'allow_actions': self.allow_actions
        }

    def rl_action_space(self) -> gym.spaces.Space:
        # （action_type, distance, angle）
        if is_off_policy_algo(self.algo):
            return gym.spaces.Box(
                    low=0,
                    high=len(self.allow_actions),
                    shape=(1,)
            )
        else:
            return gym.spaces.Discrete(len(self.allow_actions))

    def updater(self) -> typing.Iterator[Status]:
        action = self.take_action()
        if is_off_policy_algo(self.algo):
            action = int(action[0]) % len(self.allow_actions)
            action = self.allow_actions[action]
        else:
            action = self.allow_actions[action]
        
        self.put_action(action)
        yield Status.RUNNING
        # 看一下是否有变化
        info = self.env.gen_info()
        if info['is_changed']:
            yield Status.SUCCESS
        else:
            yield Status.FAILURE


class RLReward(RLNode, Reward):
    def __init__(self, high: int | str, low: int | str = 0, **kwargs):
        Reward.__init__(self, **kwargs)
        RLNode.__init__(self, **kwargs)
        self.high = high
        self.low = low

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.high = self.converter.float(self.high)
        self.low = self.converter.float(self.low)

    def to_data(self):
        return {
            **super().to_data(),
            **Reward.to_data(self),
            'low' : self.low,
            'high': self.high
        }

    def rl_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=self.low, high=self.high, shape=(1,))

    def update(self) -> Status:
        self.reward = self.take_action()
        print('RLReward', self.reward)
        return Reward.update(self)
