from __future__ import annotations
from pybts.nodes import Status, Success, Node
from pybts.decorators.nodes import Decorator
from collections import defaultdict
from rl.off_policy import *
from rl.on_policy import *
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from abc import ABC, abstractmethod
from typing import Union, Type
import gymnasium as gym
from rl.common import DummyEnv
from stable_baselines3.common.logger import Logger


class Reward(Node):
    """
    奖励节点，返回的状态一直是SUCCESS
    当走到这个节点时，会将给定的奖励累计

    只会对之后的节点生效，所以要放在前面

    reward: 给出的对应奖励
    domain: 学习域

    奖励会累积，所以PPO节点在消费奖励时要记录一下上次拿到的奖励值，然后将两次差值作为最终奖励
    """

    def __init__(self, reward: str | float, domain: str = 'default', **kwargs):
        super().__init__(**kwargs)
        self.reward = reward
        self.domain = domain
        self.curr_reward = 0

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.domain = self.converter.render(self.domain).split(',')  # 域，只会将奖励保存在对应的scope中

    def update(self) -> Status:
        assert self.context is not None, 'context is not set'
        self.curr_reward = self.converter.float(self.reward)
        if self.context is not None:
            if 'reward' not in self.context:
                self.context['reward'] = defaultdict(float)
            for sc in self.domain:
                self.context['reward'][sc] += self.curr_reward
        return Status.SUCCESS

    def to_data(self):
        return {
            # **super().to_data(),
            'curr_reward'   : self.curr_reward,
            'domain'        : self.domain,
            'context_reward': dict(self.context['reward'])
        }


class Terminate(Node):
    """
    终止一个学习域
    """

    def __init__(self, domain: str = 'default', **kwargs):
        super().__init__(**kwargs)
        self.domain = domain


class ConditionReward(Decorator):
    """
    强化学习奖励装饰节点
    根据子节点的状态来提供奖励

    success: 子节点成功时给的奖励
    failure: 子节点失败时给的奖励
    running: 子节点运行时给的奖励
    only_on_status_change: 只有在子节点的状态改变时才会提供奖励
    """

    def __init__(self,
                 scope: str = 'default',
                 success: float | str = 1, failure: float | str = 0, running: float | str = 0.5,
                 only_on_status_change: bool | str = False, **kwargs):
        super().__init__(**kwargs)
        self.success = success
        self.failure = failure
        self.running = running
        self.only_on_status_change = only_on_status_change
        self.scope = scope

        self.reward = 0  # 单次奖励
        self.accum_reward = 0  # 累积奖励

    def reset(self):
        super().reset()
        self.reward = 0
        self.accum_reward = 0

    def to_data(self):
        return {
            **super().to_data(),
            'scope'                : self.scope,
            'success'              : self.success,
            'failure'              : self.failure,
            'running'              : self.running,
            'only_on_status_change': self.only_on_status_change,
            'reward'               : self.reward,
            'accum_reward'         : self.accum_reward,
            'context'              : self.context
        }

    def setup(self, **kwargs: typing.Any) -> None:
        super().setup(**kwargs)
        self.success = self.converter.float(self.success)
        self.failure = self.converter.float(self.failure)
        self.running = self.converter.float(self.running)
        self.only_on_status_change = self.converter.bool(self.only_on_status_change)
        self.scope = self.converter.render(self.scope).split(',')  # 域，只会将奖励保存在对应的scope中

    def update(self) -> Status:
        new_status = self.decorated.status
        new_reward = 0
        if new_status == self.status and self.only_on_status_change:
            self.reward = new_reward
            return new_status

        elif new_status == Status.SUCCESS:
            new_reward = self.success
        elif new_status == Status.RUNNING:
            new_reward = self.running
        elif new_status == Status.FAILURE:
            new_reward = self.failure
        self.reward = new_reward
        self.accum_reward += new_reward

        if self.context is not None:
            for sc in self.scope:
                self.context['reward'][sc] += self.reward

        return new_status


class RLBaseNode(ABC):
    """强化学习基础节点，拿来跟其他的Node多继承用"""

    def __init__(self):
        self.rl_accum_reward = 0  # 当前累积奖励
        self.rl_info = None
        self.rl_reward = 0  # 当前奖励
        self.rl_obs = None
        self.rl_obs_index = 0  # 上一次观测的经验索引
        self.rl_done = False
        self.rl_action = None
        self.rl_model: Optional[BaseAlgorithm] = None
        self.rl_values = None
        self.rl_log_probs = None
        self.rl_log_interval = 0

    def reset(self):
        self.rl_accum_reward = 0  # 当前累积奖励
        self.rl_info = None
        self.rl_reward = 0  # 当前奖励
        self.rl_obs = None
        self.rl_obs_index = 0  # 上一次观测的经验索引
        self.rl_done = False
        self.rl_action = None
        self.rl_values = None
        self.rl_log_probs = None
        self.rl_log_interval = 0
        self.rl_handler.reset()

    def to_data(self):
        return {
            'rl_info'        : self.rl_info,
            'rl_reward'      : self.rl_reward,
            'rl_obs'         : self.rl_obs,
            'rl_accum_reward': self.rl_accum_reward,
            'rl_action'      : self.rl_action,
            'rl_reward_scope': self.rl_domain(),
        }

    @abstractmethod
    def rl_action_space(self) -> gym.spaces.Space:
        raise NotImplemented

    @abstractmethod
    def rl_observation_space(self) -> gym.spaces.Space:
        raise NotImplemented

    @abstractmethod
    def rl_gen_obs(self):
        raise NotImplemented

    @abstractmethod
    def rl_gen_info(self) -> dict:
        raise NotImplemented

    def rl_domain(self) -> str:
        """
        奖励域

        例如：default
        多个奖励域用,分隔
        如果设置了奖励域，则生成本轮奖励时会从self.context.rl_reward[scope]里获取
        """
        return ''

    @abstractmethod
    def rl_gen_reward(self) -> float:
        reward_domain = self.rl_domain()
        if reward_domain != '':
            assert isinstance(self, Node), 'RLOnPolicyNode 必须得继承Node节点'
            assert self.context is not None, 'context必须得设置好'
            assert 'reward' in self.context, 'context必须得含有rl_reward键，请使用rl.RLTree'
            scopes = reward_domain.split(',')
            curr_reward = 0
            for scope in scopes:
                curr_reward += self.context['reward'].get(scope, 0)
            return curr_reward - self.rl_accum_reward
        raise NotImplemented

    @abstractmethod
    def rl_gen_done(self) -> bool:
        # 返回当前环境是否结束
        raise NotImplemented

    def rl_setup_model(
            self,
            model_class: Union[Type[BaseAlgorithm]],
            path: str,
            policy: Union[str, typing.Type[ActorCriticPolicy]],
            train: bool = True,
            logger: Logger | None = None,
            tensorboard_log: str | None = None,
            tb_log_name: str | None = None,
            verbose: int = 1,
            device: str = 'auto',
            log_interval: int = 1,
            **kwargs
    ):
        env = DummyEnv(
                obs=self.rl_gen_obs(),
                info=self.rl_gen_info(),
                action_space=self.rl_action_space(),
                observation_space=self.rl_observation_space())
        model: typing.Optional[BaseAlgorithm] = None

        if train:
            model = model_class(
                    policy=policy,
                    env=env,
                    verbose=verbose,
                    device=device,
                    tensorboard_log=tensorboard_log,
                    **kwargs
            )

            if path != '':
                try:
                    model.set_parameters(
                            load_path_or_dict=path,
                            device=device
                    )
                except Exception as e:
                    pass
        else:
            assert path != '', f'No model path provided: {path}'
            model = model_class.load(
                    path=path,
                    env=env,
                    verbose=verbose,
                    force_reset=False,
                    device=device,
                    **kwargs
            )
        if logger is not None:
            model.set_logger(logger=logger)

        if train:
            if isinstance(model, OffPolicyAlgorithm):
                bt_off_policy_setup_learn(
                        model,
                        tb_log_name=tb_log_name,
                )
            elif isinstance(model, OnPolicyAlgorithm):
                bt_on_policy_setup_learn(
                        model,
                        obs=self.rl_gen_obs(),
                        tb_log_name=tb_log_name
                )
            else:
                raise Exception('Unrecognized model type')

        self.rl_model = model
        self.rl_log_interval = log_interval
        if isinstance(model, OffPolicyAlgorithm):
            self.rl_handler = OffPolicyRLHandler(model=model, log_interval=log_interval)
        else:
            self.rl_handler = OnPolicyRLHandler(model=model, log_interval=log_interval)
        return model

    def rl_observe(self, train: bool, action, obs, reward, done, info, obs_index: int):
        """
        观测
        :param train:
        :param action:
        :param obs:
        :param reward:
        :param done:
        :param info:
        :param obs_index: 观测的索引
        :return:
        """
        # if obs_index == self.rl_obs_index:
        #     print(f'观测了相同的经验 {self.rl_obs_index}')

        self.rl_values = None  # 上一帧可能也是经验填充的，所以这里干脆就清除调，反正之后会自动在observer里再算一个出来
        self.rl_log_probs = None
        if train:
            should_train = self.rl_handler.observe(
                    actions=action,
                    infos=info,
                    rewards=reward,
                    new_obs=obs,
                    dones=done,
                    values=None, log_probs=None)
            if should_train:
                self.rl_handler.train()
        self.rl_obs_index = obs_index

    def rl_take_action(
            self,
            train: bool,
            deterministic: bool = False,
    ):
        info = self.rl_gen_info()
        reward = self.rl_gen_reward()
        obs = self.rl_gen_obs()
        done = self.rl_gen_done()
        if train:
            self.rl_action, self.rl_values, self.rl_log_probs = self.rl_handler.predict()
        else:
            self.rl_action, _ = self.rl_model.predict(obs, deterministic=deterministic)

        self.rl_obs = obs
        self.rl_reward = reward
        self.rl_info = info
        self.rl_accum_reward += reward
        self.rl_done = done
        return self.rl_action

        # if isinstance(self.rl_model, OnPolicyAlgorithm):
        #     return self._rl_on_policy_take_action(
        #             train=train,
        #             log_interval=self.rl_log_interval,
        #             deterministic=deterministic
        #     )
        # else:
        #     return self._rl_off_policy_take_action(
        #             train=train,
        #             log_interval=self.rl_log_interval,
        #             deterministic=deterministic
        #     )

    # def _rl_off_policy_take_action(
    #         self,
    #         train: bool,
    #         log_interval: int = 1,
    #         deterministic: bool = False,
    # ):
    #     assert self.rl_model is not None, 'RL model not initialized'
    #     assert isinstance(self.rl_model, OffPolicyAlgorithm), 'RL model must be initialized with OffPolicyAlgorithm'
    #     model: OffPolicyAlgorithm = self.rl_model
    #     info = self.rl_gen_info()
    #     reward = self.rl_gen_reward()
    #     obs = self.rl_gen_obs()
    #     done = self.rl_gen_done()
    #     if train:
    #         try:
    #             if self.rl_collector is None:
    #                 self.rl_collector = bt_off_policy_collect_rollouts(
    #                         model,
    #                         train_freq=model.train_freq,
    #                         action_noise=model.action_noise,
    #                         learning_starts=model.learning_starts,
    #                         replay_buffer=model.replay_buffer,
    #                         log_interval=log_interval,
    #                 )
    #                 action = self.rl_collector.send(None)
    #             else:
    #                 info = info
    #                 action = self.rl_collector.send((obs, reward, done, info))
    #
    #             if isinstance(action, RolloutReturn):
    #                 # 结束了
    #                 rollout: RolloutReturn = action
    #                 if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
    #                     # If no `gradient_steps` is specified,
    #                     # do as many gradients steps as steps performed during the rollout
    #                     gradient_steps = model.gradient_steps if model.gradient_steps >= 0 else rollout.episode_timesteps
    #                     # Special case when the user passes `gradient_steps=0`
    #                     if gradient_steps > 0:
    #                         model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)
    #
    #                 self.rl_collector = bt_off_policy_collect_rollouts(
    #                         model,
    #                         train_freq=model.train_freq,
    #                         action_noise=model.action_noise,
    #                         learning_starts=model.learning_starts,
    #                         replay_buffer=model.replay_buffer,
    #                         log_interval=log_interval,
    #                 )
    #                 action = self.rl_collector.send(None)
    #         except StopIteration:
    #             self.rl_collector = None
    #             self.rl_iteration += 1
    #             # Display training infos
    #
    #             self.rl_collector = bt_off_policy_collect_rollouts(
    #                     model,
    #                     train_freq=model.train_freq,
    #                     action_noise=model.action_noise,
    #                     learning_starts=model.learning_starts,
    #                     replay_buffer=model.replay_buffer,
    #                     log_interval=log_interval,
    #             )
    #             action = self.rl_collector.send(None)
    #     else:
    #         # 预测模式
    #         action, state = model.predict(obs, deterministic=deterministic)
    #
    #     self.rl_obs = obs
    #     self.rl_reward = reward
    #     self.rl_info = info
    #     self.rl_accum_reward += reward
    #     self.rl_action = action
    #     self.rl_done = done
    #     return action
    #
    # def _rl_on_policy_take_action(
    #         self,
    #         train: bool,
    #         log_interval: int = 1,
    #         deterministic: bool = False,
    # ):
    #     assert self.rl_model is not None, 'RL model not initialized'
    #     assert isinstance(self.rl_model, OnPolicyAlgorithm), 'RL model must be an instance of OnPolicyAlgorithm'
    #     model: OnPolicyAlgorithm = self.rl_model
    #     info = self.rl_gen_info()
    #     reward = self.rl_gen_reward()
    #     obs = self.rl_gen_obs()
    #     done = self.rl_gen_done()
    #     if train:
    #         try:
    #             if self.rl_collector is None:
    #                 self.rl_collector = bt_on_policy_collect_rollouts(
    #                         model,
    #                         last_obs=obs)
    #                 action = self.rl_collector.send(None)
    #             else:
    #                 info = info
    #                 action = self.rl_collector.send((obs, reward, done, info))
    #         except StopIteration:
    #             self.rl_collector = None
    #             self.rl_iteration += 1
    #             # Display training infos
    #             bt_on_policy_train(model, iteration=self.rl_iteration, log_interval=log_interval)
    #
    #             self.rl_collector = bt_on_policy_collect_rollouts(
    #                     model,
    #                     last_obs=obs)
    #             action = self.rl_collector.send(None)
    #     else:
    #         # 预测模式
    #         action, state = model.predict(obs, deterministic=deterministic)
    #
    #     self.rl_obs = obs
    #     self.rl_reward = reward
    #     self.rl_info = info
    #     self.rl_accum_reward += reward
    #     self.rl_action = action
    #     self.rl_done = done
    #     return action


if __name__ == '__main__':
    node = ConditionReward(scope='a,b,c', success='{{1}}/10', children=[Success()])
    node.setup()
    print(node.success)
    print(node.update())
