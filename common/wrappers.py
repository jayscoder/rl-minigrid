from __future__ import annotations

import typing

from minigrid.core.constants import *
from common.utils import astar_find_path, manhattan_distance, iter_take, is_obs_same
from common.constants import *
from minigrid.wrappers import ObservationWrapper, ObsType, ImgObsWrapper, Any, DirectionObsWrapper
from rl.models import *
from pybts import Status
import gymnasium as gym
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from queue import Queue


class ContinuousActionWrapper(gym.ActionWrapper):
    """将离散动作变成连续动作"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n = env.action_space.n
        self.action_space = gym.spaces.Box(low=0, high=self.n, dtype=np.float32)

    def action(self, action):
        return int(action) % self.n


class MiniGridSimulation(gym.Wrapper):
    env: MiniGridEnv  # 使用类型提示指定 env 属性的类型为 MiniGridEnv

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env_id = env.unwrapped.spec.id
        self.step_count = 0
        self.episode = 0
        self.buffers = []  # 数据存储池
        self.done = False
        self.seed = 0
        self.train = False
        self.workspace = ''
        self.gif = ''
        self.gif_frames = []
        self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, self.width, self.height),
                dtype="uint8",
        )
        self.memory_obs = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        self.actions = Queue()
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.step_count > 0:
            self.episode += 1
        self.gif_frames = []
        self.seed = seed
        self.done = False
        obs, info = super().reset(seed=seed, options=options)
        self.memory_obs = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        info['is_changed'] = True
        info['terminated'] = False
        info['truncated'] = False
        self.update_memory_image(obs)
        obs = np.transpose(self.memory_obs, (2, 0, 1))
        self.buffers = [
            StepResult(
                    action=None,
                    obs=obs,
                    info=info
            )
        ]
        self.step_count = 0
        return obs, info

    def gen_obs(self):
        return self.buffers[-1].obs

    def gen_info(self):
        return self.buffers[-1].info

    def gen_reward(self):
        return self.buffers[-1].reward

    def put_action(self, action):
        self.actions.put_nowait(action)

    def update(self):
        obs, accum_reward, terminated, truncated, info = None, 0, None, None, None
        while not self.actions.empty() and not self.done:
            action = self.actions.get_nowait()
            obs, reward, terminated, truncated, info = self.step(action)
            accum_reward += reward
        return obs, accum_reward, terminated, truncated, info

    def step(self, action=None):
        old_obs = self.buffers[-1].obs

        if action is None:
            action = self.action_space.sample()
        obs, reward, terminated, truncated, info = super().step(action)
        self.done = terminated or truncated
        info['is_changed'] = is_obs_same(old_obs, obs)
        info['terminated'] = terminated
        info['truncated'] = truncated
        self.update_memory_image(obs)
        obs = np.transpose(self.memory_obs, (2, 0, 1))
        self.buffers.append(
                StepResult(action=action, obs=obs, reward=float(reward), terminated=terminated, truncated=truncated,
                           info=info))

        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def update_memory_image(self, obs):
        """
        更新记忆memory
        :param obs: 新的观测
        :return:
        这里有个潜在问题，就是如果agent携带key移动的话，上一步的位置还会留着key的观测，应该把这个背后的上个观测置为空
        """
        for i in range(self.width):
            for j in range(self.height):
                pos = self.env.relative_coords(i + 1, j + 1)  #
                # 注意：这里的坐标是从1开始的，而不是从0开始的，所以要减1
                if pos is None:
                    continue
                obs_item = obs['image'][pos[0], pos[1], :]
                if obs_item[0] == OBJECT_TO_IDX['unseen']:
                    # 墙壁会挡住视线，所以unseen不需要更新
                    continue
                self.memory_obs[i, j, :] = obs_item
        agent_obs = self.agent_obs
        agent_back_pos = self.agent_back_pos
        agent_back_obs = self.get_obs_item(agent_back_pos)
        if agent_obs.obj == 'key' and agent_back_obs is not None and agent_back_obs.obj == 'key':
            # 将背后的观测置为空
            self.memory_obs[agent_back_pos[0], agent_back_pos[1], 0] = OBJECT_TO_IDX['empty']

    @property
    def width(self) -> int:
        return self.env.unwrapped.width

    @property
    def height(self) -> int:
        return self.env.unwrapped.height

    @property
    def front_pos(self) -> (int, int):
        pos = self.env.front_pos
        return pos[0] - 1, pos[1] - 1

    @property
    def dir_vec(self) -> (int, int):
        vec = DIR_TO_VEC[self.agent_dir]
        return vec[0], vec[1]

    @property
    def agent_pos(self) -> (int, int):
        """agent前方的位置"""
        pos = self.env.unwrapped.agent_pos
        return pos[0] - 1, pos[1] - 1

    @property
    def agent_back_pos(self) -> (int, int):
        """agent后方的位置"""
        pos = self.agent_pos
        vec = DIR_TO_VEC[self.agent_dir]
        return pos[0] - vec[0], pos[1] - vec[1]

    @property
    def agent_dir(self) -> int:
        """agent的方向"""
        return self.env.unwrapped.agent_dir

    @property
    def agent_obs(self) -> ObsItem:
        return self.get_obs_item(self.agent_pos)

    def get_obs_item(self, pos: (int, int)) -> ObsItem | None:
        """
        获取指定位置的物体
        :param pos: 位置
        :return:
        """
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return None

        object_idx, color_idx, state_idx = self.memory_obs[pos[0], pos[1], :]
        return ObsItem(
                obj=IDX_TO_OBJECT[object_idx],
                color=IDX_TO_COLOR[color_idx],
                state=IDX_TO_STATE[state_idx],
                pos=pos)

    def find_can_reach_obs(self, obj: str, color: str = '') -> ObsItem | None:
        """
        找到能够到达的物体
        :param obj:
        :param color:
        :return: (object_idx, color_idx, state)
        """
        memory_obs = self.memory_obs
        for x in range(memory_obs.shape[0]):
            for y in range(memory_obs.shape[1]):
                object_idx, color_idx, state = memory_obs[x, y, :]
                if object_idx == OBJECT_TO_IDX[obj] and (color == '' or color_idx == COLOR_TO_IDX[color]):
                    if self.can_move_to(target=(x, y)):
                        return self.get_obs_item((x, y))
        return None

    def find_obs(self, obj: str, color: str = '') -> ObsItem | None:
        for x in range(self.memory_obs.shape[0]):
            for y in range(self.memory_obs.shape[1]):
                object_idx, color_idx, state = self.memory_obs[x, y, :]
                if object_idx == OBJECT_TO_IDX[obj] and (
                        color == '' or color_idx == COLOR_TO_IDX[color]):
                    return self.get_obs_item((x, y))
        return None

    def find_nearest_obs(
            self,
            obj: str,
            color: str = '',
            near_range: (int, int) = (0, 1e6)) -> ObsItem | None:
        """
        找到离自己最近的物体
        :param obj:
        :param color:
        :return: (object_idx, color_idx, state)
        """
        memory_obs = self.memory_obs
        agent_pos = self.agent_pos

        min_distance = 1e6
        min_pos = None

        for x in range(memory_obs.shape[0]):
            for y in range(memory_obs.shape[1]):
                distance = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                if distance < near_range[0] or distance > near_range[1]:
                    continue
                object_idx, color_idx, state = memory_obs[x, y, :]
                if object_idx == OBJECT_TO_IDX[obj] and (color == '' or color_idx == COLOR_TO_IDX[color]):
                    if distance < min_distance:
                        min_distance = distance
                        min_pos = (x, y)
        if min_pos is None:
            return None
        return self.get_obs_item(min_pos)

    def can_move_to(self, target: (int, int)) -> bool:
        path = astar_find_path(obs=self.memory_obs, start=self.agent_pos, target=target)
        return path is not None

    def get_target_direction(self, target: (int, int)) -> int | None:
        """
        获取目标位置相对于自己的方向，如果目标位置就是自己的位置，则返回None
        :param target: 目标位置
        :return:
        """
        try:
            if target is None:
                return None
            agent_pos = self.agent_pos
            if target[0] > agent_pos[0]:
                direction = Directions.right
            elif target[0] < agent_pos[0]:
                direction = Directions.left
            elif target[1] > agent_pos[1]:
                direction = Directions.down
            elif target[1] < agent_pos[1]:
                direction = Directions.up
            else:
                return None
            return direction
        except Exception as e:
            print(target, type(target), e)
            raise e
        return None


# plot score = 1 - (distance / max_size) ** 0.5
# x: distance
# y: score
# max_size: 16 + 16 = 32

class MemoryImageObsWrapper(ObservationWrapper):
    def observation(self, obs):
        return obs['memory_image']
