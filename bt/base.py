from __future__ import annotations

import queue
import pybts
from abc import ABC
from common.wrappers import MiniGridSimulation
from pybts import Status
from common.utils import *
import typing
from common.constants import *


class BaseBTNode(pybts.Action, ABC):
    """
    BT Policy Base Class Node
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_messages = queue.Queue(maxsize=20)  # update过程中的message

    @property
    def env(self) -> MiniGridSimulation:
        return self.context['env']

    @property
    def debug(self):
        return self.context['debug']

    def put_update_message(self, msg: str):
        if not self.debug:
            return
        if self.update_messages.full():
            self.update_messages.get_nowait()
        msg = f"{self.debug_info['tick_count']}: {msg}"
        self.update_messages.put_nowait(msg)

    def to_data(self):
        from pybts import utility
        return {
            **super().to_data(),
            'update_messages': utility.read_queue_without_destroying(self.update_messages)
        }

    def put_action(self, action):
        self.env.put_action(action)

    def move_to(self, target: (int, int), nearby: int | tuple[float, float] | list[float] = 0) -> typing.Iterator[
        Status]:
        """
        用A*算法移动到目标位置
        :param target: 目标位置
        :param nearby: 移动到目标位置附近曼哈顿距离内
        :return:
        """
        if isinstance(nearby, int):
            nearby = [0, nearby]
        if len(nearby) == 1:
            nearby = [0, nearby[0]]

        if nearby[0] <= manhattan_distance(self.env.agent_pos, target) <= nearby[1]:
            # 已经在目标位置
            yield Status.SUCCESS
            return

        stopped_count = 0
        while manhattan_distance(self.env.agent_pos, target) > nearby[1] and stopped_count < 3:

            path = astar_find_path(obs=self.env.memory_obs, start=self.env.agent_pos, target=target)

            if path is None:
                self.put_update_message(
                        f'无法生成路径 move_to {target} agent_obs={self.env.agent_obs} agent_pos={self.env.agent_pos} target_obs={self.env.get_obs_item(target)} stopped_count={stopped_count} distance={manhattan_distance(self.env.agent_pos, target)} nearby={nearby}')
                yield Status.FAILURE
                return

            stopped = False
            for i, p in enumerate(path):
                if stopped:
                    break
                if i == 0:
                    # 跳过第一个起点
                    continue

                if manhattan_distance(self.env.agent_pos, target) <= nearby[1]:
                    break

                for status in self.move_forward(target=p):
                    if status == Status.FAILURE:
                        stopped = True
                        stopped_count += 1
                        break
                    yield Status.RUNNING

        if nearby[0] <= manhattan_distance(self.env.agent_pos, target) <= nearby[1]:
            yield Status.SUCCESS
            return

        yield Status.FAILURE
        return

    def turn_to(self, target: int | (int, int) | None) -> typing.Iterator[Status]:
        """
        转向到目标或目标方向
        :param target: 目标方向或目标位置
        :return:
        """
        if target is None:
            return

        if isinstance(target, ObsItem):
            direction = self.env.get_target_direction(target.pos)
        elif isinstance(target, tuple):
            direction = self.env.get_target_direction(target)
        else:
            direction = int(target)

        if direction is None:
            return

        # print(f'agent_dir={DIRECTIONS_OPTIONS[self.env.agent_dir]} direction={DIRECTIONS_OPTIONS[direction]}')

        if direction == self.env.agent_dir:
            return

        # direction { 0: ">", 1: "V", 2: "<", 3: "^" }
        if direction == (self.env.agent_dir + 1) % 4:
            # 右转
            self.env.put_action(Actions.right)
        elif direction == (self.env.agent_dir + 2) % 4:
            # 向后转
            self.env.put_action(Actions.right)
            yield Status.RUNNING
            self.env.put_action(Actions.right)
        else:
            self.env.put_action(Actions.left)

        yield Status.RUNNING
        if self.env.agent_dir == direction:
            yield Status.SUCCESS
        else:
            yield Status.FAILURE
        return

    def move_forward(self, target: int | (int, int) | None = None) -> typing.Iterator[Status]:
        """
        向前移动step步, 如果direction不为空，则按照direction的方向移动，否则按照agent_dir的方向移动
        :param target: 方向或目标位置
        :return:
        """
        if target is None:
            direction = self.env.agent_dir
        elif isinstance(target, int):
            direction = target
        else:
            direction = self.env.get_target_direction(target)
            if direction is None:
                # '已经在目标位置, 不需要移动'
                self.put_update_message(f'move_forward direction is None')
                yield Status.SUCCESS
                return

        for _ in self.turn_to(target=direction):
            yield Status.RUNNING
        self.env.put_action(Actions.forward)
        agent_pos = self.env.agent_pos
        yield Status.RUNNING
        if self.env.agent_pos == agent_pos:
            self.put_update_message(f'move_forward 说明这次向前移动失败了')
            # 说明这次向前移动失败了
            yield Status.FAILURE
            return
        yield Status.SUCCESS
