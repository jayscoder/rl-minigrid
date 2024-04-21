from __future__ import annotations

import pybts

from bt.base import *
from pybts import *
from common.constants import *

class IsObjectFound(BaseBTNode, pybts.Condition):
    """
    检查是否发现指定的物体
    如果物体颜色为空，则只检查物体类型

    如果物体被自己拾取，则物体的位置和自己的位置相同
    """

    @property
    def object(self) -> str:
        # 目标位置的物体
        return self.converter.render(self.attrs['object'])

    @property
    def color(self) -> str:
        # 目标位置的物体的颜色
        return self.converter.render(self.attrs.get('color', ''))

    def update(self) -> Status:
        obs = self.env.find_obs(obj=self.object, color=self.color)
        if obs is not None:
            return Status.SUCCESS
        return Status.FAILURE


class IsUnseenFound(IsObjectFound):
    """
    是否发现未知区域
    """

    @property
    def object(self):
        return 'unseen'


class IsGoalFound(IsObjectFound):
    """
    是否发现目标
    """

    @property
    def object(self):
        return 'goal'


class IsDoorFound(IsObjectFound):
    """
    检查是否发现指定颜色的门
    """

    @property
    def object(self):
        return 'door'


class IsKeyFound(IsObjectFound):
    """
    检查是否发现指定颜色的钥匙
    """

    @property
    def object(self):
        return 'key'


class IsBallFound(IsObjectFound):
    """
    检查是否发现指定颜色的球
    """

    @property
    def object(self):
        return 'ball'


class IsBoxFound(IsObjectFound):
    """
    检查是否发现指定颜色的箱子
    """

    @property
    def object(self):
        return 'box'


class CanMoveToObject(IsObjectFound, pybts.Condition):
    """
    检查是否能到达目标位置
    """

    def update(self) -> Status:
        obs = self.env.find_can_reach_obs(obj=self.object, color=self.color)
        if obs is not None:
            return Status.SUCCESS
        return Status.FAILURE


class CanMoveToGoal(CanMoveToObject):
    """
    检查是否能到达目标位置
    """

    @property
    def object(self):
        return 'goal'


class CanMoveToUnseen(CanMoveToObject):
    """
    检查是否能到达未知物体
    """

    @property
    def object(self):
        return 'unseen'


class CanApproachDoor(CanMoveToObject):
    """
    检查是否能接近门
    """

    @property
    def object(self):
        return 'door'


class CanApproachKey(CanMoveToObject):
    """
    检查是否能接近钥匙
    """

    @property
    def object(self):
        return 'key'

    def condition(self):
        return IsKeyFound(color=self.color)


class IsReachObject(IsObjectFound, pybts.Condition):
    """
    检查是否到达指定的物体
    如果物体颜色为空，则只检查物体类型

    如果物体被自己拾取，则物体的位置和自己的位置相同
    """

    def update(self) -> Status:
        agent_obs = self.env.agent_obs
        if agent_obs is None:
            return Status.FAILURE

        if agent_obs.obj == Objects[self.object] and (self.color == '' or agent_obs.color == Colors[self.color]):
            return Status.SUCCESS

        return Status.FAILURE


class IsReachGoal(IsReachObject):
    """
    检查是否到达目标位置
    """

    @property
    def object(self):
        return 'goal'


class IsNearObject(IsObjectFound, pybts.Condition):
    """
    检查自己是否在目标物体位置附近，如果distance为0，则检查是否在目标位置
    """

    @property
    def distance(self) -> int:
        return self.converter.int(self.attrs.get('distance', 1))

    def update(self) -> Status:
        find_obj = self.env.find_nearest_obs(obj=self.object, color=self.color,
                                             near_range=(self.distance, self.distance))
        if find_obj is None:
            return Status.FAILURE
        return Status.SUCCESS


class IsNearGoal(IsNearObject):
    """
    检查自己是否在目标位置
    """

    @property
    def object(self) -> str:
        return 'goal'


class IsNearDoor(IsNearObject):
    """
    检查自己是否在门前
    """

    @property
    def object(self) -> str:
        return 'door'


class IsNearKey(IsNearObject):
    """
    检查自己是否在钥匙附近
    """

    @property
    def object(self) -> str:
        return 'key'


class IsNearBall(IsNearObject):
    """
    检查自己是否在球附近
    """

    @property
    def object(self) -> str:
        return 'ball'


class IsNearBox(IsNearObject):
    """
    检查自己是否在箱子附近
    """

    @property
    def object(self) -> str:
        return 'box'


class IsKeyHeld(IsReachObject):
    """
    检查自己是否持有钥匙
    """

    @property
    def object(self) -> str:
        return 'key'


class IsObjectInFront(IsObjectFound, pybts.Condition):
    """
    检查自己是否在物体正前方
    """

    def update(self) -> Status:
        front_pos = self.env.front_pos
        find_obj = self.env.get_obs_item(front_pos)
        if find_obj is None:
            return Status.FAILURE
        if find_obj.obj == Objects[self.object] and (self.color == '' or find_obj.color == Colors[self.color]):
            return Status.SUCCESS
        return Status.FAILURE


class IsKeyInFront(IsObjectInFront):
    """
    钥匙是否在正前方
    """

    @property
    def object(self) -> str:
        return 'key'


class IsDoorInFront(IsObjectInFront):
    """
    门是否在正前方
    """

    @property
    def object(self) -> str:
        return 'door'


class IsDoorOpen(IsDoorFound, pybts.Condition):
    @property
    def object(self) -> str:
        return 'door'

    def update(self) -> Status:
        # 找到离自己最近的门
        door = self.env.find_nearest_obs(obj='door', color=self.color)
        if door is None:
            return Status.FAILURE
        # 检查门是否打开了
        if door.state == States.open:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsDoorClosed(IsDoorFound, pybts.Condition):

    def update(self) -> Status:
        # 找到离自己最近的门
        door = self.env.find_nearest_obs(obj='door', color=self.color)
        if door is None:
            return Status.FAILURE
        # 检查门是否打开了
        if door.state == States.closed:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsDoorLocked(IsDoorFound, pybts.Condition):
    def update(self) -> Status:
        # 找到离自己最近的门
        door = self.env.find_nearest_obs(obj='door', color=self.color)
        if door is None:
            return Status.FAILURE
        # 检查门是否打开了
        if door.state == States.locked:
            return Status.SUCCESS
        else:
            return Status.FAILURE
