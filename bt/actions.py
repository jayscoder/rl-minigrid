from __future__ import annotations
from bt.base import *
from pybts import *
from common.constants import *


class TurnLeft(BaseBTNode):
    def update(self) -> Status:
        self.put_action(Actions.left)
        return Status.SUCCESS


class TurnRight(BaseBTNode):
    def update(self) -> Status:
        self.put_action(Actions.right)
        return Status.SUCCESS


class MoveForward(BaseBTNode):
    @property
    def direction(self) -> int | None:
        if 'direction' in self.attrs:
            return self.converter.int(self.attrs['direction'])
        return None

    def updater(self) -> typing.Iterator[Status]:
        return self.move_forward(self.direction)


class ManualControl(BaseBTNode):

    def update(self):
        import pygame
        if not pygame.get_init():
            pygame.init()

        event = pygame.event.poll()
        if event.type == pygame.KEYDOWN:
            event_key = pygame.key.name(int(event.key))
            if event_key == "escape":
                self.env.close()
                return

            key_to_action = {
                "left"      : Actions.left,
                "right"     : Actions.right,
                "up"        : Actions.forward,
                "space"     : Actions.toggle,
                "pageup"    : Actions.pickup,
                "pagedown"  : Actions.drop,
                "tab"       : Actions.pickup,
                "left shift": Actions.drop,
                "enter"     : Actions.done,
            }

            if event_key in key_to_action.keys():
                action = key_to_action[event_key]
                self.put_action(action)

        return Status.SUCCESS


class OpenDoor(BaseBTNode):
    """
    目标：打开门
    后置条件：附近的门已经打开
    前置条件：附近1格内有门，且门未打开
    条件为: 附近1格内有门

    color: 门的颜色
    """

    def __init__(self, color: str = '', **kwargs):
        super().__init__(**kwargs)
        self._color = color

    @property
    def color(self) -> str:
        return self.converter.render(self._color)

    def updater(self) -> typing.Iterator[Status]:
        door = self.env.find_nearest_obs(obj='door', color=self.color, near_range=(1, 1))
        if door is None:
            # 旁边1格内没有门
            yield Status.FAILURE
            return
        # 检查门是否打开了
        if door.state == States.open:
            # 门之前已经打开了
            yield Status.FAILURE
            return

        for _ in self.turn_to(door.pos):
            yield Status.RUNNING

        self.put_action(Actions.toggle)
        yield Status.RUNNING
        door = self.env.get_obs_item(door.pos)
        # 检查门是否打开了
        if door.state == States.open:
            yield Status.SUCCESS
        else:
            yield Status.FAILURE
        return


class Pickup(BaseBTNode):
    def updater(self) -> typing.Iterator[Status]:
        self.put_action(Actions.pickup)
        agent_obs = self.env.agent_obs
        yield Status.RUNNING
        if agent_obs == self.env.agent_obs:
            yield Status.FAILURE
            return
        yield Status.SUCCESS


class Drop(BaseBTNode):
    def updater(self) -> typing.Iterator[Status]:
        self.put_action(Actions.drop)
        agent_obs = self.env.agent_obs
        yield Status.RUNNING
        if agent_obs == self.env.agent_obs:
            yield Status.FAILURE
            return
        yield Status.SUCCESS


class Toggle(BaseBTNode):
    def updater(self) -> typing.Iterator[Status]:
        self.put_action(Actions.toggle)
        agent_obs = self.env.agent_obs
        yield Status.RUNNING
        if agent_obs == self.env.agent_obs:
            yield Status.FAILURE
        else:
            yield Status.SUCCESS


class PickUpKey(BaseBTNode):
    """
    拾取钥匙
    color: 钥匙的颜色
    """

    @property
    def color(self) -> str:
        return self.converter.render(self.attrs.get('color', ''))

    def updater(self) -> typing.Iterator[Status]:
        key_obs = self.env.find_nearest_obs(obj='key', color=self.color, near_range=(1, 1))
        if key_obs is None:
            self.put_update_message('找不到钥匙')
            yield Status.FAILURE
            return

        # 先转向
        for _ in self.turn_to(key_obs.pos):
            yield Status.RUNNING
        # 再拾取
        self.put_action(Actions.pickup)
        yield Status.RUNNING

        # 检查钥匙是否拾取成功
        key_obs = self.env.get_obs_item(key_obs.pos)
        if key_obs.obj != Objects.key:
            # 钥匙已拾取
            self.put_update_message('钥匙已拾取')
            yield Status.SUCCESS
        else:
            self.put_update_message('钥匙拾取失败')
            yield Status.FAILURE
        return


class MoveLeft(MoveForward):
    """
    向左移动
    """

    def direction(self) -> int:
        return Directions.left


class MoveUp(MoveForward):
    """
    向上移动
    """

    def direction(self) -> int:
        return Directions.up


class MoveDown(MoveForward):
    """
    向下移动
    """

    def direction(self) -> int:
        return Directions.down


class MoveToPosition(BaseBTNode):
    """
    移动到指定的位置
    """

    def x(self):
        return self.converter.int(self.attrs['x'])

    def y(self):
        return self.converter.int(self.attrs['x'])

    def updater(self) -> typing.Iterator[Status]:
        yield from self.move_to((self.x, self.y))


class MoveToObject(BaseBTNode):
    """
    移动到指定的物体位置
    """

    @property
    def object(self) -> str:
        # 目标位置的物体
        return self.converter.render(self.attrs['object'])

    @property
    def color(self) -> str:
        # 目标位置的物体的颜色
        return self.converter.render(self.attrs.get('color', ''))

    @property
    def nearby(self) -> list[int]:
        # 是否只移动到目标区域旁边n格内
        return self.converter.int_list(self.attrs.get('nearby', 0))

    def updater(self) -> typing.Iterator[Status]:
        object_obs = self.env.find_nearest_obs(obj=self.object, color=self.color)
        self.put_update_message(f'MoveToObject object={object_obs} agent_obs={self.env.agent_obs}')
        if object_obs is None:
            # 找不到目标位置
            self.put_update_message('找不到目标位置')
            yield Status.FAILURE
            return
        yield from self.move_to(object_obs.pos, nearby=self.nearby)


class ApproachObject(MoveToObject):
    """
    移动到指定的目标位置附近1格
    """

    @property
    def nearby(self) -> list[int]:
        return [1, 1]


class MoveToGoal(MoveToObject):
    """
    移动到目标位置
    """

    @property
    def object(self) -> str:
        return 'goal'


class ApproachKey(ApproachObject):
    """
    移动到钥匙位置附近
    """

    @property
    def object(self) -> str:
        return 'key'


class ApproachDoor(ApproachObject):
    """
    移动到门位置
    """

    @property
    def object(self) -> str:
        return 'door'


class ExploreUnseen(BaseBTNode):
    """
    探索未知区域
    使用AStar算法从当前位置开始探索未知区域
    """

    def updater(self) -> typing.Iterator[Status]:
        target_obs = self.env.find_can_reach_obs(obj='unseen')
        if target_obs is None:
            # 没有未知区域了
            self.put_update_message('没有未知区域了')
            yield Status.FAILURE
            return
        # if target_obs is None:
        #     # 没有未知区域了
        #     unseen_obs = self.env.find_nearest_obs(obj='unseen')
        #     if unseen_obs is not None:
        #         yield from self.move_to(unseen_obs.pos)
        #         return
        #     return Status.FAILURE
        for status in self.move_to(target_obs.pos):
            yield status
            target_obs = self.env.get_obs_item(target_obs.pos)
            if target_obs.obj != Objects.unseen:
                self.put_update_message('目标未知区域已经达到')
                yield Status.SUCCESS
                return


class TurnToObject(BaseBTNode):
    """
    转向指定的物体
    """

    @property
    def object(self) -> str:
        # 目标位置的物体
        return self.converter.render(self.attrs['object'])

    @property
    def color(self) -> str:
        # 目标位置的物体的颜色
        return self.converter.render(self.attrs.get('color', ''))

    def updater(self) -> typing.Iterator[Status]:
        target_obs = self.env.find_nearest_obs(obj=self.object, color=self.color)

        if target_obs is None:
            # 找不到目标位置
            yield Status.FAILURE
            return
        yield from self.turn_to(target_obs.pos)


class TurnToKey(TurnToObject):
    """
    转向钥匙
    """

    @property
    def object(self) -> str:
        return 'key'


class TurnToDoor(TurnToObject):
    """
    转向门
    """

    @property
    def object(self) -> str:
        return 'door'


class ToggleTargetColor(BaseBTNode):
    """
    切换目标颜色，并存储到黑板变量里
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = 0

    @property
    def colors(self) -> list[str]:
        return self.converter.str_list(self.attrs['colors'].split(','))

    def update(self) -> Status:
        self.context['color'] = self.colors[self.index]
        self.index = (self.index + 1) % len(self.colors)
        return Status.SUCCESS


