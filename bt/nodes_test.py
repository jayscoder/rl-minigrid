from bt.base import *
import typing


class MakeTrouble(BaseBTNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trouble_count = 0

    def reset(self):
        super().reset()
        self.trouble_count = 0

    def updater(self) -> typing.Iterator[Status]:
        if self.trouble_count > 0:
            yield Status.FAILURE
            return

        door_obs = self.env.find_nearest_obs(obj='door')
        if door_obs is None:
            yield Status.FAILURE
            return

        if door_obs.state == States.open:
            self.trouble_count += 1
            yield from self.move_to((0, 0))
            return

        yield Status.FAILURE
        return

