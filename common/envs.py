from __future__ import annotations

import random

from minigrid.envs.doorkey import DoorKeyEnv

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register

class RandomGoalDoorKeyEnv(DoorKeyEnv):
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        # Place a goal in the bottom-right corner

        self.place_obj(obj=Goal(), top=(0, 0), size=(width, height))

        self.mission = "use the key to open the door and then get to the goal"


class TwoDoorKeyEnv(RandomGoalDoorKeyEnv):
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdxs = []
        while True:
            splitIdxs = [self._rand_int(1, width - 2), self._rand_int(1, width - 2)]
            if abs(splitIdxs[0] - splitIdxs[1]) > 2:
                break
        colors = ['red', 'yellow']
        random.shuffle(colors)

        self.place_agent(size=(splitIdxs[0], height))
        for i, splitIdx in enumerate(splitIdxs):
            self.grid.vert_wall(splitIdx, 0)
            # Place the agent at a random position and orientation
            # on the left side of the splitting wall

            # Place a door in the wall
            doorIdx = self._rand_int(1, width - 2)
            self.put_obj(Door(colors[i], is_locked=True), splitIdx, doorIdx)

            # Place a yellow key on the left side
            self.place_obj(obj=Key(colors[i]), top=(0, 0), size=(splitIdx, height))

        # Place a goal in the random position
        self.place_obj(obj=Goal(), top=(splitIdxs[1], 0), size=(width - splitIdxs[1] - 1, height))

        self.mission = "use the key to open the door and then get to the goal"


register(
        id="MiniGrid-RandomGoalDoorKeyEnv-5x5-v0",
        entry_point="common.envs:RandomGoalDoorKeyEnv",
        kwargs={ "size": 5 },
)

register(
        id="MiniGrid-RandomGoalDoorKeyEnv-16x16-v0",
        entry_point="common.envs:RandomGoalDoorKeyEnv",
        kwargs={ "size": 16 },
)

register(
        id="MiniGrid-TwoDoorKeyEnv-16x16-v0",
        entry_point="common.envs:TwoDoorKeyEnv",
        kwargs={ "size": 16 },
)
