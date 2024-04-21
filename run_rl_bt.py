import pybts
from bt import *
from manager import *
from common import *
import minigrid
from rl import RLTree

"""
在这个例子里，如果给机器人加上一个捣乱的前置任务，即在门打开的时候需要去做一件别的事情，
那么它原来的逻辑就没办法正常运行了，因为它需要从头开始执行
"""

# pybts.logging.level = pybts.logging.Level.DEBUG


ENVS = {
    '1': 'MiniGrid-DoorKey-5x5-v0',
    '2': 'MiniGrid-DoorKey-8x8-v0',
    '3': "MiniGrid-DoorKey-16x16-v0",
}

env = make_env(ENVS['3'], render=False)
env = MiniGridSimulation(env)

builder = BTBuilder()

TREE = {
    '1': 'scripts/DoorKey-RLSequence-SAC.xml',
    '2': 'scripts/DoorKey-RLSequence-MakeTrouble-SAC.xml'
}

tree = RLTree(
        root=builder.build_from_file(TREE['2']),
)

policy = BTPolicy(env=env, tree=tree)

if __name__ == '__main__':
    manager = Manager(code_file=__file__, env=env, debug=True)
    manager.run_policy(policy=policy, track=True, train=True)