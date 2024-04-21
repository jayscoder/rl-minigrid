import pybts
from bt import *
from manager import *
from common import *
import minigrid
from rl import RLTree

"""
单纯的BT
在这里执行不同的脚本以实现不同的行为树策略

在这个例子里，如果给机器人加上一个捣乱的前置任务，即在门打开的时候需要去做一件别的事情，
那么它原来的逻辑就没办法正常运行了，因为它需要从头开始执行
"""

# pybts.logging.level = pybts.logging.Level.DEBUG

builder = BTBuilder()

ENVS = {
    '1': 'MiniGrid-DoorKey-5x5-v0',
    '2': 'MiniGrid-DoorKey-8x8-v0',
    '3': "MiniGrid-DoorKey-16x16-v0",
}

env = make_env(ENVS['3'], render=True)
env = MiniGridSimulation(env)

TREE = {
    '1': 'scripts/DoorKey-Simplified.xml',
    '2': 'scripts/DoorKey-Simplified-MakeTrouble.xml',
    '3': 'scripts/DoorKey-Full.xml',
    '4': 'scripts/DoorKey-Full-MakeTrouble.xml'
}

tree = RLTree(
        root=builder.build_from_file(TREE['1']),
)

policy = BTPolicy(env=env, tree=tree)

if __name__ == '__main__':
    Manager(code_file=__file__, env=env).run_policy(policy=policy, track=True)
