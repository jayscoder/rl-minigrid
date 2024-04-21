from bt import *
from manager import *
from common import *
from rl import RLTree

"""
在这个例子里，如果给机器人加上一个捣乱的前置任务，即在门打开的时候需要去做一件别的事情，
那么它原来的逻辑就没办法正常运行了，因为它需要从头开始执行
"""

# pybts.logging.level = pybts.logging.Level.DEBUG


ENVS = {
    'DK5'  : 'MiniGrid-DoorKey-5x5-v0',
    'DK8'  : 'MiniGrid-DoorKey-8x8-v0',
    'DK16' : "MiniGrid-DoorKey-16x16-v0",
    'RDK5' : 'MiniGrid-RandomGoalDoorKeyEnv-5x5-v0',
    'RDK16': 'MiniGrid-RandomGoalDoorKeyEnv-16x16-v0',
    'TDK16': 'MiniGrid-TwoDoorKeyEnv-16x16-v0',
}

ENV_NO = 'TDK16'
env = make_env(ENVS[ENV_NO], render=False)
env = MiniGridSimulation(env)

builder = BTBuilder()

TREE = {
    '1'    : 'scripts/DoorKey-RLSequence-SAC.xml',
    '2'    : 'scripts/DoorKey-RLSequence-MakeTrouble-SAC.xml',
    '3'    : 'scripts/DoorKey-RLSwitcher-SAC-Basic.xml',
    '4'    : 'scripts/DoorKey-RLSwitcher-SAC-经验填充.xml',
    '5'    : 'scripts/DoorKey-RLSwitcher-SAC-无经验填充.xml',
    'G5-T1': 'scripts/G5-T1.xml',
    'G5-T2': 'scripts/G5-T2.xml',
    'G5-T3': 'scripts/G5-T3.xml',
}

TREE_NO = 'G5-T3'  # 采用的行为树的序号
TREE_NAME = TREE[TREE_NO].split('/')[-1].split('.')[0]
tree = RLTree(
        root=builder.build_from_file(TREE[TREE_NO]),
        context={
            # 这里放一些环境变量

        }
)

policy = BTPolicy(env=env, tree=tree)

name = f'{ENV_NO}-{TREE_NO}'

if __name__ == '__main__':
    manager = Manager(code_file=__file__, env=env, debug=True, name=name, version='16x16')
    manager.run_policy(policy=policy, track=True, train=True, episodes=5000)
