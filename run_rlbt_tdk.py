import os.path

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
    'TDK16': 'MiniGrid-TwoDoorKeyEnv-16x16-v0',  # 双门
}

ENV_NO = 'TDK16'
env = make_env(ENVS[ENV_NO], render=False)
env = MiniGridSimulation(env)

builder = BTBuilder()

TREE_GROUP = 'TDK'
TREE_NAME = 'G2-T2' # 树名

tree = RLTree(
        root=builder.build_from_file(os.path.join('scripts', TREE_GROUP, TREE_NAME + '.xml')),
        context={
            # 这里放一些环境变量

        }
)

policy = BTPolicy(env=env, tree=tree)

name = f'{ENV_NO}-{TREE_NAME}'

if __name__ == '__main__':
    manager = Manager(code_file=__file__, env=env, debug=True, name=name)
    manager.run_policy(policy=policy, track=False, train=True, episodes=2000)
