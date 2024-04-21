import gym
import pybts
from common.wrappers import MiniGridSimulation
from rl import RLBaseNode
from rl import RLTree

class BTPolicy:
    def __init__(self,
                 tree: RLTree,
                 env: MiniGridSimulation,
                 ):
        self.env = env
        self.tree: RLTree = tree
        self.tree.context.update({
            'step'   : env.step_count,
            'env'    : env,
            'episode': env.episode
        })

    def reset(self):
        self.tree.reset()

    def take_action(self):
        # 更新时间
        # 在tick之前更新时间、agent信息，方便后面使用
        self.tree.context['step'] = self.env.step_count
        self.tree.context['episode'] = self.env.episode

        if self.env.done:
            self.terminate()
            return

        self.tree.tick()
        # 收集所有节点的行为，并放到自己的行为库里
        for node in self.tree.root.iterate():
            if isinstance(node, pybts.Action):
                while not node.actions.empty():
                    action = node.actions.get_nowait()
                    self.env.put_action(action)

    def terminate(self):
        # 在最后结束的时候强制更新一下每个强化学习节点，方便触发强化学习节点
        for node in self.tree.root.iterate():
            if isinstance(node, RLBaseNode) and isinstance(node, pybts.Node):
                node.update()
