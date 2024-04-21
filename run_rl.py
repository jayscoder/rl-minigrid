from manager import *
from common import *
from stable_baselines3 import SAC, PPO, TD3
from gym import spaces
import minigrid
from minigrid.wrappers import DictObservationSpaceWrapper

MODEL_NAME = __file__.split('/')[-1].split('.')[0]

"""
常规的强化学习：没有使用行为树
"""


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """组合节点用到的特征提取器"""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space['image'].shape[0]
        self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
        )

        # # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(in_features=n_flatten + 4, out_features=features_dim), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        features = self.cnn(observations['image'])
        direction = observations['direction']
        if len(direction.shape) > 2: # 有时候会突然多出来一维，不知道为啥
            direction = direction.squeeze(dim=1)
        return self.linear(torch.cat((direction, features), dim=1))


class CustomEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
                {
                    "image"    : env.observation_space["image"],
                    "direction": spaces.Discrete(4),
                }
        )

    def observation(self, observation):
        del observation['missile']
        return observation


ENVS = {
    '1': 'MiniGrid-DoorKey-5x5-v0',
    '2': 'MiniGrid-DoorKey-8x8-v0',
    '3': "MiniGrid-DoorKey-16x16-v0",
}

env = make_env(ENVS['1'], render=False)

# env = CustomEnvWrapper(env)
# env = ImgObsWrapper(env)
env = DictObservationSpaceWrapper(env)
env = ContinuousActionWrapper(env)
# policy = 'CnnPolicy'
policy = 'MultiInputPolicy'

policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        # features_extractor_kwargs=dict(features_dim=128),
)

model = SAC(policy, env, verbose=1, device='cpu', policy_kwargs=policy_kwargs)

if __name__ == '__main__':
    Manager(code_file=__file__, env=env).run_model(train=True, model=model)
