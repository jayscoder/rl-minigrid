import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import torch.nn as nn
from tqdm import tqdm
import utils
MODEL_NAME = 'PPO-1'

N = 100


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
        )
        # self.image_conv = nn.Sequential(
        #         nn.Conv2d(3, 16, (2, 2)),
        #         nn.ReLU(),
        #         nn.MaxPool2d((2, 2)),
        #         nn.Conv2d(16, 32, (2, 2)),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 64, (2, 2)),
        #         nn.ReLU()
        # )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make("MiniGrid-DoorKey-16x16-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f'logs/{MODEL_NAME}', device='cpu')
model.learn(2e5, progress_bar=True)

model.save(MODEL_NAME)
del model

model = PPO.load(MODEL_NAME)

# writer.add_scalar("losses/explained_variance", explained_var, global_step)
# writer = SummaryWriter(f"runs/{MODEL_NAME}")
# writer.close()
utils.evaluate(name=MODEL_NAME, model=model, env=env)
