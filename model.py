# model.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import config


class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ActorCritic, self).__init__()

        # 经典 CNN
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy_input = torch.zeros(
            1, input_channels, config.IMG_HEIGHT, config.IMG_WIDTH
        )
        conv_out_size = self._get_conv_out(dummy_input)

        # actor 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

        # critic 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(shape)
        return int(np.prod(o.size()))

    def forward(self, x, action_mask=None):
        # gym 爷爷说过，输入的 x 一定要是 [batch, height, width, channels]
        # 但是我们的 pytorch 宝宝要的是 [batch, channels, height, width]
        # 但我们已经在 agent 和 run.py 转好了
        conv_out = self.conv(x).reshape(x.size(0), -1)  # 展平

        action_logits = self.actor(conv_out)

        if action_mask is not None:
            action_logits[~action_mask] = -1e8  # 将无效动作的 logits 设置为 -inf

        action_dist = Categorical(logits=action_logits)

        value = self.critic(conv_out)

        return action_dist, value
