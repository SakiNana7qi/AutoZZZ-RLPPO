# model.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import config


class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions, action_history_len):
        super(ActorCritic, self).__init__()

        # 图像处理 - 经典 CNN
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

        # 历史动作处理 - 迷你 MLP
        self.action_embedding = nn.Embedding(
            num_actions + 1, 32
        )  # +1 是为了处理 padding 值-1
        self.action_linear = nn.Sequential(
            nn.Linear(action_history_len * 32, 128), nn.ReLU()
        )

        combined_features_size = conv_out_size + 128  # 合并

        # actor 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(combined_features_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        # critic 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(combined_features_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(shape)
        return int(np.prod(o.size()))

    def forward(self, image_input, action_history_input, action_mask=None):
        # gym 爷爷说过，输入的 x 一定要是 [batch, height, width, channels]
        # 但是我们的 pytorch 宝宝要的是 [batch, channels, height, width]
        # 但我们已经在 agent 和 run.py 转好了

        # 图像
        image_features = self.conv(image_input).reshape(image_input.size(0), -1)

        # 历史动作
        # 把 padding -1 映射成嵌入表 size-1
        action_history_mapped = action_history_input.clone()
        action_history_mapped[action_history_mapped == -1] = (
            self.action_embedding.num_embeddings - 1
        )

        action_embedded = self.action_embedding(action_history_mapped)
        # 展平 . [batch, history_len, embed_dim] -> [batch, history_len * embed_dim]
        action_features = self.action_linear(
            action_embedded.view(action_embedded.size(0), -1)
        )

        # 合并
        combined_features = torch.cat([image_features, action_features], dim=1)

        action_logits = self.actor(combined_features)

        if action_mask is not None:
            action_logits[~action_mask] = -1e8  # action mask ，-inf 防止死灰复燃

        action_dist = Categorical(logits=action_logits)
        value = self.critic(combined_features)

        return action_dist, value
