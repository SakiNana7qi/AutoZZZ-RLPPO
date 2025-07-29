# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class PPOAgent:
    def __init__(
        self,
        model,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        ppo_epochs=10,
        batch_size=64,
        vf_coef=0.5,
        ent_coef=0.01,
        device="cuda",
    ):

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # PPO超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef  # value function loss coefficient
        self.ent_coef = ent_coef  # entropy bonus coefficient

        # 内存（存要学的东西）
        self.memory = {
            "image_states": [],
            "action_history_states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "action_masks": [],
        }

    def _clear_memory(self):
        for key in self.memory:
            self.memory[key].clear()

    def store_transition(
        self,
        state,
        action,
        log_prob,
        reward,
        done,
        value,
        action_mask,
        is_stage_victory=False,
    ):
        gae_done = (
            done or is_stage_victory
        )  # 转阶段伪造 done 信号，使得价值回溯能够正常运行
        self.memory["image_states"].append(
            torch.tensor(state["image"], dtype=torch.uint8)
        )
        self.memory["action_history_states"].append(
            torch.tensor(state["action_history"], dtype=torch.int64)
        )
        self.memory["actions"].append(torch.tensor(action, dtype=torch.int64))
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(torch.tensor(reward, dtype=torch.bfloat16))
        self.memory["dones"].append(torch.tensor(gae_done, dtype=torch.bfloat16))
        self.memory["values"].append(value)
        self.memory["action_masks"].append(torch.tensor(action_mask, dtype=torch.bool))

    def select_action(self, state, action_mask):

        image_tensor = (
            torch.tensor(state["image"], dtype=torch.bfloat16)
            .unsqueeze(0)  # [K, H, W] -> [1, K, H, W]
            .to(self.device)
        )  # state["image"] 是 [K, H, W]
        # 将 numpy state 转换为 torch tensor
        action_history_tensor = (
            torch.tensor(state["action_history"], dtype=torch.int64)
            .unsqueeze(0)
            .to(self.device)
        )

        action_mask_tensor = (
            torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            dist, value = self.model(
                image_tensor, action_history_tensor, action_mask_tensor
            )

        action = dist.sample()  # 采样一个动作
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.cpu(), value.cpu()

    def learn(self):
        # 从内存中读取数据
        image_states = torch.stack(
            [s.to(self.device) for s in self.memory["image_states"]]
        ).to(
            dtype=torch.bfloat16
        )  # [N, K, H, W]
        action_history_states = torch.stack(
            [s.to(self.device) for s in self.memory["action_history_states"]]
        )
        actions = torch.stack([s.to(self.device) for s in self.memory["actions"]])
        old_log_probs = torch.stack(
            [s.to(self.device) for s in self.memory["log_probs"]]
        )
        rewards = [r.to(self.device) for r in self.memory["rewards"]]
        dones = [d.to(self.device) for d in self.memory["dones"]]
        values = torch.stack(
            [v.to(self.device) for v in self.memory["values"]]
        ).squeeze()
        action_masks = torch.stack(
            [m.to(self.device) for m in self.memory["action_masks"]]
        )
        # 这里为什么要在里面 todevice 而不是外面？
        # 因为如果里面两个东西的 device 不一样，一起 todevice 就炸了

        # 用 GAE 计算 Advantage 和 Returns
        advantages = torch.zeros(len(rewards), dtype=torch.bfloat16).to(self.device)
        last_advantage = 0
        last_value = (
            values[-1] if len(values) > 0 else torch.tensor(0.0).to(self.device)
        )  # Bootstrapping

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            # if dones[t] ，那么 last_value , last_advantage 应该都是 0

            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            advantages[t] = last_advantage = (
                delta + self.gamma * self.gae_lambda * last_advantage * mask
            )
            last_value = values[t]

        returns = advantages + values

        # standardized
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # epochjs
        # image_states = image_states.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W] # Obsolete

        num_samples = len(image_states)
        indices = np.arange(num_samples)

        # losses

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            print(f"ppoepochs {_}/{self.ppo_epochs}")
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # 获取批数据
                batch_image_states = image_states[batch_indices]
                batch_action_history_states = action_history_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]

                # 用旧的 states 来算新的动作分布和价值
                new_dist, new_values = self.model(
                    batch_image_states, batch_action_history_states, batch_action_masks
                )
                new_values = new_values.squeeze()

                # r = exp(new_log_prob - old_log_prob)
                new_log_probs = new_dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # actor loss
                # 要用 Clipped Surrogate Objective 来防止爆炸
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss (MSE)
                critic_loss_unclipped = F.mse_loss(new_values, batch_returns)
                # with value clipping
                values_clipped = values[batch_indices] + torch.clamp(
                    new_values - values[batch_indices],
                    -self.policy_clip,
                    self.policy_clip,
                )
                critic_loss_clipped = F.mse_loss(values_clipped, batch_returns)
                critic_loss = 0.5 * torch.max(
                    critic_loss_unclipped, critic_loss_clipped
                )

                # 计算熵损失，防止摆烂
                entropy_loss = -new_dist.entropy().mean()

                # total_loss
                total_loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    + self.ent_coef * entropy_loss
                )

                # 反向传播，优化
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 0.5
                )  # 梯度裁剪，防止梯度爆炸
                self.optimizer.step()

                total_policy_loss += actor_loss.item()
                total_value_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1

        self._clear_memory()

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
        }
