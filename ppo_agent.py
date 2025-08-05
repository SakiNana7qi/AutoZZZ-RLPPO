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
        is_stage_victory_for_gae=False,
    ):
        gae_done = done or is_stage_victory_for_gae
        # 将张量创建在 Pinned Memory 中，为快速异步传输做准备
        self.memory["image_states"].append(
            torch.tensor(state["image"], dtype=torch.uint8).pin_memory()
        )
        self.memory["action_history_states"].append(
            torch.tensor(state["action_history"], dtype=torch.int64).pin_memory()
        )
        self.memory["actions"].append(
            torch.tensor(action, dtype=torch.int64).pin_memory()
        )
        self.memory["log_probs"].append(log_prob.detach().cpu().pin_memory())
        self.memory["rewards"].append(
            torch.tensor(reward, dtype=torch.bfloat16).pin_memory()
        )
        self.memory["dones"].append(
            torch.tensor(gae_done, dtype=torch.bfloat16).pin_memory()
        )
        self.memory["values"].append(value.detach().cpu().pin_memory())
        self.memory["action_masks"].append(
            torch.tensor(action_mask, dtype=torch.bool).pin_memory()
        )

    def select_action(self, state, action_mask):

        image_tensor = (
            torch.tensor(state["image"], dtype=torch.bfloat16).unsqueeze(0) / 255.0
        ).to(  # [K, H, W] -> [1, K, H, W]
            self.device
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

    def learn(self, last_value):
        # 从内存中读取数据
        image_states = (
            torch.stack(self.memory["image_states"]).to(
                self.device, dtype=torch.bfloat16, non_blocking=True
            )
            / 255.0
        )

        action_history_states = torch.stack(self.memory["action_history_states"]).to(
            self.device, non_blocking=True
        )
        actions = torch.stack(self.memory["actions"]).to(self.device, non_blocking=True)
        old_log_probs = torch.stack(self.memory["log_probs"]).to(
            self.device, non_blocking=True
        )
        rewards = torch.stack(self.memory["rewards"]).to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        )
        dones = torch.stack(self.memory["dones"]).to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        )
        values = (
            torch.stack(self.memory["values"])
            .squeeze()
            .to(self.device, dtype=torch.bfloat16, non_blocking=True)
        )
        action_masks = torch.stack(self.memory["action_masks"]).to(
            self.device, non_blocking=True
        )

        # 用 GAE 计算 Advantage 和 Returns
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0
        last_value = last_value.to(self.device, dtype=torch.bfloat16).squeeze()

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            # if dones[t] ，那么 last_value , last_advantage 应该都是 0
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage * mask
            last_advantage = advantages[t].item()
            last_value = values[t]

        original_advantages_mean = advantages.mean().item()
        original_advantages_std = advantages.std().item()

        returns = advantages + values

        returns_mean = returns.mean().item()
        returns_std = returns.std().item()
        values_mean = values.mean().item()
        values_std = values.std().item()

        # 标准化 advantage, returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._clear_memory()  # 计算完 GAE，memory 就可以 clear 了

        num_samples = len(image_states)
        indices = np.arange(num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_ratio = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            print(f"ppoepochs {epoch}/{self.ppo_epochs}")
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_images = image_states[batch_idx]
                batch_action_history = action_history_states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_action_masks = action_masks[batch_idx]

                new_dist, new_values = self.model(
                    batch_images.to(self.device),
                    batch_action_history.to(self.device),
                    batch_action_masks.to(self.device),
                )
                new_values = new_values.squeeze()

                with torch.no_grad():
                    batch_advantages = advantages[batch_idx]
                    batch_returns = returns[batch_idx]

                    value_pred_clipped = values[batch_idx] + torch.clamp(
                        new_values.detach() - values[batch_idx],
                        -self.policy_clip,
                        self.policy_clip,
                    )

                # actor loss
                # 要用 Clipped Surrogate Objective 来防止爆炸
                # r = exp(new_log_prob - old_log_prob)
                new_log_probs = new_dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss (MSE)
                # 还有 value clipping

                loss_unclipped = F.mse_loss(new_values, batch_returns)
                loss_clipped = F.mse_loss(value_pred_clipped, batch_returns)
                critic_loss = 0.5 * torch.max(loss_unclipped, loss_clipped)

                # 计算熵损失，防止摆烂
                entropy_loss = -new_dist.entropy().mean()

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
                total_ratio += ratio.mean().item()
                num_updates += 1

        # 计算 Explained Variance
        y_pred = values
        y_true = returns
        var_y = torch.var(y_true)
        explained_variance = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        explained_variance = explained_variance.item()

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_ratio = total_ratio / num_updates

        return {
            "Loss/policy_loss": avg_policy_loss,
            "Loss/value_loss": avg_value_loss,
            "Loss/entropy": avg_entropy_loss,
            "Charts/explained_variance": explained_variance,
            "Charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "Charts/advantages_mean": original_advantages_mean,
            "Charts/advantages_std": original_advantages_std,
            "Charts/returns_mean": returns_mean,
            "Charts/returns_std": returns_std,
            "Charts/values_mean": values_mean,
            "Charts/values_std": values_std,
            "Charts/policy_ratio": avg_ratio,
            "Charts/log_prob_mean": old_log_probs.mean().item(),
        }
