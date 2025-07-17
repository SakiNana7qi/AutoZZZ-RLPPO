# run.py 主程序

import torch
import numpy as np
import time
from zzzenv import ZZZEnv, test_observation_capture
from model import ActorCritic
from ppo_agent import PPOAgent
import config
import os
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from myutils import get_human_time


def main():
    log_dir = f"log/ppo_zzz_{get_human_time()}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")  # tensorboard --logdir=log
    recent_rewards = []

    print("开始PPO训练...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    env = ZZZEnv()
    env.start_debug_renderer()

    model = ActorCritic(
        input_channels=3,
        num_actions=config.N_ACTIONS,
        action_history_len=config.ACTION_HISTORY_LEN,
    )
    model.to(device=device, dtype=torch.bfloat16)

    agent = PPOAgent(
        model=model,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        policy_clip=config.POLICY_CLIP,
        ppo_epochs=config.PPO_EPOCHS,
        batch_size=config.BATCH_SIZE,
        vf_coef=config.VF_COEF,
        ent_coef=config.ENT_COEF,
        device=device,
    )

    # model.load_state_dict(torch.load("ppo_zzz_agent.pth"))
    # print("已加载预训练模型。")

    total_timesteps = 0
    episode_num = 0

    while total_timesteps < config.MAX_TIMESTEPS:
        state, info = env.reset()
        action_mask = info.get("action_mask", np.ones(config.N_ACTIONS, dtype=np.bool_))
        if state is None:
            print("环境重置失败，退出训练。")
            break

        done = False
        episode_reward = 0
        episode_len = 0

        # --- Episode Loop ---
        while not done:
            episode_time = time.time()
            action, log_prob, value = agent.select_action(state, action_mask)

            next_state, reward, terminated, truncated, info = env.step(action)

            debug_info = {
                "reward": reward,
                "action": action,
                "action_history": env.action_history,
                "action_mask": info.get("action_mask"),
                "ui_state": env.ui_state,
                "hp_agents": env.hp_agents,
                "hp_boss": env.hp_boss,
            }
            env.update_debug_window(debug_info)

            next_action_mask = info.get(
                "action_mask", np.ones(config.N_ACTIONS, dtype=np.bool_)
            )

            # 暂停处理
            if info.get("is_paused", False):
                env.key_manager.release_all()
                while True:
                    time.sleep(0.5)

                    # 检查环境是否已恢复
                    temp_obs = env._get_observation()
                    env._detect_all_ui_elements()
                    if temp_obs is None:  # 窗口关闭了
                        print("暂停期间窗口关闭，终止本轮。")
                        terminated = True
                        break  # 跳出暂停循环

                    env.hp_boss = env._calculate_boss_hp()
                    env.hp_agents = env._calculate_agents_hp()

                    if env.game_state() != "break":
                        print("检测到暂停已结束。")
                        recovered_state = env.recover_from_pause()  # 调用恢复函数
                        if recovered_state is None:
                            print("恢复失败，终止本轮。")
                            terminated = True
                        else:
                            state = recovered_state  # 覆盖当前状态
                            env._detect_all_ui_elements()
                            action_mask = env._get_action_mask()
                        break

                # 如果因为一些原因终止，则进入下一个 episode
                if terminated:
                    done = True
                    continue

                # 成功恢复
                continue

            done = terminated or truncated

            # 存储
            agent.store_transition(
                state, action, log_prob, reward, done, value, action_mask
            )

            state = next_state
            action_mask = next_action_mask
            episode_reward += reward
            episode_len += 1
            total_timesteps += 1
            print(
                f"E {episode_num+1:2d} | Step {total_timesteps:05d} | action {action:2d} | reward {reward:8.6f} | mask {env._get_action_mask()}"
            )

            if len(agent.memory["image_states"]) >= config.UPDATE_INTERVAL:
                print(f"达到更新步数 {config.UPDATE_INTERVAL}，开始学习...")
                from directkeys import ESC

                env.key_manager.press(ESC, 0.1)
                env.key_manager.update()
                loss_info = agent.learn()  # 获取损失

                # 记录损失到 TensorBoard
                writer.add_scalar(
                    "Loss/Policy_Loss", loss_info["policy_loss"], total_timesteps
                )
                writer.add_scalar(
                    "Loss/Value_Loss", loss_info["value_loss"], total_timesteps
                )
                writer.add_scalar(
                    "Loss/Entropy", loss_info["entropy_loss"], total_timesteps
                )

                env.key_manager.update()
                time.sleep(0.1)
                env.key_manager.press(ESC, 0.1)
                env.key_manager.update()
                time.sleep(0.2)
                env.key_manager.update()
                print("学习完成。")
                env.key_manager.release_all()

            current_time = time.time()
            if current_time - episode_time < 0.02:
                time.sleep(0.02 - (current_time - episode_time))
            episode_time = time.time()

        # --- Episode End ---
        episode_num += 1
        print(
            f"Episode: {episode_num}, Timesteps: {total_timesteps}, Length: {episode_len}, Reward: {episode_reward:.2f}"
        )
        recent_rewards.append(episode_reward)
        mean_reward_100 = np.mean(recent_rewards)

        writer.add_scalar("Performance/Episode_Reward", episode_reward, total_timesteps)
        writer.add_scalar("Performance/Episode_Length", episode_len, total_timesteps)
        writer.add_scalar(
            "Performance/Mean_Reward_100", mean_reward_100, total_timesteps
        )

        # 定期保存模型
        if episode_num % config.SAVE_INTERVAL == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(), f"checkpoints/ppo_zzz_agent_ep{episode_num}.pth"
            )
            print(f"模型已保存至 checkpoints/ppo_zzz_agent_ep{episode_num}.pth")

    env.close()
    writer.close()
    print("训练结束。")


if __name__ == "__main__":
    mp.freeze_support()
    main()

    # env = ZZZEnv()
    # test_observation_capture(env)
