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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import glob
import collections


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    agent,
    total_timesteps,
    episode_num,
    recent_rewards,
    path,
):
    print(f"[Info] 正在保存 checkpoint 至 {path} ...")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "agent_memory": agent.memory,
        "total_timesteps": total_timesteps,
        "episode_num": episode_num,
        "recent_rewards": recent_rewards,
    }
    torch.save(checkpoint, path)
    print("[Info] checkpoint 保存成功。")


def load_checkpoint(model, optimizer, scheduler, agent, path, device):
    print(f"[Info] 正在从 {path} 加载 checkpoint ...")
    if not os.path.exists(path):
        print("[Error] checkpoint 文件不存在，将从头开始训练。")
        return 0, 0, []  # 返回初始值

    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )  # weights_only=False 以不安全的方式加载（因为我还有别的东西呢）

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    agent.memory = checkpoint.get("agent_memory", agent.memory)

    total_timesteps = checkpoint["total_timesteps"]
    episode_num = checkpoint["episode_num"]
    loaded_rewards = checkpoint.get("recent_rewards", [])
    recent_rewards = collections.deque(loaded_rewards, maxlen=100)

    print(
        f"[Info] checkpoint 加载成功。已恢复至 Episode: {episode_num}, Timesteps: {total_timesteps}"
    )

    return total_timesteps, episode_num, recent_rewards


def main():
    checkpoint_dir = "checkpoints"
    log_dir_base = "log"
    run_name = f"ppo_zzz_{get_human_time()}"

    log_dir = os.path.join(log_dir_base, run_name)
    writer = SummaryWriter(log_dir)
    print(f"[Info] TensorBoard 日志将保存在: {log_dir}")  # tensorboard --logdir=log

    print("[Info] 开始PPO训练...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    env = ZZZEnv()
    env.start_debug_renderer()

    model = ActorCritic(
        input_channels=config.FRAME_STACK_SIZE,
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

    # lr 调度器
    total_updates = config.MAX_TIMESTEPS // config.UPDATE_INTERVAL
    warmup_updates = int(total_updates * 0.1)
    print(f"[Info] 总更新次数: {total_updates} | Warm-up 更新次数: {warmup_updates}")

    warmup_scheduler = LinearLR(
        agent.optimizer, start_factor=0.01, total_iters=warmup_updates
    )
    cosine_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=total_updates - warmup_updates, eta_min=1e-6
    )
    scheduler = SequentialLR(
        agent.optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_updates],
    )

    total_timesteps = 0
    episode_num = 0
    recent_rewards = collections.deque(maxlen=100)

    # 查找最新的 checkpoint 文件
    list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if list_of_files:
        latest_checkpoint_path = max(list_of_files, key=os.path.getctime)
        user_input = input(
            f"[Info] 发现最新的检查点: {latest_checkpoint_path}\n是否加载? (y/n): "
        ).lower()
        if user_input == "y":
            total_timesteps, episode_num, recent_rewards = load_checkpoint(
                model, agent.optimizer, scheduler, agent, latest_checkpoint_path, device
            )

    sSpd_sum = 0.0
    sSpd_cnt = 0

    time.sleep(2)

    while total_timesteps < config.MAX_TIMESTEPS:
        state, info = env.reset()
        action_mask = info.get("action_mask", np.ones(config.N_ACTIONS, dtype=np.bool_))
        if state is None:
            print("[Error] 环境重置失败，退出训练。")
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
                    time.sleep(0.1)

                    # 检查环境是否已恢复
                    temp_obs = env._get_observation()
                    env._detect_all_ui_elements()
                    if temp_obs is None:  # 窗口关闭了
                        print("[Warning] 暂停期间窗口关闭，终止本轮。")
                        terminated = True
                        break  # 跳出暂停循环

                    env.hp_boss = env._calculate_boss_hp()
                    env.hp_agents = env._calculate_agents_hp()

                    # 胜利也可能会进入这个状态，特判一下
                    if env.ui_state["is_victory"]:
                        terminated = env._is_terminated()
                    if terminated:
                        break

                    if env.game_state() != "break":
                        print("[Info] 检测到暂停已结束。")
                        recovered_state = env.recover_from_pause()  # 调用恢复函数
                        if recovered_state is None:
                            print("[Warning] 恢复失败，终止本轮。")
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

            is_stage_victory = info.get("is_stage_victory", False)

            # 存储
            agent.store_transition(
                state,
                action,
                log_prob,
                reward,
                done,
                value,
                action_mask,
                is_stage_victory,
            )

            # 阶段性胜利处理
            if is_stage_victory:
                env.key_manager.release_all()
                while True:
                    time.sleep(0.1)

                    # 检查环境是否已恢复
                    temp_obs = env._get_observation()
                    env._detect_all_ui_elements()
                    if temp_obs is None:  # 窗口关闭了
                        print("[Warning] 等待下阶段期间窗口关闭，终止本轮。")
                        terminated = True
                        break  # 跳出暂停循环

                    env.hp_boss = env._calculate_boss_hp()
                    env.hp_agents = env._calculate_agents_hp()

                    # 胜利也可能会进入这个状态，特判一下
                    if env.ui_state["is_victory"]:
                        terminated = env._is_terminated()
                    if terminated:
                        break

                    if env.game_state() != "break" and env.hp_boss > 0.9:
                        print("[Info] 检测到下阶段开始。")
                        recovered_state = env.recover_from_pause()  # 调用恢复函数
                        if recovered_state is None:
                            print("[Warning] 恢复失败，终止本轮。")
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

            state = next_state
            action_mask = next_action_mask
            episode_reward += reward
            episode_len += 1
            total_timesteps += 1

            sSpd = 1.0 / (time.time() - episode_time)
            sSpd_sum += sSpd
            sSpd_cnt += 1
            print(
                f"E {episode_num+1:2d} | Step {total_timesteps:05d} | reward {reward:+8.6f} | sSpd: {sSpd:4.1f}steps/s avg {sSpd_sum/sSpd_cnt:4.1f}steps/s"
            )

            if len(agent.memory["image_states"]) >= config.UPDATE_INTERVAL:
                print(f"[Info] 达到更新步数 {config.UPDATE_INTERVAL}，开始学习...")
                from directkeys import ESC

                env.key_manager.press(ESC, 0.1)
                env.key_manager.update()
                loss_info = agent.learn()  # 获取损失

                scheduler.step()

                # 记录损失到 TensorBoard
                writer.add_scalar(
                    "Info/Learning_Rate", scheduler.get_last_lr()[0], total_timesteps
                )
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
                env.key_manager.press(ESC, 0.02)
                env.key_manager.update()
                time.sleep(0.1)
                print("[Info] 学习完成。")
                env.key_manager.update()
                env.key_manager.release_all()
                while True:
                    # 检查环境是否已恢复
                    temp_obs = env._get_observation()
                    env._detect_all_ui_elements()
                    if temp_obs is None:  # 窗口关闭了
                        print("[Warning] 暂停期间窗口关闭，终止本轮。")
                        terminated = True
                        break  # 跳出暂停循环

                    env.hp_boss = env._calculate_boss_hp()
                    env.hp_agents = env._calculate_agents_hp()

                    if env.game_state() != "break":
                        print("[Info] 检测到暂停已结束。")
                        recovered_state = env.recover_from_pause()  # 调用恢复函数
                        if recovered_state is None:
                            print("[Warning] 恢复失败，终止本轮。")
                            terminated = True
                        else:
                            state = recovered_state  # 覆盖当前状态
                            env._detect_all_ui_elements()
                            action_mask = env._get_action_mask()
                        break

                    if env._no_in_game():
                        print("[Info] 检测到暂停已结束，且战斗结束。")
                        terminated = True
                        break

                    time.sleep(0.1)

        time.sleep(4)  # 等结算

        # --- Episode End ---
        episode_num += 1
        print(
            f"Episode: {episode_num}, Timesteps: {total_timesteps}, Length: {episode_len}, Reward: {episode_reward:.2f}"
        )
        recent_rewards.append(episode_reward)
        mean_reward_100 = np.mean(np.array(recent_rewards, dtype=np.float32))

        writer.add_scalar("Performance/Episode_Reward", episode_reward, total_timesteps)
        writer.add_scalar("Performance/Episode_Length", episode_len, total_timesteps)
        writer.add_scalar(
            "Performance/Mean_Reward_100", mean_reward_100, total_timesteps
        )

        # 定期保存模型
        if episode_num % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_ep{episode_num}.pth"
            )
            save_checkpoint(
                model,
                agent.optimizer,
                scheduler,
                agent,
                total_timesteps,
                episode_num,
                recent_rewards,
                checkpoint_path,
            )

    env.close()
    writer.close()
    print("[Info] 训练结束。")


if __name__ == "__main__":
    mp.freeze_support()
    main()

    # env = ZZZEnv()
    # test_observation_capture(env)
