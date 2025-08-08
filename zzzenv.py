# zzzenv.py

import config
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import win32gui
from directkeys import KeyManager, move_to_and_click_relative
from myutils import get_dpi_scale_factor, select_target_window, put_text_chinese
import time
import random
from scipy import stats
import os
from yolo_detection import detect_batch
import collections
import multiprocessing as mp
import time

# from debug_renderer import renderer_process
from debug_renderer_thread import DebugRendererThread

eps = 1e-6


class RewardNormalizer:
    """
    使用 Welford's online algorithm 动态地将奖励归一化到零均值和单位标准差。
    我们只使用标准差来缩放奖励，而不减去均值，以保留奖励的原始正负信号。
    基于 OpenAI Baselines 和 Stable-Baselines3 中的标准做法
    """

    def __init__(self, num_envs=1, clip_reward=10.0, gamma=0.99, epsilon=1e-8):
        self.mean = np.zeros(num_envs)
        self.var = np.ones(num_envs)
        self.count = epsilon

        self.return_rms = RunningMeanStd(shape=())
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

    def __call__(self, reward):
        """
        根据运行时的标准差缩放奖励。
        """
        reward = np.clip(reward, -self.clip_reward, self.clip_reward)
        self.return_rms.update(np.array([reward]))
        scaled_reward = reward / (np.sqrt(self.return_rms.var) + self.epsilon)
        return scaled_reward


class RunningMeanStd:
    """Welford's online algorithm"""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ZZZEnv(gym.Env):
    def __init__(self):
        super(ZZZEnv, self).__init__()

        # 定义动作空间 action space
        """
        0   不动
        1   A     O
        2   闪避    L Shift
        3   E       E
        4   Q       Q
        5   切人上  C
        6   切人下  Space 

        7   连携上  T
        8   连携下  Y
        9   连携取消 I
        10  W       W
        11  A       A
        12  S       S
        13  D       D

        14 手动锁定 U
        """
        self.action_space = spaces.Discrete(config.N_ACTIONS)

        # 定义状态空间 obs space
        # 画面要缩小，并不需要很清楚就足以提供需要的信息了
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.FRAME_STACK_SIZE),
            dtype=np.uint8,
        )

        self.action_space = spaces.Discrete(config.N_ACTIONS)

        self.hwnd = select_target_window()
        if not self.hwnd:
            return
        print(
            f"已选择窗口句柄: {self.hwnd}, Title: '{win32gui.GetWindowText(self.hwnd)}'"
        )

        self.sct = mss.mss()  # 截图要用的

        self.key_manager = KeyManager()  # 创建非阻塞键盘管理器
        self.img = None  # 实时截的图

        self.frame_stack = collections.deque(maxlen=config.FRAME_STACK_SIZE)
        # maxlen 指定后，满了自动会 pop

        self.hp_agents = [0.0, 0.0, 0.0]  # 血条
        self.hp_boss = 0.0
        self.last_hp_agents = [0.0, 0.0, 0.0]
        self.last_hp_boss = 0.0

        self.consecutive_action_count = 0  # 连续相同的动作次数
        self.last_action = -1  # 初始化为 -1

        # 动作历史队列，长度为N
        self.action_history = collections.deque(maxlen=config.ACTION_HISTORY_LEN)
        for _ in range(config.ACTION_HISTORY_LEN):
            self.action_history.append(-1)

        # self.ui_pipe = None
        # self.ui_process = None
        self.debug_thread = None

        self.ui_state = {}

        self.killed_confirm_frames = 0
        self.victory_confirm_frames = 0
        self.defeat_confirm_frames = 0
        self.final_reward_given = False

        # CD 状态计时器 0=可用, 1=小cd, 2=大cd
        self.dodge_state = 0
        self.dodge_timer = 0.0

        self.switch_state = 0
        self.switch_timer = 0.0

        self.q_active = False
        self.q_timer = 0.0

        self.reward_normalizer = RewardNormalizer(gamma=config.GAMMA)

        self.last_special_event = "none"

    def _get_stacked_frames(self):
        """将帧堆叠从队列变成 numpy 数组"""
        # frames_array = np.stack(self.frame_stack, axis=0)
        # 将 K 和 C 两个维度合并，从 [K, H, W, 3] -> [H, W, K*3]
        # PyTorch 的 CNN (channels_first) 需要 [C*K, H, W]
        # TensorFlow 的 CNN (channels_last) 需要 [H, W, C*K]
        # 我们先适配 PyTorch 的习惯，在 agent 里再 permute
        # [K, H, W, 3] -> [K, 3, H, W]
        # frames_transposed = frames_array.transpose(0, 3, 1, 2)
        # [K, 3, H, W] -> [K*3, H, W]
        # stacked_frames = frames_transposed.reshape(
        # -1, config.IMG_HEIGHT, config.IMG_WIDTH
        # )
        # return stacked_frames

        return np.stack(self.frame_stack, axis=0)

    def _get_current_state(self):
        """将堆叠后的图像和动作历史打包成一个字典作为状态"""
        return {
            "image": self._get_stacked_frames(),
            "action_history": np.array(self.action_history, dtype=np.int32),
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        print("开始新一轮战斗...")

        # 进入关卡
        self.restart_level()
        # self.into_level()
        print("已进入关卡，等待战斗开始...")

        # 等一下 boss 启动
        time.sleep(config.BATTLE_START_WAIT_TIME)

        # 初始化血量状态
        print("正在初始化状态...")
        current_obs = self._get_observation()
        self._detect_all_ui_elements()

        if current_obs is None:
            print("[Error] 无法在 reset 时获取画面，请检查游戏窗口。")
            current_obs = np.zeros(
                self.observation_space.shape, dtype=np.uint8
            )  # 用全黑填充

        # 用第一帧填满
        self.frame_stack.clear()
        for _ in range(config.FRAME_STACK_SIZE):
            self.frame_stack.append(current_obs)

        self.hp_boss = self._calculate_boss_hp()
        self.hp_agents = self._calculate_agents_hp()
        self.last_hp_boss = self.hp_boss
        self.last_hp_agents = np.copy(self.hp_agents)  # 使用 np.copy 来深拷贝

        # self.last_action = -1 # 连续动作惩罚
        # self.consecutive_action_count = 0

        self.killed_confirm_frames = 0
        self.defeat_confirm_frames = 0
        self.victory_confirm_frames = 0
        self.final_reward_given = False

        self.dodge_state = 0
        self.dodge_timer = 0.0

        self.switch_state = 0
        self.switch_timer = 0.0

        self.q_active = False
        self.q_timer = 0.0

        print(
            f"初始化完成。Boss HP: {self.hp_boss:.2f}, Agent HPs: {[f'{h:.2f}' for h in self.hp_agents]}"
        )

        for _ in range(config.ACTION_HISTORY_LEN):
            self.action_history.append(-1)

        info = {"action_mask": self._get_action_mask()}
        return self._get_current_state(), info

    def update_cd_timer(self, action):
        if action == 2:  # 按下闪避
            if self.dodge_state == 0:
                self.dodge_state = 1
                self.dodge_timer = time.time()
            elif self.dodge_state == 1:
                if 0.35 <= (time.time() - self.dodge_timer) <= 1.0:
                    self.dodge_state = 2
                    self.dodge_timer = time.time()
        elif action == 5 or action == 6:
            if self.switch_state == 0:
                self.switch_state = 1
                self.switch_timer = time.time()
            elif self.switch_state == 1:
                if (time.time() - self.switch_timer) <= 0.65:
                    self.switch_state = 2
                    self.switch_timer = time.time()

    def step(self, action):
        """执行动作"""
        self.key_manager.update()
        self.key_manager.do_action(action, 0.001)
        time.sleep(0.002)
        self.key_manager.update()

        self.action_history.append(action)
        self.update_cd_timer(action)

        new_obs = self._get_observation()
        if new_obs is None:
            # 同样用最后一帧填充
            new_obs = self.frame_stack[-1]
        self.frame_stack.append(new_obs)

        # 获取新状态
        # time.sleep(config.ACTION_DELAY)
        next_state = self._get_current_state()
        self._detect_all_ui_elements()

        self.last_hp_boss = self.hp_boss
        self.last_hp_agents = np.copy(self.hp_agents)
        self.hp_boss = self._calculate_boss_hp()
        self.hp_agents = self._calculate_agents_hp()

        # 检查是否处于非正常状态（血条不对劲了）
        if self.game_state() == "break":
            print("检测到暂停状态 (is_break)，将暂停训练...")
            # 返回当前观察值，奖励为 0 ，done 为 False，***在 info 中返回一个暂停信号***
            info = {"is_paused": True}
            return next_state, 0.0, False, False, info

        # 计算奖励
        raw_reward = self._calculate_reward()

        if action != 0:
            raw_reward -= config.ACTION_COST  # 做非空动作有惩罚

        """
        repetition_penalty = 0
        if action == self.last_action:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1
            self.last_action = action

        
        # 如果连续次数超过了一个阈值，才开始施加连续用同一个动作的惩罚
        # upd250715 感觉现在这个没啥用，先注释了
        if self.consecutive_action_count > config.REPETITION_THRESHOLD:
            penalty_factor = (
                self.consecutive_action_count - config.REPETITION_THRESHOLD
            ) ** 2
            repetition_penalty = config.REPETITION_PENALTY_SCALE * penalty_factor

        if action:
            reward -= repetition_penalty
        """

        # reward = self.reward_normalizer(raw_reward)
        # byd 先别急，用了直接训爆了

        scaled_reward = raw_reward * 0.1  # 先用老手艺
        clipped_reward = np.clip(scaled_reward, -1.0, 1.0)

        reward = clipped_reward

        # 判断是否结束
        terminate_state = self._is_terminated()
        terminated = terminate_state == 1
        is_stage_victory = terminate_state == 2  # 阶段性胜利
        truncated = False

        if terminate_state == 2:
            print("检测到阶段胜利，将暂停训练...")
            reward += config.VICTORY_REWARD
            self.key_manager.release_all()

            wait_start_time = time.time()
            while time.time() - wait_start_time < 30:  # 最多等待 30s
                time.sleep(0.5)

                temp_obs = self._get_observation()
                if temp_obs is not None:
                    self.frame_stack.append(temp_obs)  # 保持帧栈更新
                self._detect_all_ui_elements()
                self.hp_boss = self._calculate_boss_hp()

                if self._is_terminated() == 1:
                    terminated = True
                    break

                if self.game_state() != "break" and self.hp_boss > 0.9:
                    print("[Info] 检测到下阶段开始，恢复运行。")
                    # self.recover_from_pause()
                    # 只需更新血量和UI状态，帧栈已经是连续的了，不需要 recover_from_pause
                    self.last_hp_boss = self.hp_boss
                    self.last_hp_agents = np.copy(self._calculate_agents_hp())
                    break

            # 超时退出
            if not terminated and not (
                self.game_state() != "break" and self.hp_boss > 0.9
            ):
                print("[Warning] 等待Boss下一阶段超时，终止episode。")
                terminated = True

        if terminated and not self.final_reward_given:
            # 胜利/败北 reward
            if self.ui_state["is_victory"]:
                reward += config.VICTORY_REWARD
            elif self._is_defeat():
                # print("DEFEAT!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                reward += config.DEFEAT_PENALTY
            self.final_reward_given = True

        info = {
            "is_paused": False,
            "consecutive_actions": self.consecutive_action_count,
            "current_action": action,
            "action_mask": self._get_action_mask(),
            "is_stage_victory_for_gae": is_stage_victory,
        }
        # TODO:
        # truncated 是时间到了而不是结束 terminated 了，可能在有利局面下结束
        # 所以如果 truncated 那还是要计算贡献的，只不过暂时不会遇到这种情况
        return next_state, reward, terminated, truncated, info

    def recover_from_pause(self):
        """
        从暂停状态恢复
        """
        print("[Info] 从暂停中恢复，正在重新校准状态...")
        time.sleep(0.1)  # 等待画面稳定

        obs = self._get_observation()
        if obs is None:
            print("[Error] 恢复失败，无法获取画面。")
            return None

        if len(self.frame_stack) > 0:
            self.frame_stack.pop()
            self.frame_stack.append(obs)
        # self.frame_stack.clear()
        # for _ in range(config.FRAME_STACK_SIZE):
        # self.frame_stack.append(obs)

        current_state = self._get_current_state()
        if current_state is None:
            print("[Error] 恢复失败，无法获取画面。")
            return None

        self.hp_boss = self._calculate_boss_hp()
        self.hp_agents = self._calculate_agents_hp()

        self.last_hp_boss = self.hp_boss
        self.last_hp_agents = np.copy(self.hp_agents)

        print(
            f"[Info] 状态已校准。Boss HP: {self.hp_boss:.2f}, Agent HPs: {[f'{h:.2f}' for h in self.hp_agents]}"
        )

        return current_state

    def _get_observation(self):
        if not self.hwnd or not self.sct:
            return None

        try:
            if not win32gui.IsWindow(self.hwnd):
                print("[Error] 目标窗口已关闭。")
                self.hwnd = None
                return None

            left_client, top_client, right_client, bottom_client = (
                win32gui.GetClientRect(self.hwnd)
            )
            client_width = right_client - left_client
            client_height = bottom_client - top_client

            screen_x_logical, screen_y_logical = win32gui.ClientToScreen(
                self.hwnd, (left_client, top_client)
            )
            monitor = {
                "top": screen_y_logical,
                "left": screen_x_logical,
                "width": client_width,
                "height": client_height,
            }

            if monitor["width"] <= 0 or monitor["height"] <= 0:
                return None

            self.img = np.array(self.sct.grab(monitor))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2BGR)

            # 压缩图片
            img_resized = cv2.resize(self.img, (config.IMG_WIDTH, config.IMG_HEIGHT))
            # 转成灰度图
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            return img_gray
            # return img_resized
        except Exception as e:
            print(f"捕捉画面时发生错误: {e}")
            return None

    def flush(self):
        _ = self._get_observation()
        self.last_hp_boss = self.hp_boss
        self.last_hp_agents = np.copy(self.hp_agents)
        self.hp_boss = self._calculate_boss_hp()
        self.hp_agents = self._calculate_agents_hp()
        self._detect_all_ui_elements()

    def _calculate_agents_hp(self):
        if self.img is None or self.img.size == 0:
            return [0, 0, 0]

        hp_values = []
        for i, (x, y, w, h) in enumerate(config.HP_COORD):
            roi = self.img[y : y + h, x : x + w]

            if roi.size == 0:
                continue

            green_channel = roi[:, :, 1]
            mask = green_channel > 250
            # img 是 BGR!!! zzz 虽然渐变血条但是 G 都是 255

            row_sums = np.sum(mask, axis=1)

            if row_sums.size > 0:
                result_hp = (
                    np.median(row_sums) + np.max(row_sums) + np.min(row_sums) / 3
                )
                # [细节控] byd 绝区零血条后面有动态效果，这里 xjb 降低扰动，剩下的只能说长期来看无所谓
            else:
                result_hp = 0.0

            result_hp /= config.HP_LENGTH[i]
            if result_hp > 1.0:
                result_hp = 1.0
            if result_hp < 0.0:
                result_hp = 0.0
            hp_values.append(result_hp)

        return hp_values

    def _calculate_boss_hp(self):
        if self.img is None or self.img.size == 0:
            return 0

        x, y, w, h = config.BOSS_HP_COORD
        roi = self.img[y : y + h, x : x + w]

        if roi.size == 0:
            return 0

        if roi[5][5][1] < 10:  # not a hp sticker
            return 0.0

        green_channel = roi[:, :, 1]
        mask = green_channel > 250

        row_sums = np.sum(mask, axis=1)

        if row_sums.size == 0:
            return 0
        result_hp = np.median(row_sums) + np.max(row_sums) + np.min(row_sums) / 3

        result_hp /= config.BOSS_HP_LENGTH

        if result_hp > 1.0:
            result_hp = 1.0
        if result_hp < 0.0:
            result_hp = 0.0
        return result_hp

    def _detect_all_ui_elements(self):
        """
        更新 ui_state 的过程（没有返回值！！！）
        """
        if self.img is None or self.img.size == 0:
            self.ui_state = {}
            return

        rois = {
            "q": self.img[
                config.Q_COORD[1] : config.Q_COORD[1] + config.Q_COORD[3],
                config.Q_COORD[0] : config.Q_COORD[0] + config.Q_COORD[2],
            ],
            "chain_attack": self.img[
                config.CHAINATK_COORD[1] : config.CHAINATK_COORD[1]
                + config.CHAINATK_COORD[3],
                config.CHAINATK_COORD[0] : config.CHAINATK_COORD[0]
                + config.CHAINATK_COORD[2],
            ],
            "back_pause": self.img[
                config.BACK_PAUSE_COORD[1] : config.BACK_PAUSE_COORD[1]
                + config.BACK_PAUSE_COORD[3],
                config.BACK_PAUSE_COORD[0] : config.BACK_PAUSE_COORD[0]
                + config.BACK_PAUSE_COORD[2],
            ],
            "investigation": self.img[
                config.INVESTIGATION_COMPLETE_COORD[
                    1
                ] : config.INVESTIGATION_COMPLETE_COORD[1]
                + config.INVESTIGATION_COMPLETE_COORD[3],
                config.INVESTIGATION_COMPLETE_COORD[
                    0
                ] : config.INVESTIGATION_COMPLETE_COORD[0]
                + config.INVESTIGATION_COMPLETE_COORD[2],
            ],
            "decibel": self.img[
                config.DECIBEL_COORD[1] : config.DECIBEL_COORD[1]
                + config.DECIBEL_COORD[3],
                config.DECIBEL_COORD[0] : config.DECIBEL_COORD[0]
                + config.DECIBEL_COORD[2],
            ],
        }

        roi_names_ordered = [name for name, roi in rois.items() if roi.size > 0]
        roi_images_ordered = [roi for roi in rois.values() if roi.size > 0]

        if not roi_names_ordered:
            self.ui_state = {}
            return

        detection_results = detect_batch(roi_images_ordered)

        detection_final_results = {}

        for i, name in enumerate(roi_names_ordered):
            class_name, score = detection_results[i]
            detection_final_results[name] = class_name

        self.ui_state["is_esc"] = detection_final_results["back_pause"] == "back"
        self.ui_state["is_controllable"] = (
            detection_final_results["back_pause"] == "pause"
        )
        self.ui_state["is_chain_attack"] = (
            detection_final_results["chain_attack"] == "chainatk"
        )

        self.ui_state["is_qable"] = detection_final_results["q"] == "qable"
        self.ui_state["is_victory"] = (not self.ui_state["is_controllable"]) and (
            detection_final_results["investigation"] == "investigation_complete"
        )

        result_decibel = detection_final_results["decibel"]
        self.ui_state["decibel"] = "none"
        if len(result_decibel) > 8:
            if result_decibel[:8] == "decibel_":
                self.ui_state["decibel"] = result_decibel[8:]

    def _no_in_game(self):
        """从 PPO learning 中恢复特判，在开终结技后胜利途中 ESC 不了"""
        return (not self.ui_state["is_esc"]) and (not self.ui_state["is_controllable"])

    def _is_terminated(self):
        is_killed_condition_met = self._is_killed()
        is_victory_condition_met = self.ui_state["is_victory"]
        is_defeat_condition_met = self._is_defeat()

        if (self.killed_confirm_frames == 0 and self.defeat_confirm_frames == 0) and (
            is_killed_condition_met
            and is_defeat_condition_met
            and not is_victory_condition_met
        ):
            return 0

        if is_victory_condition_met:
            self.victory_confirm_frames += 1
            self.killed_confirm_frames = 0
            self.defeat_confirm_frames = 0
        elif is_killed_condition_met:
            self.killed_confirm_frames += 1
            self.defeat_confirm_frames = 0
            self.victory_confirm_frames = 0
        elif is_defeat_condition_met:
            self.defeat_confirm_frames += 1
            self.killed_confirm_frames = 0
            self.victory_confirm_frames = 0
        else:
            self.killed_confirm_frames = 0
            self.defeat_confirm_frames = 0
            self.victory_confirm_frames = 0

        if self.killed_confirm_frames > 0:
            print(f"阶段胜利计数器 {self.killed_confirm_frames}")

        if self.defeat_confirm_frames > 0:
            print(f"失败计数器 {self.defeat_confirm_frames}")

        if self.victory_confirm_frames > 0:
            print(f"胜利计数器 {self.victory_confirm_frames}")

        if self.victory_confirm_frames >= config.TERMINATED_COUNT:
            print(f"胜利")
            return 1
        if self.killed_confirm_frames >= config.TERMINATED_COUNT:
            print(f"阶段性胜利")
            return 2
        if self.defeat_confirm_frames >= config.TERMINATED_COUNT:
            print(f"失败")
            return 1
        return 0

    def _is_env_safe(self):  # learn 需要一个安全的环境
        return self.ui_state.get("is_controllable", False) and self.hp_boss > 0.1

    def _is_killed(self):
        no_hp_values = np.min(self.hp_agents) > eps and self.hp_boss < eps
        hp_top_row = self.img[
            config.BOSS_HP_TOP_COORD[1],
            config.BOSS_HP_TOP_COORD[0] : config.BOSS_HP_TOP_COORD[0]
            + config.BOSS_HP_TOP_COORD[2],
            :,
        ]
        max_channel_values = np.max(hp_top_row, axis=1)
        has_hp_status = np.all(max_channel_values < 10)
        return no_hp_values and (not has_hp_status)

    def _is_defeat(self):
        return np.min(self.hp_agents) < eps and self.hp_boss > eps

    def _is_firstline_black(
        self,
    ):  # 从ESC恢复/从running到ESC 的时候要特判一下，有一个黑条渐变动画
        first_row = self.img[0, :20, :]
        max_channel_values = np.max(first_row, axis=1)
        return np.all(max_channel_values < 2)

    def game_state(self):
        # if self.defeat_confirm_frames > 0 or self.victory_confirm_frames > 0:  # 判定中
        # killed 不参与是因为可能还有丝血
        # return "running"
        if self.ui_state.get("is_controllable", False):
            if self._is_firstline_black():
                return "break"  # 防止 hp 0 0 0 0 导致 reward 爆了
            return "running"
        if self.ui_state.get("is_esc", False):
            return "break"
        if self.ui_state.get("is_chain_attack", False):
            print("进入连携技")
            self.key_manager.update()
            self.key_manager.do_action(7, 0.05)
            self.key_manager.update()
            time.sleep(0.1)
            self.key_manager.update()
            self.key_manager.release_all()
            print("连携技完成")
            return "break"
        print("开大了")
        self.q_active = True
        self.q_timer = time.time()
        return "break"

    def _get_action_mask(self):
        """
        表示哪些动作是可用的，one-hot 编码，1可用，0不可用
        """
        mask = np.ones(config.N_ACTIONS, dtype=np.bool_)

        current_time = time.time()

        # 开大后停 4os 让他知道是开 q 才有 reward
        if self.q_active:
            if current_time - self.q_timer > 1.0:
                self.q_active = False
            else:
                mask[:] = False
                mask[0] = True
                return mask

        is_dodge_available = False
        if self.dodge_state == 0:
            is_dodge_available = True
        elif self.dodge_state == 1:
            elapsed = current_time - self.dodge_timer
            if elapsed > 1.0:
                self.dodge_state = 0
                is_dodge_available = True
            elif 0.35 <= elapsed <= 1.0:
                is_dodge_available = True
        elif self.dodge_state == 2:
            elapsed = current_time - self.dodge_timer
            if elapsed > 0.8:
                self.dodge_state = 0
                is_dodge_available = True
        mask[2] = is_dodge_available

        is_switch_available = False
        if self.switch_state == 0:
            is_switch_available = True
        elif self.switch_state == 1:
            elapsed = current_time - self.switch_timer
            if elapsed > 0.65:
                self.switch_state = 0
                is_switch_available = True
            else:
                is_switch_available = True
        elif self.switch_state == 2:
            elapsed = current_time - self.switch_timer
            if elapsed > 0.65:
                self.switch_state = 0
                is_switch_available = True
        mask[5] = is_switch_available
        mask[6] = is_switch_available

        mask[0] = True
        mask[1] = True
        mask[3] = True

        mask[4] = self.ui_state.get("is_qable", False)

        return mask

    def _calculate_reward(self):
        boss_hp_diff = self.hp_boss - self.last_hp_boss
        agents_hp_diff = np.sum(self.hp_agents) - np.sum(self.last_hp_agents)

        def scale_change(change, exponent=3):  # 降低扰动带来的影响，小的小，大的大
            if change == 0:
                return 0
            return np.sign(change) * (abs(change) ** exponent)

        boss_hp_diff *= 50
        agents_hp_diff *= 20

        boss_damage_reward = (
            -scale_change(boss_hp_diff, exponent=1.5) * config.BOSS_DMG_REWARD_SCALE
        )
        agent_damage_penalty = (
            scale_change(agents_hp_diff, exponent=2.0) * config.AGENT_DMG_PENALTY_SCALE
        )

        reward = boss_damage_reward + agent_damage_penalty

        # 时间惩罚
        reward -= config.TIME_PENALTY

        current_special_event = self.ui_state["decibel"]
        special_reward = 0
        if (
            current_special_event != "none"
            and current_special_event != self.last_special_event
        ):
            if not current_special_event == "maximum":
                special_reward += config.DECIBEL_REWARD
            if current_special_event == "perfect_dodge":
                special_reward = config.PERFECT_DODGE_REWARD
            elif current_special_event == "assist_attack":
                special_reward = config.ASSIST_ATTACK_REWARD
            # ... 其他事件
            print(
                f"[Info/Reward Shaping] 触发新事件: {current_special_event}，奖励: {special_reward}"
            )

        self.last_special_event = current_special_event
        reward += special_reward

        return reward

    # def start_debug_renderer(self):
    #     """启动独立的 ui 渲染进程。"""
    #     if self.ui_process and self.ui_process.is_alive():
    #         print("调试窗口已在运行。")
    #         return

    #     parent_conn, child_conn = mp.Pipe()  # 双向管道
    #     self.ui_pipe = parent_conn

    #     # 启动进程
    #     self.ui_process = mp.Process(target=renderer_process, args=(child_conn,))
    #     self.ui_process.start()
    #     print("调试窗口进程已启动。")

    # def update_debug_window(self, info_dict):
    #     """通过管道向 ui 进程发送新数据。"""
    #     if self.ui_pipe and not self.ui_pipe.closed:
    #         try:
    #             self.ui_pipe.send(info_dict)
    #         except (BrokenPipeError, EOFError):
    #             print("无法发送数据到调试窗口，管道已关闭。")
    #             self.ui_pipe = None

    def start_debug_renderer(self):
        if self.debug_thread and self.debug_thread.is_alive():
            return
        self.debug_thread = DebugRendererThread()
        self.debug_thread.start()
        print("调试线程已启动。")

    def update_debug_window(self, info_dict):
        if self.debug_thread:
            self.debug_thread.update_data(info_dict)

    def close(self):
        # if self.ui_pipe:
        #     print("正在关闭调试窗口...")
        #     try:
        #         self.ui_pipe.send("TERMINATE")
        #         self.ui_pipe.close()
        #     except (BrokenPipeError, EOFError):
        #         pass  # 如果管道已经坏了，就不用管了

        # if self.ui_process:
        #     self.ui_process.join(timeout=2)  # 等待进程结束
        #     if self.ui_process.is_alive():
        #         self.ui_process.terminate()  # 如果 2s 内还没结束就强制终止

        if self.debug_thread:
            self.debug_thread.stop()
            self.debug_thread.join(timeout=1)  # 等待线程结束

        # cv2.destroyAllWindows() # renderer_process 里头有了
        self.key_manager.release_all()

    def into_level(self):
        print("开始进入关卡")
        start_time = time.time()
        idx = 0
        while time.time() - start_time < 6.5:
            from directkeys import _1, _2, _3, F, U

            if idx == 0 and time.time() - start_time > 0.5:
                self.key_manager.do_action(11, 0.5)
                idx += 1
            if idx == 1 and time.time() - start_time > 1.0:
                self.key_manager.do_action(10, 0.75)
                idx += 1
            if idx == 2 and time.time() - start_time > 2.0:
                self.key_manager.press(F, 0.1)
                idx += 1
            if idx == 3 and time.time() - start_time > 3.0:
                self.key_manager.press(_2, 0.1)
                idx += 1
            if idx == 4 and time.time() - start_time > 3.5:
                self.key_manager.do_action(10, 3.5)
                idx += 1
            if idx == 5 and time.time() - start_time > 6:
                self.key_manager.press(U, 0.1)
                idx += 1

            self.key_manager.update()

        self.key_manager.release_all()

        print("进入关卡完成")

    def restart_level(self):
        print("开始手动重置关卡")
        start_time = time.time()
        idx = 0
        while True:
            from directkeys import _1, _2, _3, F, U, ESC

            if idx == 0 and time.time() - start_time > 0.1:
                self.key_manager.press(ESC, 0.5)
                idx += 1
            if idx == 1 and time.time() - start_time > 1.0:
                dx = int(config.MOUSE_ESC_TO_RESTART_WIDTH)
                dy = int(config.MOUSE_ESC_TO_RESTART_HEIGHT)
                move_to_and_click_relative(dx, dy)
                idx += 1

            if idx == 2 and time.time() - start_time > 2.0:
                dx = int(config.MOUSE_RESTART_TO_CONFIRM_WIDTH)
                dy = int(config.MOUSE_RESTART_TO_CONFIRM_HEIGHT)
                move_to_and_click_relative(dx, dy)
                idx += 1

            if idx == 3 and time.time() - start_time > 5.0:
                _ = self._get_observation()
                # print(self._calculate_agents_hp())
                if self._calculate_agents_hp()[0] > 0.01:
                    time.sleep(1)
                    break
                time.sleep(0.1)

            self.key_manager.update()

        self.key_manager.release_all()
        self.into_level()

        print("重置关卡完成")


# 测试用的，不用管
def test_observation_capture(env: ZZZEnv):
    """
    接收一个环境实例，持续调用其 _get_observation 方法，并实时显示结果。
    """
    if not env.hwnd:
        print("环境初始化失败，无法开始测试。")
        return

    print("\n--- 开始实时捕获画面 ---")
    print(f"将会显示一个名为 'Live Observation' 的窗口。")
    print("按 'q' 键或关闭该窗口来退出测试。")

    time.sleep(3)

    print("wakeup!")

    test_time = time.time()

    env.restart_level()
    # env.into_level()

    while True:
        observation = env._get_observation()

        if observation is None:
            print("无法获取画面，可能窗口已关闭。正在退出...")
            break

        env.hp_agents = env._calculate_agents_hp()
        env.hp_boss = env._calculate_boss_hp()

        cv2.imshow("Live Observation", observation)

        if 0 and time.time() - test_time > 0.2:
            env.key_manager.do_action(random.randint(0, config.N_ACTIONS - 1))
            test_time = time.time()

        env.key_manager.update()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    env.close()
    print("测试结束。")
