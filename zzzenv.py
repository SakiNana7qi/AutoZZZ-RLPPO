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
from myutils import get_dpi_scale_factor, select_target_window
import time
import random
from scipy import stats
import os
from yolo_detection import detect_image
import collections

eps = 1e-6


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
            shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
            dtype=np.uint8,
        )

        self.hwnd = select_target_window()
        if not self.hwnd:
            return
        print(
            f"已选择窗口句柄: {self.hwnd}, Title: '{win32gui.GetWindowText(self.hwnd)}'"
        )

        self.sct = mss.mss()  # 截图要用的

        self.key_manager = KeyManager()  # 创建非阻塞键盘管理器
        self.img = None  # 实时截的图

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

    def _get_current_state(self):
        """将图像和动作历史打包成一个字典作为状态"""
        return {
            "image": self._get_observation(),
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
        if current_obs is None:
            print("[Error] 无法在 reset 时获取画面，请检查游戏窗口。")
            # 返回全黑
            return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

        self.hp_boss = self._calculate_boss_hp()
        self.hp_agents = self._calculate_agents_hp()
        self.last_hp_boss = self.hp_boss
        self.last_hp_agents = np.copy(self.hp_agents)  # 使用 np.copy 来深拷贝

        self.last_action = -1
        self.consecutive_action_count = 0

        print(
            f"初始化完成。Boss HP: {self.hp_boss:.2f}, Agent HPs: {[f'{h:.2f}' for h in self.hp_agents]}"
        )

        for _ in range(config.ACTION_HISTORY_LEN):
            self.action_history.append(-1)

        info = {"action_mask": self._get_action_mask()}
        return self._get_current_state(), info

    def step(self, action):
        """执行动作"""
        self.key_manager.update()
        self.key_manager.do_action(action, 0.001)
        time.sleep(0.002)
        self.key_manager.update()

        self.action_history.append(action)

        # 获取新状态
        # time.sleep(config.ACTION_DELAY)
        next_state = self._get_current_state()

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
        reward = self._calculate_reward()

        if action != 0:
            reward -= config.ACTION_COST  # 做非空动作有惩罚

        repetition_penalty = 0
        if action == self.last_action:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1
            self.last_action = action

        """
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

        # 判断是否结束
        terminated = self._is_terminated()
        truncated = False

        info = {
            "is_paused": False,
            "repetition_penalty": repetition_penalty,
            "consecutive_actions": self.consecutive_action_count,
            "current_action": action,
            "action_mask": self._get_action_mask(),
        }
        # TODO:
        # truncated 是时间到了而不是结束 terminated 了，可能在有利局面下结束
        # 所以如果 truncated 那还是要计算贡献的，只不过暂时不会遇到这种情况
        return next_state, reward, terminated, truncated, info

    def recover_from_pause(self):
        """
        从暂停状态恢复
        """
        print("从暂停中恢复，正在重新校准状态...")
        time.sleep(0.5)  # 等待画面稳定

        current_state = self._get_current_state()
        if current_state is None:
            print("恢复失败，无法获取画面。")
            return None

        self.hp_boss = self._calculate_boss_hp()
        self.hp_agents = self._calculate_agents_hp()

        self.last_hp_boss = self.hp_boss
        self.last_hp_agents = np.copy(self.hp_agents)

        print(
            f"状态已校准。Boss HP: {self.hp_boss:.2f}, Agent HPs: {[f'{h:.2f}' for h in self.hp_agents]}"
        )

        return current_state

    def _get_observation(self):
        if not self.hwnd or not self.sct:
            return None

        try:
            if not win32gui.IsWindow(self.hwnd):
                print("目标窗口已关闭。")
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
            return img_resized
        except Exception as e:
            print(f"捕捉画面时发生错误: {e}")
            return None

    def flush(self):
        _ = self._get_observation()
        self.last_hp_boss = self.hp_boss
        self.last_hp_agents = np.copy(self.hp_agents)
        self.hp_boss = self._calculate_boss_hp()
        self.hp_agents = self._calculate_agents_hp()

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

    def _is_esc(self):
        if self.img is None or self.img.size == 0:
            return False
        x, y, w, h = config.ESC_COORD
        roi = self.img[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        name, score = detect_image(roi)
        return name == "settings"

    def _is_terminated(self):
        if self._is_victory() or self._is_defeat():  # 先只关心一阶段
            # if self._is_controllable():
            #     if self._is_victory():
            #         return True
            #     return False
            return True
        return False

    def _is_controllable(self):  # 有闹钟说明在正常战斗界面：
        if self.img is None or self.img.size == 0:
            return False
        x, y, w, h = config.CLOCK_COORD
        roi = self.img[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        name, score = detect_image(roi)
        # print(f"controllable {name} {score:.2f}")
        return name == "clock"

    def _is_chain_attack(self):
        """是否在连携技"""
        if self.img is None or self.img.size == 0:
            return False
        x, y, w, h = config.CHAINATK_COORD
        roi = self.img[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        name, score = detect_image(roi)
        # print(f"chainatk {name} {score:.2f}")
        return name == "chainatk"

    def _is_qable(self):
        """是否在连携技"""
        if self.img is None or self.img.size == 0:
            return False
        x, y, w, h = config.Q_COORD
        roi = self.img[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        name, score = detect_image(roi)
        # print(f"chainatk {name} {score:.2f}")
        return name == "qable"

    def _is_victory(self):
        count = 0
        while count < config.TERMINATED_COUNT and (
            np.min(self.hp_agents) > eps and self.hp_boss < eps
        ):
            time.sleep(0.1)
            self.flush()
            count += 1
            print(f"胜利判定次数 {count}")
        if count == config.TERMINATED_COUNT:
            return True
        return False

    def _is_defeat(self):
        count = 0
        while count < config.TERMINATED_COUNT and (
            np.min(self.hp_agents) < eps and self.hp_boss > eps
        ):
            time.sleep(0.1)
            self.flush()
            count += 1
            print(f"死亡判定次数 {count}")
        if count == config.TERMINATED_COUNT:
            return True
        return False

    def game_state(self):
        if self._is_controllable():
            return "running"
        if self._is_esc():
            return "break"
        if self._is_chain_attack():
            print("进入连携技")
            self.key_manager.update()
            self.key_manager.do_action(7, 0.05)
            self.key_manager.update()
            time.sleep(0.1)
            self.key_manager.update()
            self.key_manager.release_all()
            print("连携技完成")
            return "running"
        # if self._is_victory() or self._is_defeat():
        # return "terminated"
        # if np.max(self.hp_agents) < eps and self.hp_boss < eps:
        # return "running"
        print("开大了")
        return "break"

    def _is_switch_able(self):
        """切人能不能用"""
        if self.img is None or self.img.size == 0:
            return False
        x, y, w, h = config.SWITCH_COORD
        roi = self.img[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        name, score = detect_image(roi)
        return not (name == "switchdisable")

    def _is_dodge_able(self):
        """闪避能不能用"""
        if self.img is None or self.img.size == 0:
            return False
        x, y, w, h = config.DODGE_COORD
        roi = self.img[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        name, score = detect_image(roi)
        return name == "dodgeable"

    def _get_action_mask(self):
        """
        表示哪些动作是可用的，one-hot 编码，1可用，0不可用
        """
        mask = np.ones(config.N_ACTIONS, dtype=np.bool_)

        mask[0] = True
        mask[1] = True
        mask[2] = self._is_dodge_able()
        mask[3] = True

        mask[4] = self._is_qable()
        mask[5] = mask[6] = self._is_switch_able()

        return mask

    def _calculate_reward(self):
        boss_hp_diff = self.hp_boss - self.last_hp_boss
        agents_hp_diff = np.sum(self.hp_agents) - np.sum(self.last_hp_agents)

        def scale_change(change, exponent=3):  # 降低扰动带来的影响，小的小，大的大
            if change == 0:
                return 0
            return np.sign(change) * (abs(change) ** exponent)

        boss_hp_diff *= 2
        agents_hp_diff *= 4

        boss_damage_reward = (
            -scale_change(boss_hp_diff, exponent=1.5) * config.BOSS_DMG_REWARD_SCALE
        )
        agent_damage_penalty = (
            scale_change(agents_hp_diff, exponent=2.0) * config.AGENT_DMG_PENALTY_SCALE
        )

        reward = boss_damage_reward + agent_damage_penalty

        # 时间惩罚
        reward -= config.TIME_PENALTY

        # 胜利/败北
        if self._is_victory():
            reward += config.VICTORY_REWARD
        elif self._is_defeat():
            # print("DEFEAT!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            reward += config.DEFEAT_PENALTY

        # TODO: 在这里加入其他奖励，比如操作判定极限闪避、弹刀之类的

        return reward

    def close(self):
        cv2.destroyAllWindows()
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
                self.key_manager.do_action(10, 1.0)
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
