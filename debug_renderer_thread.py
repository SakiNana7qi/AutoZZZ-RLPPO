# debug_renderer_thread.py

import threading
import cv2
import time
import numpy as np
from myutils import put_text_chinese
import config
from PIL import ImageFont
import copy


class DebugRendererThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)  # 设置为守护线程，主线程退出时它会自动退出
        self.latest_data = {}
        self.lock = threading.Lock()  # 线程锁，用于安全地更新数据
        self.running = True

    def update_data(self, new_data):
        with self.lock:
            self.latest_data.update(new_data)

    def stop(self):
        self.running = False

    def run(self):  # 线程的主体
        window_name = "AutoZZZ Debug Status"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        font = ImageFont.truetype(config.font_path, 18)

        while self.running:
            with self.lock:
                # 复制数据以避免在渲染时数据被修改
                data_to_render = self.latest_data.copy()

            debug_panel = np.zeros((500, 400, 3), dtype=np.uint8)
            line_height = 25
            x_pos = 10

            reward = data_to_render.get("reward", 0)
            reward_color = (0, 255, 0) if reward > 0 else (0, 0, 255)
            debug_panel = put_text_chinese(
                debug_panel,
                f"Reward: {reward:+.4f}",
                (x_pos, line_height * 0),
                font,
                reward_color,
            )

            action = data_to_render.get("action", -1)
            debug_panel = put_text_chinese(
                debug_panel,
                f"Action: {config.ACTION_NAME[action]}",
                (x_pos, line_height * 1),
                font,
                (255, 255, 255),
            )

            action_history = data_to_render.get(
                "action_history",
                [
                    -1,
                ],
            )
            history_str = " ".join(map(str, action_history))
            debug_panel = put_text_chinese(
                debug_panel,
                f"Action History: {history_str}",
                (x_pos, line_height * 3),
                font,
                (200, 200, 200),
            )

            hp_boss = data_to_render.get("hp_boss", -1)

            debug_panel = put_text_chinese(
                debug_panel,
                f"Boss HP: {hp_boss:.2f}",
                (x_pos, line_height * 4),
                font,
                (0, 165, 255),
            )
            hp_agents = data_to_render.get("hp_agents", [-1, -1, -1])
            agent_hps_str = ", ".join([f"{hp:.2f}" for hp in hp_agents])
            debug_panel = put_text_chinese(
                debug_panel,
                f"Agents HP: {agent_hps_str}",
                (x_pos, line_height * 5),
                font,
                (0, 255, 0),
            )

            action_mask = data_to_render.get("action_mask", "N/A")

            debug_panel = put_text_chinese(
                debug_panel,
                f"Action Mask:",
                (x_pos, line_height * 7),
                font,
                (255, 255, 255),
            )

            if action_mask is not None:
                action_mask_position = x_pos
                for i, m in enumerate(action_mask):
                    action_mask_str = config.ACTION_NAME[i]
                    action_mask_color = (0, 255, 0) if m else (0, 0, 255)
                    debug_panel = put_text_chinese(
                        debug_panel,
                        action_mask_str,
                        (action_mask_position, line_height * 8),
                        font,
                        action_mask_color,
                    )
                    action_mask_position += 30 if i != 5 else 60
            else:
                mask_str = "N/A"
                debug_panel = put_text_chinese(
                    debug_panel,
                    mask_str,
                    (x_pos, line_height * 8),
                    font,
                    (255, 255, 0),
                )

            status_str = "N/A"
            status_color = (255, 0, 0)
            ui_state = data_to_render.get("ui_state", None)
            if not ui_state == None:
                if ui_state.get("is_controllable", False):
                    status_str = "Running"
                    status_color = (255, 255, 0)
                elif ui_state.get("is_esc", False):
                    status_str = "ESC"
                    status_color = (255, 0, 0)
                elif ui_state.get("is_chain_attack", False):
                    status_str = "连携技"
                    status_color = (255, 0, 0)

            debug_panel = put_text_chinese(
                debug_panel,
                "状态: " + status_str,
                (x_pos, line_height * 10),
                font,
                status_color,
            )

            debug_panel = put_text_chinese(
                debug_panel,
                f"Steps: {data_to_render.get("timesteps", -1):06}",
                (x_pos, line_height * 11),
                font,
                (255, 255, 255),
            )

            # frame_stack

            debug_panel = put_text_chinese(
                debug_panel,
                "Frame Stack (T-3 to T):",
                (x_pos, line_height * 13),
                font,
                (255, 255, 255),
            )

            frame_stack = data_to_render.get("frame_stack")
            if frame_stack:
                thumb_w, thumb_h = 80, 45
                spacing = 10

                thumbnails = []
                for frame in frame_stack:
                    if frame.ndim == 2:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_bgr = frame

                    thumbnail = cv2.resize(frame_bgr, (thumb_w, thumb_h))
                    thumbnails.append(thumbnail)

                while len(thumbnails) < config.FRAME_STACK_SIZE:
                    thumbnails.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

                try:
                    h_stack = cv2.hconcat(thumbnails)

                    y_offset = int(line_height * 14)
                    debug_panel[
                        y_offset : y_offset + thumb_h, x_pos : x_pos + h_stack.shape[1]
                    ] = h_stack
                except cv2.error as e:
                    print(f"[Error] cv2.error: {e}")

            episode_reward = data_to_render.get("episode_reward", 0)
            episode_reward_color = (0, 255, 0) if episode_reward > 0 else (0, 0, 255)
            debug_panel = put_text_chinese(
                debug_panel,
                f"Episode Reward: {episode_reward:+.4f}",
                (x_pos, line_height * 17),
                font,
                episode_reward_color,
            )

            cv2.imshow(window_name, debug_panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.05)  # 20fps

        cv2.destroyAllWindows()
        print("调试线程已关闭。")
