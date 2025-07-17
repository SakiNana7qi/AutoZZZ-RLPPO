# debug_renderer.py

import cv2
import numpy as np
from multiprocessing.connection import Connection
import time
from myutils import put_text_chinese
import config


def renderer_process(pipe: Connection):
    """
    这是一个独立的进程，负责接收数据并渲染调试窗口。
    :param pipe: 用于接收数据的管道连接对象。
    :param font_path: 字体文件的路径。
    """
    window_name = "AutoZZZ Debug Status"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    font_path = config.font_path

    latest_data = {}

    while True:
        try:
            # [非阻塞] 检查管道中是否有新数据
            if pipe.poll():
                data = pipe.recv()
                if data == "TERMINATE":  # 收到终止信号
                    break
                latest_data.update(data)

            debug_panel = np.zeros((400, 500, 3), dtype=np.uint8)
            font_size = 18
            line_height = 25
            x_pos = 10

            reward = latest_data.get("reward", 0)
            reward_color = (0, 255, 0) if reward > 0 else (0, 0, 255)
            debug_panel = put_text_chinese(
                debug_panel,
                f"Reward: {reward:.4f}",
                (x_pos, line_height * 0),
                font_path,
                font_size,
                reward_color,
            )

            action = latest_data.get("action", -1)
            debug_panel = put_text_chinese(
                debug_panel,
                f"Action: {config.ACTION_NAME[action]}",
                (x_pos, line_height * 1),
                font_path,
                font_size,
                (255, 255, 255),
            )

            action_history = latest_data.get(
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
                font_path,
                font_size,
                (200, 200, 200),
            )

            hp_boss = latest_data.get("hp_boss", -1)

            debug_panel = put_text_chinese(
                debug_panel,
                f"Boss HP: {hp_boss:.2f}",
                (x_pos, line_height * 4),
                font_path,
                font_size,
                (0, 165, 255),
            )
            hp_agents = latest_data.get("hp_agents", [-1, -1, -1])
            agent_hps_str = ", ".join([f"{hp:.2f}" for hp in hp_agents])
            debug_panel = put_text_chinese(
                debug_panel,
                f"Agents HP: {agent_hps_str}",
                (x_pos, line_height * 5),
                font_path,
                font_size,
                (0, 255, 0),
            )

            action_mask = latest_data.get("action_mask", "N/A")

            debug_panel = put_text_chinese(
                debug_panel,
                f"Action Mask:",
                (x_pos, line_height * 7),
                font_path,
                font_size,
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
                        font_path,
                        font_size,
                        action_mask_color,
                    )
                    action_mask_position += 30 if i != 5 else 60
            else:
                mask_str = "N/A"
                debug_panel = put_text_chinese(
                    debug_panel,
                    mask_str,
                    (x_pos, line_height * 8),
                    font_path,
                    font_size,
                    (255, 255, 0),
                )

            status_str = "N/A"
            status_color = (255, 0, 0)
            ui_state = latest_data.get("ui_state", None)
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
                font_path,
                font_size,
                status_color,
            )

            cv2.imshow(window_name, debug_panel)

            # 等待一小段时间
            if cv2.waitKey(1) & 0xFF == ord("q"):  # 按 q 退出
                break

            time.sleep(0.05)  # 20 fps

        except (EOFError, BrokenPipeError):
            # 如果主进程意外关闭了管道，也退出循环
            print("调试窗口：主进程已关闭，窗口将关闭。")
            break
        except Exception as e:
            print(f"调试窗口发生错误: {e}")
            break

    cv2.destroyAllWindows()
    print("调试窗口已关闭。")
