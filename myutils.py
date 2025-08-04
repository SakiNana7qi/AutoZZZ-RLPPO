# myutils.py

import ctypes
import win32gui
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_dpi_scale_factor(hwnd: int) -> float:
    """获取指定窗口的DPI缩放比例。"""
    try:
        # 推荐使用 GetDpiForWindow (Windows 10 1607+)
        dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        return dpi / 96.0  # 标准DPI为96
    except AttributeError:
        # 如果API不存在（例如在旧版Windows上），则回退
        try:
            hDC = ctypes.windll.user32.GetDC(hwnd)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hDC, 88)  # 88 = LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(hwnd, hDC)
            return dpi / 96.0
        except Exception:
            return 1.0  # 如果所有方法都失败，则返回1.0（无缩放）


def select_target_window() -> int | None:
    """
    显示当前可见窗口列表，并让用户通过命令行选择一个。
    返回所选窗口的句柄(hwnd)，如果选择无效则返回None。
    """
    windows = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            windows.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(callback, None)

    if not windows:
        print("[Error] 未找到任何可见窗口。")
        return None

    r_windows = []

    for i, (hwnd, title) in enumerate(windows):
        if "绝区零" in title:
            r_windows.append([hwnd, title])

    windows = r_windows

    print("\n请选择要捕捉的窗口:")
    for i, (hwnd, title) in enumerate(windows):
        print(f"[{i}] {title}")

    try:
        choice_str = input("请输入窗口编号: ")
        if not choice_str:  # 用户直接回车
            print("未做选择，程序退出。")
            return None
        choice = int(choice_str)
        if 0 <= choice < len(windows):
            time.sleep(2)
            return windows[choice][0]
    except (ValueError, IndexError):
        pass

    print("无效的选择。")
    return None


def put_text_chinese(image, text, position, font, color):
    """
    在OpenCV图像上绘制支持中文的文本。
    :param image: OpenCV图像 (NumPy ndarray)
    :param text: 要绘制的文本 (可以是中文)
    :param position: 文本左上角的坐标 (x, y)
    :param font 字体
    :param color: 文本颜色，格式为 BGR (e.g., (255, 0, 0) for blue)
    :return: 绘制了文本的OpenCV图像
    """
    # 将OpenCV图像 (BGR) 转换为Pillow图像 (RGB)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)

    # 在指定位置绘制文本
    # 注意：Pillow的颜色格式是RGB，但我们传入的color是BGR，需要转换
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    # 将Pillow图像 (RGB) 转换回OpenCV图像 (BGR)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def get_human_time():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
