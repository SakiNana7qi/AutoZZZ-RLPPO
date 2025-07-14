# myutils.py

import ctypes
import win32gui
import time


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
            time.sleep(1)
            return windows[choice][0]
    except (ValueError, IndexError):
        pass

    print("无效的选择。")
    return None
