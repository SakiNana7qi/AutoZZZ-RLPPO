# directkeys.py


import ctypes
import time

"""
0   不动
1   A       O
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

import ctypes
import time

_1 = 0x02
_2 = 0x03
_3 = 0x04
Q = 0x10
W = 0x11
E = 0x12
R = 0x13
T = 0x14
Y = 0x15
U = 0x16
I = 0x17
O = 0x18
P = 0x19
A = 0x1E
S = 0x1F
D = 0x20
F = 0x21
J = 0x24
K = 0x25
C = 0x2E
V = 0x2F
M = 0x32
F = 0x21
LSHIFT = 0x2A
SPACE = 0x39
UP = 0xC8  # 方向键
DOWN = 0xD0
LEFT = 0xCB
RIGHT = 0xCD
ESC = 0x01

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


# Low-level key press/release functions
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


action_dict = [0x00, O, LSHIFT, E, Q, C, SPACE, T, Y, I, W, A, S, D]
action_time = [
    0.0,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.5,
    0.5,
    0.5,
    0.5,
]


class KeyManager:
    """
    要同时处理多个按键操作，所以弄了个非阻塞的按键管理器。
    """

    def __init__(self):
        self.scheduled_releases = {}

    def press(self, key_code, duration=0.1):
        """
        按下按键，以刷新持续时间的方式更新
        """

        is_already_pressed = key_code in self.scheduled_releases

        if not is_already_pressed:
            PressKey(key_code)

        release_time = time.time() + duration
        self.scheduled_releases[key_code] = release_time

    def release(self, key_code):
        """
        立即释放一个之前被按住的按键。
        """
        released = False
        if key_code in self.scheduled_releases:
            del self.scheduled_releases[key_code]
            released = True

        if released:
            ReleaseKey(key_code)

    def update(self):
        """
        刷新一下，释放该释放的
        """
        current_time = time.time()
        for key_code, release_time in list(self.scheduled_releases.items()):
            if current_time >= release_time:
                ReleaseKey(key_code)
                del self.scheduled_releases[key_code]

    def release_all(self):
        """
        全部强制释放
        """
        for key_code in list(self.scheduled_releases.keys()):
            ReleaseKey(key_code)
        self.scheduled_releases.clear()
        print("All keys have been released.")

    def do_action(self, option, time=None):
        """
        以 0~13 编码按下（给外部调用用用的）
        """
        if option == 0:
            return
        self.press(action_dict[option], time if time else action_time[option])


# =============== 鼠标 =================
# 定义鼠标事件常量
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
INPUT_MOUSE = 0


def _send_mouse_input(flags, dx=0, dy=0, data=0):
    """一个通用的函数，用于发送鼠标输入事件。"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    # 注意这里需要使用 dx, dy, mouseData, dwFlags 这些字段
    ii_.mi = MouseInput(dx, dy, data, flags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(INPUT_MOUSE), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def move_mouse_to(x, y):
    """
    将鼠标移动到屏幕的绝对像素坐标 (x, y)。
    """
    # 获取屏幕分辨率
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    # 转换为 0-65535 的绝对坐标
    absolute_x = (x * 65535) // screen_width
    absolute_y = (y * 65535) // screen_height

    # 构建并发送鼠标移动事件
    flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE
    _send_mouse_input(flags, dx=absolute_x, dy=absolute_y)


def left_click():
    """
    在当前鼠标位置执行一次左键单击。
    """
    _send_mouse_input(MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.05)  # 模拟真实的点击间隔
    _send_mouse_input(MOUSEEVENTF_LEFTUP)


def get_cursor_pos():
    """获取当前鼠标的屏幕坐标"""

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def move_to_absolute(target_x, target_y):
    """
    使用绝对坐标将鼠标移动到目标位置。
    """
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    absolute_x = (target_x * 65535) // screen_width
    absolute_y = (target_y * 65535) // screen_height

    flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE
    _send_mouse_input(flags, dx=absolute_x, dy=absolute_y)


def move_to_and_click_relative(dx, dy):
    """
    相对移动，然后左键一下
    """
    current_x, current_y = get_cursor_pos()
    target_x, target_y = current_x + dx, current_y + dy
    move_to_absolute(target_x, target_y)
    time.sleep(0.01)
    left_click()
