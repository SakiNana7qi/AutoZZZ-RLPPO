# config.py

# --- 环境配置 ---
IMG_WIDTH = 512  # 缩小后的图像宽度
IMG_HEIGHT = 288  # 缩小后的图像高度
N_ACTIONS = 7  # 动作空间大小 WASD10111213 连携上下取消789

# --- 游戏交互配置 ---
BATTLE_START_WAIT_TIME = 0.1  # (秒) reset后等待战斗稳定的时间
ACTION_DELAY = 0.1  # (秒) 执行动作后等待游戏响应的时间
HP_COORD = [
    [260, 52, 314, 29],
    [656, 52, 98, 11],
    [838, 52, 98, 11],
]  # 三个代理人的血量坐标 ，1920x1080， [left,top,width,height]
HP_LENGTH = [414, 188, 189]  # 血条像素长度
BOSS_HP_COORD = [1141, 51, 571, 11]  # boss 的
BOSS_HP_LENGTH = 1317
BATTLE_START_WAIT_TIME = 3  # 战斗前等待时间，加载动画和过渡效果
MOUSE_ESC_TO_RESTART_WIDTH = 425  # 重开鼠标移动相对距离
MOUSE_ESC_TO_RESTART_HEIGHT = 480
MOUSE_RESTART_TO_CONFIRM_WIDTH = -280
MOUSE_RESTART_TO_CONFIRM_HEIGHT = -418

# --- Reward ---
BOSS_DMG_REWARD_SCALE = 0.03  # 对 boss 造成伤害的奖励系数
AGENT_DMG_PENALTY_SCALE = 0.1  # 代理人受到伤害的惩罚系数
TIME_PENALTY = 0.001  # 每一步的时间惩罚
VICTORY_REWARD = 10.0  # 胜利大奖励
DEFEAT_PENALTY = -10.0  # 失败大惩罚
ACTION_COST = 0.005  # 执行非空动作的成本
REPETITION_THRESHOLD = 20  # 重复 x 次开始惩罚
REPETITION_PENALTY_SCALE = 0.001  # 惩罚系数
# TODO: 添加其他奖励系数
# PERFECT_DODGE_REWARD = 5.0


# --- PPO 训练配置 ---
MAX_TIMESTEPS = 1000000  # 总训练步数
UPDATE_INTERVAL = 1024  # 每收集多少步数据后进行一次更新
SAVE_INTERVAL = 10  # 每多少个 episode 保存一次模型

# --- PPO 算法超参数 ---
LEARNING_RATE = 1.5e-4
GAMMA = 0.99  # γ
GAE_LAMBDA = 0.95  # GAE 参数
POLICY_CLIP = 0.2  # PPO 裁剪范围
PPO_EPOCHS = 10  # 每次更新时，对数据进行优化的轮数
BATCH_SIZE = 64  # 每轮优化中的 batch_size
VF_COEF = 0.5  # 价值函数损失的系数
ENT_COEF = 0.1  # 熵奖励的系数

# --- YOLO 参数 ---
conf_threshold = 0.45  # 置信度
ESC_COORD = [1650, 67, 203, 59]  # ESC 界面 “设置” 区域
CLOCK_COORD = [1690, 43, 150, 150]  # 计时器区域
Q_COORD = [1736, 787, 83, 83]  # 终结技区域
CHAINATK_COORD = [606, 871, 708, 159]  # 连携技区域
SWITCH_COORD = [1737, 929, 183, 126]  # 切人区域
DODGE_COORD = [1522, 923, 84, 133]  # 闪避区域
