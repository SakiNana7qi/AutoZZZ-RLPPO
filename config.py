# config.py

# --- 环境配置 ---
IMG_WIDTH = 512  # 缩小后的图像宽度
IMG_HEIGHT = 288  # 缩小后的图像高度
N_ACTIONS = 7  # 动作空间大小 WASD10111213 连携上下取消789
ACTION_HISTORY_LEN = 8  # 历史动作记录长度
FRAME_STACK_SIZE = 4  # 帧堆叠的数量
font_path = "./fonts/msyh.ttc"
UI_UPDATE_INTERVAL = 0.05  # 每秒最多更新20次UI

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
BOSS_HP_TOP_COORD = [1187, 45, 436, 1]  # 用来判定 boss 似了（阶段性）没的
BOSS_HP_LENGTH = 1317
BATTLE_START_WAIT_TIME = 2  # 战斗前等待时间，加载动画和过渡效果
MOUSE_ESC_TO_RESTART_WIDTH = 425  # 重开鼠标移动相对距离
MOUSE_ESC_TO_RESTART_HEIGHT = 480
MOUSE_RESTART_TO_CONFIRM_WIDTH = -300
MOUSE_RESTART_TO_CONFIRM_HEIGHT = -375
TERMINATED_COUNT = 4  # 胜利/死亡判定次数
ACTION_NAME = ["空", "A", "闪", "E", "Q", "切人上", "切人下"]

# --- Reward ---
BOSS_DMG_REWARD_SCALE = 0.1  # 对 boss 造成伤害的奖励系数
AGENT_DMG_PENALTY_SCALE = 0.1  # 代理人受到伤害的惩罚系数
TIME_PENALTY = 0.001  # 每一步的时间惩罚
VICTORY_REWARD = 2.0  # 胜利大奖励
DEFEAT_PENALTY = -2.0  # 失败大惩罚
ACTION_COST = 0.002  # 执行非空动作的成本
REPETITION_THRESHOLD = 20  # 重复 x 次开始惩罚
REPETITION_PENALTY_SCALE = 0.001  # 惩罚系数
PERFECT_DODGE_REWARD = 0.001  # 特殊招式额外奖励 Reward Shaping
ASSIST_ATTACK_REWARD = 0.001
DECIBEL_REWARD = 0.002


# --- PPO 训练配置 ---
MAX_TIMESTEPS = 2000000  # 总训练步数
UPDATE_INTERVAL = 1024  # 每收集多少步数据后进行一次更新
SAVE_INTERVAL = 10  # 每多少个 episode 保存一次模型

# --- PPO 算法超参数 ---
LEARNING_RATE = 1e-5
GAMMA = 0.99  # γ
GAE_LAMBDA = 0.99  # GAE 参数
POLICY_CLIP = 0.2  # PPO 裁剪范围
PPO_EPOCHS = 10  # 每次更新时，对数据进行优化的轮数
BATCH_SIZE = 64  # 每轮优化中的 batch_size
VF_COEF = 0.5  # 价值函数损失的系数
ENT_COEF = 0.005  # 熵奖励的系数

# --- YOLO 参数 ---
conf_threshold = 0.45  # 置信度
Q_COORD = [1736, 787, 83, 83]  # 终结技区域
CHAINATK_COORD = [606, 871, 708, 159]  # 连携技区域
BACK_PAUSE_COORD = [46, 0, 135, 122]  # 左上角暂停和 ESC 界面返回 区域
INVESTIGATION_COMPLETE_COORD = [362, 477, 1180, 134]  # 胜利判定区域
DECIBEL_COORD = [189, 200, 207, 38]  # 招式判定区域（喧响值下面那个）
