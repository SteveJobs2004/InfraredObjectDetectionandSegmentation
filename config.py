# -*- coding: utf-8 -*-
"""
项目配置文件
"""

# 应用配置
APP_NAME = "视频分割检测系统"
APP_VERSION = "1.0.0"
APP_AUTHOR = "AI Assistant"

# 窗口配置
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
NAVIGATION_PANEL_WIDTH = 200

# 视频配置
DEFAULT_FPS = 30.0
MAX_FPS = 60.0
YOLO_IMAGE_SIZE = 320  # YOLO输入尺寸
MSAM_DOWNSCALE = 0.5   # MSAM下采样比例

# 模型配置
SUPPORTED_MODELS = ["yoloseg", "yolo+msam"]
SUPPORTED_DEVICES = ["cpu", "cuda"]
SUPPORTED_CLASSES = ["all", "person", "car", "plane", "bike"]

# 文件格式
SUPPORTED_VIDEO_FORMATS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"]
SUPPORTED_MODEL_FORMATS = ["*.pt", "*.engine", "*.onnx"]

# 跟踪器配置
TRACKER_ALPHA = 0.5      # EMA系数
MAX_TRACK_AGE = 10       # 最大跟踪年龄
MSAM_INTERVAL = 1        # MSAM调用间隔

# 日志配置
LOG_MAX_LINES = 1000     # 日志最大行数
LOG_TIMESTAMP_FORMAT = "%H:%M:%S"

# 性能配置
FRAME_PROCESSING_TIMEOUT = 1.0  # 帧处理超时时间
SEEK_TIMEOUT = 0.5              # 跳转超时时间
