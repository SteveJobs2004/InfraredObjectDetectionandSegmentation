import torch
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class NavigationPanel(QFrame):
    """左侧导航面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setFrameStyle(QFrame.Box)
        self.setMaximumWidth(200)
        self.setMinimumWidth(200)
        
        nav_layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("功能选择")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(title_label)
        
        # 导航按钮
        self.realtime_btn = QPushButton("实时监测")
        self.realtime_btn.setMinimumHeight(50)
        nav_layout.addWidget(self.realtime_btn)
        
        self.video_btn = QPushButton("视频检测")
        self.video_btn.setMinimumHeight(50)
        nav_layout.addWidget(self.video_btn)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        # 统一为线程中使用的类型标识：yoloseg / yolo+msam
        self.model_combo.addItems(["yoloseg", "yolo+msam"])
        model_layout.addWidget(self.model_combo)
        
        nav_layout.addWidget(model_group)
        
        # 设备与权重路径显示
        weights_group = QGroupBox("权重路径")
        weights_layout = QVBoxLayout(weights_group)
        
        # 设备选择
        device_row = QHBoxLayout()
        device_label = QLabel("设备:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda" if torch.cuda.is_available() else "cpu", "cpu"])  # 默认优先 cuda
        # 去重，防止在没 CUDA 时出现两个 cpu
        if torch.cuda.is_available():
            pass
        else:
            self.device_combo.removeItem(0)
            self.device_combo.setCurrentText("cpu")
        device_row.addWidget(device_label)
        device_row.addWidget(self.device_combo)
        weights_layout.addLayout(device_row)

        # 权重选择按钮与路径
        self.yolo_path_label = QLabel("YOLO权重: 未设置")
        self.msam_path_label = QLabel("MSAM权重: 未设置")
        self.yolo_btn = QPushButton("选择 YOLO 权重 (.pt/.engine)")
        self.msam_btn = QPushButton("选择 MSAM 权重 (.pt)")
        
        weights_layout.addWidget(self.yolo_path_label)
        weights_layout.addWidget(self.msam_path_label)
        weights_layout.addWidget(self.yolo_btn)
        weights_layout.addWidget(self.msam_btn)
        
        nav_layout.addWidget(weights_group)

        # 类别过滤
        class_group = QGroupBox("类别过滤")
        class_layout = QHBoxLayout(class_group)
        class_label = QLabel("只检测:")
        self.class_combo = QComboBox()
        self.class_combo.addItems(["all", "person", "car", "plane", "bike"])
        class_layout.addWidget(class_label)
        class_layout.addWidget(self.class_combo)
        nav_layout.addWidget(class_group)
        
        nav_layout.addStretch()
