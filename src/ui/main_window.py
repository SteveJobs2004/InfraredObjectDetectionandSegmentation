import sys
import cv2
import time
import torch
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont

from ..core.video_thread import VideoThread
from .navigation import NavigationPanel
from .controls import ControlPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.current_video_path = ""
        self.yolo_weights_path = ""  # 选中的 YOLO 权重（.pt 或 .engine）
        self.msam_weights_path = ""  # 选中的 MSAM 权重（.pt）
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("视频分割检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧导航栏
        self.navigation_panel = NavigationPanel()
        main_layout.addWidget(self.navigation_panel)
        
        # 右侧主内容区
        self.create_main_content(main_layout)
        
        # 根据默认选择的模型，立即刷新一次权重控件的显隐与文案
        try:
            self.on_model_changed(self.navigation_panel.model_combo.currentText())
        except Exception:
            pass
        
    def create_main_content(self, parent_layout):
        """创建右侧主内容区"""
        content_frame = QFrame()
        content_frame.setFrameStyle(QFrame.Box)
        
        content_layout = QVBoxLayout(content_frame)
        
        # 视频显示区
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("请选择功能开始检测")
        content_layout.addWidget(self.video_label)
        
        # 控制面板
        self.control_panel = ControlPanel(content_layout)
        
        parent_layout.addWidget(content_frame)
        
    def connect_signals(self):
        """连接信号槽"""
        # 导航面板信号
        self.navigation_panel.realtime_btn.clicked.connect(self.switch_to_realtime)
        self.navigation_panel.video_btn.clicked.connect(self.switch_to_video)
        self.navigation_panel.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.navigation_panel.device_combo.currentTextChanged.connect(self.on_device_changed)
        self.navigation_panel.class_combo.currentTextChanged.connect(self.on_class_changed)
        self.navigation_panel.yolo_btn.clicked.connect(self.select_yolo_weights)
        self.navigation_panel.msam_btn.clicked.connect(self.select_msam_weights)
        
        # 控制面板信号
        self.control_panel.video_select_btn.clicked.connect(self.select_video)
        self.control_panel.start_btn.clicked.connect(self.start_detection)
        self.control_panel.pause_btn.clicked.connect(self.pause_detection)
        self.control_panel.stop_btn.clicked.connect(self.stop_detection)
        self.control_panel.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.control_panel.progress_slider.sliderReleased.connect(self.slider_released)
        self.control_panel.progress_slider.valueChanged.connect(self.slider_value_changed)
        
    def switch_to_realtime(self):
        """切换到实时监测模式"""
        self.control_panel.switch_to_realtime()
        self.video_label.setText("实时监测模式\n请点击开始按钮")
        
    def switch_to_video(self):
        """切换到视频检测模式"""
        self.control_panel.switch_to_video()
        self.video_label.setText("视频检测模式\n请选择视频文件")
        
    def select_video(self):
        """选择视频文件"""
        file_path = self.control_panel.select_video()
        if file_path:
            self.current_video_path = file_path
            
    def on_model_changed(self, model_name):
        """模型改变时的处理"""
        self.control_panel.log_status(f"模型已切换到: {model_name}")

        show_msam = (model_name == "yolo+msam")
        self.navigation_panel.msam_path_label.setVisible(show_msam)
        self.navigation_panel.msam_btn.setVisible(show_msam)
        
        if model_name == "yoloseg":
            if self.navigation_panel.yolo_path_label.text().startswith("YOLO权重"):
                suffix = self.navigation_panel.yolo_path_label.text().split(":", 1)[-1]
                self.navigation_panel.yolo_path_label.setText(f"YOLOSeg权重:{suffix}")
            self.navigation_panel.yolo_btn.setText("选择 YOLOSeg 权重 (.pt/.engine)")
        else:
            if self.navigation_panel.yolo_path_label.text().startswith("YOLOSeg权重"):
                suffix = self.navigation_panel.yolo_path_label.text().split(":", 1)[-1]
                self.navigation_panel.yolo_path_label.setText(f"YOLO权重:{suffix}")
            self.navigation_panel.yolo_btn.setText("选择 YOLO 权重 (.pt/.engine)")

        if self.control_panel.is_playing:
            self.control_panel.log_status("模型切换已记录，请点击暂停后再开始以应用新模型")
        
    def on_device_changed(self, device_name: str):
        """设备切换回调"""
        self.current_device = device_name
        self.control_panel.log_status(f"设备切换为: {device_name}")
        if self.video_thread is not None:
            self.video_thread.set_device(device_name)
            
    def start_detection(self):
        """开始检测"""
        if self.control_panel.current_mode == "video" and not self.current_video_path:
            self.control_panel.log_status("请先选择视频文件")
            return
            
        if self.video_thread and self.video_thread.paused:
            self.video_thread.resume()
            self.control_panel.set_playing_state(True)
            self.control_panel.log_status("检测已恢复")
            return
            
        self.video_thread = VideoThread(self.navigation_panel.model_combo.currentText())
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.video_thread.fps_signal.connect(self.control_panel.update_fps)
        self.video_thread.progress_signal.connect(self.control_panel.update_progress) 
        
        # 设置设备优先为 CUDA（Jetson），可通过左侧下拉切换
        self.video_thread.set_device(self.current_device)
        
        # 启动前校验 engine 权重与设备
        if self.yolo_weights_path.lower().endswith('.engine') and self.current_device != 'cuda':
            self.control_panel.log_status("已选择 .engine 权重但当前设备为 CPU，无法运行。请切换设备到 cuda。")
            return

        # 设置模型
        if self.navigation_panel.model_combo.currentText() == "yoloseg":
            self.video_thread.set_model("yoloseg", self.yolo_weights_path)
        else:
            self.video_thread.set_model("yolo+msam", self.yolo_weights_path, self.msam_weights_path)

        # 设置类别过滤
        self.video_thread.set_class_filter(self.navigation_panel.class_combo.currentText())
        
        # 设置视频源
        if self.control_panel.current_mode == "realtime":
            success = self.video_thread.set_video_source("camera")
        else:
            success = self.video_thread.set_video_source(self.current_video_path)
            
        if not success:
            self.control_panel.log_status("无法打开视频源")
            return
            
        # 启动线程
        self.video_thread.start()
        
        self.control_panel.set_playing_state(True)
        
        def _basename(p):
            if not p:
                return "未设置"
            return p.split('/')[-1] if '/' in p else p.split('\\')[-1]
        model_name = self.navigation_panel.model_combo.currentText()
        if model_name == 'yoloseg':
            self.control_panel.log_status(f"检测已开始 | 模型: yoloseg | YOLOSeg权重: {_basename(self.yolo_weights_path)} | 过滤: {self.navigation_panel.class_combo.currentText()}")
        else:
            self.control_panel.log_status(f"检测已开始 | 模型: yolo+msam | YOLO权重: {_basename(self.yolo_weights_path)} | MSAM权重: {_basename(self.msam_weights_path)} | 过滤: {self.navigation_panel.class_combo.currentText()}")
        
    def pause_detection(self):
        """暂停检测"""
        if self.video_thread:
            self.video_thread.pause() 
            
        self.control_panel.set_playing_state(False)
        self.video_label.setText("检测已暂停")
        self.control_panel.log_status("检测已暂停")

    def stop_detection(self):
        """停止检测"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            
        self.control_panel.set_playing_state(False)
        self.video_label.setText("检测已停止")
        self.control_panel.log_status("检测已停止")

    def select_yolo_weights(self):
        title = "选择 YOLOSeg 权重" if self.navigation_panel.model_combo.currentText() == "yoloseg" else "选择 YOLO 权重"
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, "", "模型文件 (*.pt *.engine)"
        )
        if file_path:
            self.yolo_weights_path = file_path
            base_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
            if self.navigation_panel.model_combo.currentText() == "yoloseg":
                self.navigation_panel.yolo_path_label.setText(f"YOLOSeg权重: {base_name}")
                self.control_panel.log_status(f"已选择YOLOSeg权重: {base_name}")
            else:
                self.navigation_panel.yolo_path_label.setText(f"YOLO权重: {base_name}")
                self.control_panel.log_status(f"已选择YOLO权重: {base_name}")

    def select_msam_weights(self):
        """选择 MobileSAM 权重（.pt）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 MSAM 权重", "", "模型文件 (*.pt)"
        )
        if file_path:
            self.msam_weights_path = file_path
            base_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
            self.navigation_panel.msam_path_label.setText(f"MSAM权重: {base_name}")
            self.control_panel.log_status(f"已选择MSAM权重: {base_name}")
        
    def update_video_frame(self, frame):
        """更新视频帧显示"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
    def slider_pressed(self):
        """滑块被按下"""
        self.control_panel.slider_pressed()
        
    def slider_released(self):
        """滑块被释放"""
        self.control_panel.slider_released()
        if self.video_thread and self.video_thread.is_video_file:
            # 计算目标帧位置
            target_frame = int((self.control_panel.progress_slider.value() / 1000.0) * self.video_thread.total_frames)
            # 跳转到指定帧
            self.video_thread.seek_video(target_frame)
    
    def slider_value_changed(self, value):
        """滑块值改变"""
        self.control_panel.slider_value_changed(value)
        if self.video_thread and self.video_thread.is_video_file and self.control_panel.slider_is_pressed:
            fps = self.video_thread.source_fps
            total_frames = self.video_thread.total_frames
            target_frame = int((value / 1000.0) * total_frames)
            current_time = target_frame / fps
            total_time = total_frames / fps
            
            current_min = int(current_time // 60)
            current_sec = int(current_time % 60)
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            
            self.control_panel.time_label.setText(f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
        
    def on_class_changed(self, class_name: str):
        # 更新线程中的过滤器，让新选择马上影响后续帧
        try:
            if self.video_thread is not None:
                self.video_thread.set_class_filter(class_name)
            self.control_panel.log_status(f"类别过滤切换为: {class_name}")
        except Exception as e:
            self.control_panel.log_status(f"类别过滤切换失败: {e}")
        
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        event.accept()
