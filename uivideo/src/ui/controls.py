import time
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QTextEdit, QFileDialog)
from PyQt5.QtCore import Qt
import cv2


class ControlPanel:
    """控制面板"""
    
    def __init__(self, parent_layout):
        self.parent_layout = parent_layout
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 进度条
        self.progress_layout = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(100)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(1000)  # 使用1000作为最大值，实现更精细的控制
        self.progress_slider.setVisible(False)  # 初始隐藏
        
        self.progress_layout.addWidget(self.time_label)
        self.progress_layout.addWidget(self.progress_slider)
        self.parent_layout.addLayout(self.progress_layout)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 视频选择（仅视频检测模式显示）
        self.video_select_btn = QPushButton("选择视频")
        self.video_select_btn.setVisible(False)
        control_layout.addWidget(self.video_select_btn)
        
        # 当前视频名称显示
        self.video_name_label = QLabel("未选择视频")
        self.video_name_label.setVisible(False)
        control_layout.addWidget(self.video_name_label)
        
        control_layout.addStretch()
        
        # 播放控制
        self.start_btn = QPushButton("开始")
        self.start_btn.setEnabled(False)  # 初始禁用
        control_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)  # 初始禁用
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("结束")
        self.stop_btn.setEnabled(False)  # 初始禁用
        control_layout.addWidget(self.stop_btn)
        
        # FPS显示
        self.fps_label = QLabel("FPS: 0.0")
        control_layout.addWidget(self.fps_label)
        
        self.parent_layout.addLayout(control_layout)
        
        # 状态信息
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlaceholderText("状态信息...")
        self.parent_layout.addWidget(self.status_text)
        
        # 设置默认状态
        self.current_mode = "realtime"
        self.is_playing = False
        self.slider_is_pressed = False  
        
        # 初始化按钮状态
        self.start_btn.setEnabled(True)  
        self.pause_btn.setEnabled(False)  
        self.stop_btn.setEnabled(False)
        
    def switch_to_realtime(self):
        """切换到实时监测模式"""
        self.current_mode = "realtime"
        self.video_select_btn.setVisible(False)
        self.video_name_label.setVisible(False)
        self.progress_slider.setVisible(False)  
        self.time_label.setVisible(False)  
        
        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.log_status("切换到实时监测模式")
        
    def switch_to_video(self):
        """切换到视频检测模式"""
        self.current_mode = "video"
        self.video_select_btn.setVisible(True)
        self.video_name_label.setVisible(True)
        self.progress_slider.setVisible(True)  
        self.time_label.setVisible(True)  
        
        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.log_status("切换到视频检测模式")
        
    def select_video(self):
        """选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            None, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            video_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
            self.video_name_label.setText(f"当前视频: {video_name}")
            self.log_status(f"已选择视频: {video_name}")
            self.start_btn.setEnabled(True)
            return file_path
        return None
            
    def update_fps(self, fps):
        """更新FPS显示"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
    def update_progress(self, current_frame, total_frames):
        """更新进度条显示"""
        if total_frames > 0 and not self.slider_is_pressed:
            progress = int((current_frame / total_frames) * 1000) 
            self.progress_slider.setValue(progress)
            
            # 计算时间显示
            fps = 30.0  # 默认帧率，实际应该从视频线程获取
            current_time = current_frame / fps
            total_time = total_frames / fps
            
            current_min = int(current_time // 60)
            current_sec = int(current_time % 60)
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            
            self.time_label.setText(f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
    
    def slider_pressed(self):
        """滑块被按下"""
        self.slider_is_pressed = True
        
    def slider_released(self):
        """滑块被释放"""
        self.slider_is_pressed = False
        
    def slider_value_changed(self, value):
        """滑块值改变"""
        if self.slider_is_pressed:
            # 这里可以添加预览时间显示逻辑
            pass
        
    def log_status(self, message):
        """记录状态信息"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        
    def set_playing_state(self, is_playing):
        """设置播放状态"""
        self.is_playing = is_playing
        if is_playing:
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
