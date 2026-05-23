import os
import time


def format_time(seconds):
    """格式化时间为 MM:SS 格式"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def get_file_basename(file_path):
    """获取文件路径的基础文件名"""
    if not file_path:
        return "未设置"
    return os.path.basename(file_path)


def validate_file_path(file_path):
    """验证文件路径是否存在"""
    if not file_path:
        return False
    return os.path.isfile(file_path)


def get_supported_video_formats():
    """获取支持的视频格式"""
    return ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"]


def get_supported_model_formats():
    """获取支持的模型格式"""
    return ["*.pt", "*.engine", "*.onnx"]


def create_timestamp():
    """创建时间戳字符串"""
    return time.strftime("%Y%m%d_%H%M%S")


def log_message(message, level="INFO"):
    """格式化日志消息"""
    timestamp = time.strftime("%H:%M:%S")
    return f"[{timestamp}] [{level}] {message}"


def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"
