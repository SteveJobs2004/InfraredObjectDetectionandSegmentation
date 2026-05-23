# 视频分割检测系统



## 功能特性

### 核心功能
- **双模型支持**
  - **YOLO-Seg**: 端到端分割，速度快，适合实时场景
  - **YOLO + MobileSAM**: 联合推理，精度高，边缘更精细
  
- **多种检测模式**
  - 实时摄像头监测
  - 视频文件检测（支持 MP4/AVI/MOV/MKV/WMV/FLV）
  
- **智能追踪与平滑**
  - IoU 匹配算法实现目标追踪
  - EMA (指数移动平均) 平滑框抖动
  - 隔帧优化提升 FPS

### 高级特性
- **多设备支持**: CPU / CUDA GPU
- **类别过滤**: 支持 person, car, plane, bike 等类别筛选
- **TensorRT 加速**: 支持 `.engine` 格式模型
- **实时性能监控**: FPS 显示、处理时间统计
- **视频控制**: 播放/暂停、进度条拖动、帧跳转

---

## 系统要求

### 基础环境
- Python 3.7+
- Windows / Linux / macOS

### 硬件建议
- **CPU 模式**: 4 核心以上处理器
- **GPU 模式**: NVIDIA GPU (支持 CUDA 11.0+)，显存 ≥ 4GB

---

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/SteveJobs2004/InfraredObjectDetectionandSegmentation.git
cd InfraredObjectDetectionandSegmentation

```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

**依赖包列表**:
- PyQt5 >= 5.15.0
- opencv-python >= 4.5.0
- torch >= 1.9.0
- numpy >= 1.21.0
- ultralytics >= 8.0.0
- mobile-sam >= 0.1.0

### 3. 准备模型权重
将 YOLO 模型权重文件（`.pt` / `.engine` / `.onnx`）放入 `weight(.pt)/` 目录。

推荐模型:
- YOLOv8n-seg.pt (轻量级)
- YOLOv8s-seg.pt (平衡型)
- YOLOv8m-seg.pt (高精度)

### 4. 运行程序
```bash
# 方法 1: 直接运行
python main.py

# 方法 2: Windows 启动脚本
run.bat
```

---

## 📖 使用说明

### 基本流程

1. **选择检测模式**
   - 实时监测: 使用摄像头进行实时检测
   - 视频检测: 加载本地视频文件进行分析

2. **配置模型参数**
   - 模型类型: `yoloseg` (快速) 或 `yolo+msam` (精确)
   - 推理设备: `CPU` 或 `CUDA`
   - 模型权重: 选择对应的 `.pt` / `.engine` 文件

3. **设置过滤条件**
   - 类别过滤: 选择需要检测的目标类别
   - 置信度阈值: 调整检测灵敏度

4. **开始检测**
   - 点击"开始"按钮启动检测
   - 使用控制面板进行播放控制
   - 查看实时 FPS 和日志信息

### 视频控制操作

| 功能 | 操作 |
|------|------|
| 播放/暂停 | 点击播放按钮 |
| 进度跳转 | 拖动进度条 |
| 快进/快退 | 点击前进/后退按钮 |
| 停止检测 | 点击停止按钮 |

---

## 项目结构

```
video-segmentation-system/
├── main.py                    # 程序入口
├── config.py                  # 全局配置文件
├── requirements.txt           # Python 依赖
├── run.bat                    # Windows 启动脚本
├── README.md                  # 项目说明
├── PROJECT_STRUCTURE.md       # 详细结构文档
├── PROJECT_WORKFLOWS.md       # 工作流详解
├── weight(.pt)/               # 模型权重目录
└── src/                       # 源代码目录
    ├── __init__.py
    ├── core/                  # 核心模块
    │   ├── __init__.py
    │   ├── video_thread.py    # 视频处理线程 (560行)
    │   └── models.py          # 模型管理 (60行)
    ├── ui/                    # 用户界面模块
    │   ├── __init__.py
    │   ├── main_window.py     # 主窗口 (297行)
    │   ├── navigation.py      # 导航面板 (120行)
    │   └── controls.py        # 控制面板 (175行)
    └── utils/                 # 工具模块
        ├── __init__.py
        └── helpers.py         # 辅助函数 (50行)
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `core/video_thread.py` | 视频处理、模型推理、帧处理 |
| `core/models.py` | 模型加载、设备管理 |
| `ui/main_window.py` | 主窗口整合 |
| `ui/navigation.py` | 功能选择、模型设置 |
| `ui/controls.py` | 播放控制、进度条、状态显示 |
| `utils/helpers.py` | 时间格式化、文件操作等工具函数 |

---



## ⚡ 性能优化

### 已实现的优化

1. **图像尺寸优化**
   - YOLO 输入: 320x320 (可配置)
   - MobileSAM 降采样: 0.5 倍

2. **追踪算法优化**
   - IoU 阈值: 0.1 (快速匹配)
   - EMA 系数: 0.5 (平滑与响应平衡)
   - 最大追踪年龄: 10 帧

3. **隔帧推理**
   - MSAM 调用间隔: 1 帧 (可配置)
   - 中间帧复用上一次结果

4. **多线程架构**
   - 视频处理线程独立运行
   - UI 线程不阻塞

### 性能参考

| 模式 | 设备 | 分辨率 | FPS |
|------|------|--------|-----|
| YOLO-Seg | CPU | 640x480 | 15-20 |
| YOLO-Seg | CUDA | 640x480 | 40-60 |
| YOLO+MSAM | CPU | 640x480 | 5-8 |
| YOLO+MSAM | CUDA | 640x480 | 15-25 |

---

## 常见问题

### Q1: 如何提升检测速度？
- 使用 CUDA GPU 加速
- 选择 YOLO-Seg 模式
- 降低输入图像分辨率
- 使用 TensorRT 引擎格式

### Q2: 如何提升检测精度？
- 选择 YOLO+MSAM 模式
- 使用更大的 YOLO 模型 (如 YOLOv8m)
- 调整置信度阈值
- 增加 MSAM 调用频率

### Q3: 支持哪些视频格式？
支持常见格式: MP4, AVI, MOV, MKV, WMV, FLV

### Q4: 如何添加自定义类别？
修改 `config.py` 中的 `SUPPORTED_CLASSES` 列表

### Q5: 程序崩溃怎么办？
- 检查 CUDA 版本与 PyTorch 兼容性
- 确认模型权重文件完整
- 查看日志面板的错误信息

---

## 贡献指南

欢迎贡献代码、报告问题或提出建议！







---

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO 实现
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) - 轻量级 SAM 模型
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架

---

<div align="center">

**如果这个项目对你有帮助，请给个 Star**

</div>
