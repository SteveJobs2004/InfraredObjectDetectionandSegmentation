# 视频分割检测系统

基于 YOLO 和 MobileSAM 的视频分割检测系统，支持实时监测和视频文件检测。

## 功能特性

- 支持 YOLO-Seg 和 YOLO+MobileSAM 两种模型
- 实时摄像头监测
- 视频文件检测（支持进度条和跳转）
- 类别过滤（person, car, plane, bike）
- 设备选择（CPU/CUDA）
- TensorRT 引擎支持

## 项目结构

```
project/
├── src/                    # 源代码目录
│   ├── core/              # 核心模块
│   │   ├── __init__.py
│   │   ├── video_thread.py # 视频处理线程
│   │   └── models.py      # 模型管理
│   ├── ui/                # 用户界面模块
│   │   ├── __init__.py
│   │   ├── main_window.py # 主窗口
│   │   ├── navigation.py  # 导航面板
│   │   └── controls.py    # 控制面板
│   └── utils/             # 工具模块
│       ├── __init__.py
│       └── helpers.py     # 辅助函数
├── main.py                # 程序入口
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行程序

```bash
python main.py
```

## 使用说明

1. 选择检测模式（实时监测/视频检测）
2. 选择模型类型（yoloseg/yolo+msam）
3. 选择推理设备（CPU/CUDA）
4. 选择模型权重文件
5. 设置类别过滤
6. 点击开始按钮开始检测
