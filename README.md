# Imaged_Speech
ZJU-IIP course project, based on https://osf.io/pq7vb/ Track #3, classify 5 EEG signal.

## 结构建议
project/
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py      # 主窗口
│   ├── register_dialog.py  # 注册对话框
│   ├── train_widget.py     # 训练界面
│   └── test_widget.py      # 测试界面
│
├── models/                 # 存储训练好的模型
│   └── user_models/       
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py     # 数据加载
│   ├── model_manager.py   # 模型管理
│   └── plot_utils.py      # 绘图工具
│
├── config.py              # 配置文件
├── main.py               # 程序入口
├── ui_main.py              # UI主入口文件
└── eeg_wavelet_lstm_pipeline.py  # 现有的模型代码

## 当前开发结构
project/
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py      # 主窗口
│   ├── register_dialog.py  # 注册对话框
<!-- │   ├── train_widget.py     # 训练界面
│   └── test_widget.py      # 测试界面
│
├── models/                 # 存储训练好的模型
│   └── user_models/       
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py     # 数据加载
│   ├── model_manager.py   # 模型管理
│   └── plot_utils.py      # 绘图工具 -->
│
<!-- ├── config.py              # 配置文件
├── main.py               # 程序入口 -->
├── ui_main.py              # UI主入口文件
└── eeg_wavelet_lstm_pipeline.py  # 现有的模型代码

## UI开发顺序
1. 创建基本窗口框架
2. 实现用户注册/登录功能
3. 添加EEG数据可视化
4. 实现模型训练功能
5. 实现模型测试功能
6. 添加性能展示功能
7. 优化用户体验
## 关键注意点
信号与槽机制：使用PyQt5的信号槽机制处理用户交互
线程处理：将耗时操作（如模型训练）放在单独线程中
数据管理：建立清晰的数据流转机制
错误处理：添加完善的错误处理和用户提示
配置文件：使用配置文件管理路径等参数
