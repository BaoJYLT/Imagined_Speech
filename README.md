# Imagined_Speech
ZJU-IIP course project, based on https://osf.io/pq7vb/ Track #3, classify 5 EEG signal.

## 当前开发结构
```
project/
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # 主窗口
│   ├── register_dialog.py      # 注册对话框
│   ├── train_dialog.py         # 训练界面
|   ├── test_dialog.py          # 测试界面
|   └── performance_dialog.py   # best model性能界面
│
├── modelList/                  # 存储训练好的模型
│   └── .pth files      
│
├── utils/
│   └── plot_utils.py           # EEG可视化绘图工具
│
├── ui_main.py                  # UI入口文件
└── EWLP.py                     # 现有的模型代码
├── EEG_DATASET/                # 数据集文件
│   ├── Test_Sample01_preprocess
|   ├── Training_Sample01_preprocess
|   └── Validation_Sample01_preprocess
```

## 数据集命名规范
### 训练集
训练集文件首先经过了数据预处理与数据增强操作，对于程序会直接调用增强后的数据集，对于data文件和labels文件分别是：\
`"Data_Sample{user_id}_data_aug_0_1.npy"`和`"Data_Sample{user_id}_labels_aug_0_1.npy"`。
比如对于ID01的用户其命名示例为:
```
"EEG_DATASET\Training_Sample01_preprocess\Data_Sample01_data_aug_0_1.npy"
"EEG_DATASET\Training_Sample01_preprocess\Data_Sample01_labels_aug_0_1.npy"
```
### 验证集
验证集需要进行预处理，但是不需要经过数据增强的过程，所以其data文件和labels文件命名规范分别为：\
`"Data_Sample{user_id}_data_pre_0_1.npy"`和`"Data_Sample{user_id}_labels_pre_0_1.npy"`。
比如对于ID01的用户其命名示例为:
```
"EEG_DATASET\Validation_Sample01_preprocess\Data_Sample01_data_pre_0_1.npy"
"EEG_DATASET\Validation_Sample01_preprocess\Data_Sample01_labels_pre_0_1.npy"
```
### 测试集
测试集也需要进行预处理，但是不需要经过数据增强的过程，所以其data文件命名规范为：
`"Test_Sample{user_id}_preprocess\Data_Sample{user_id}_data_pre.npy"`。
比如对于ID01的用户其命名示例为:
```
"EEG_DATASET\Test_Sample01_preprocess\Data_Sample01_data_pre.npy"
```

## 运行方式
### 命令行运行方式
在project目录下，运行python，对应的指令为`python ui_mian.py`即可运行程序，开始EEG imagined speech模型训练、评估和测试。
### 可执行文件Imagined Speech.exe运行方式
推荐将EEG_DATASET和.exe文件放在同一个目录下，直接使用exe文件运行。

## requirements
```
pyqt5==5.15.11
numpy==2.2.5
sklearn==1.6.1
re==2024.11.6
torch==2.5.1
matplotlib==3.10.0
seaborn==0.13.2
pywt==1.8.0
scipy==1.15.3
``` 
## UI开发顺序
1. 创建基本窗口框架
2. 实现用户注册/登录功能
3. 添加EEG数据可视化
4. 实现模型训练功能
5. 实现模型测试功能
6. 添加性能展示功能
