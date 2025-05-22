import sys
import numpy as np
import os
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from torch.utils.data import DataLoader, TensorDataset

from utils.plot_utils import EEGCanvas
# from eeg_wavelet_lstm_pipeline import *
from eeg_wavelet_lstm_pipeline import preprocess_file, extract_wavelet_features, train_model, init_weights
from eeg_wavelet_lstm_pipeline import EEGWaveletLSTM

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_user = None
        self.eeg_canvas = None
        self.user_name = None
        self.user_id = None
        self.initUI()
        
    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('Imagined Speech Classification')
        self.setGeometry(100, 100, 1600, 1200) # 左上x,y,宽度，高度
        
        # 创建主widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)  # 设置边距
        main_widget.setLayout(layout)
        
        # 添加状态栏显示用户信息
        self.statusBar().showMessage('Current User: Group us!   |        Mode: Main')
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建主要部件
        self.create_plot_area()
        self.create_control_panel()
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        help_menu = menubar.addMenu('Help')
        
    def create_plot_area(self):
        # 使用matplotlib创建绘图区域
        '''
        plot_label = QLabel('EEG Plot Area')
        plot_label.setStyleSheet('background-color: white; border: 1px solid black;')
        plot_label.setMinimumHeight(400)  # 设置最小高度
        plot_label.setAlignment(Qt.AlignCenter)
        self.centralWidget().layout().addWidget(plot_label)
        '''
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        self.eeg_canvas = EEGCanvas(self)# 创建plt画布
        plot_layout.addWidget(self.eeg_canvas)

        # 设置固定高度为主窗口高度的0.75
        height = int(self.height() * 0.75)  # 获取主窗口高度的60%
        plot_widget.setFixedHeight(height)
        # 添加到主布局
        main_layout = self.centralWidget().layout()
        main_layout.addWidget(plot_widget)
        # 确保窗口大小变化时EEG显示区域也相应调整
        self.resizeEvent = lambda event: plot_widget.setFixedHeight(int(self.height() * 0.75))
        '''
        # 设置绘图区域的大小策略
        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 将绘图区域添加到主布局
        main_layout = self.centralWidget().layout()
        main_layout.addWidget(plot_widget, stretch=50)  # 设置stretch为75表示占用75%的空间
        # self.centralWidget().layout().addWidget(self.eeg_canvas)'
        '''

    def update_eeg_display(self, user_id):
        """更新EEG显示区域"""
        try:
            # 加载用户的训练数据
            data_path = f'Training_set_npy/Data_Sample{user_id}_data.npy'
            eeg_data = np.load(data_path)
            
            # 显示第一个trial的数据!
            first_trial = eeg_data[0]  # shape: (channels, samples)
            self.eeg_canvas.plot_eeg(first_trial, user_id=user_id)
            
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'加载EEG数据失败: {str(e)}')

    def create_control_panel(self):
        # 创建控制按钮面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        # 创建按钮
        register_btn = QPushButton('Register/Login')
        register_btn.clicked.connect(self.show_register_dialog)

        train_btn = QPushButton('Train')
        train_btn.clicked.connect(self.show_train_dialog)
        
        model_list_btn = QPushButton('Model List')
        test_btn = QPushButton('Test')
        performance_btn = QPushButton('Performance') # model performance
        
        # 将按钮添加到布局
        for btn in [register_btn, train_btn, model_list_btn, test_btn, performance_btn]:
            control_layout.addWidget(btn)
            
        control_panel.setLayout(control_layout)
        self.centralWidget().layout().addWidget(control_panel)

    # main window 注册流程控制
    def show_register_dialog(self):
        from .register_dialog import RegisterDialog
        dialog = RegisterDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.user_name = dialog.get_user_name()
            self.statusBar().showMessage(f'Current User: {self.user_name}')
            # 根据功能修改
            # self.current_user = dialog.get_current_user()   # 当前user
            # self.statusBar().showMessage(f'Current User:  {self.current_user}       |       Mode: Logged In')
            # 可以在这里启用其他按钮
            self.enable_user_functions()

    def enable_user_functions(self):
        # 登录后的功能启用逻辑
        pass

    # Train设置窗口
    def show_train_dialog(self):
        from .train_dialog import TrainDialog
        if not self.user_name:
            QMessageBox.warning(self, 'Error', 'Please register first')
            return
            
        dialog = TrainDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                # first trail EEG显示
                self.user_id = dialog.user_id
                training_data = dialog.files
                train_data = np.load(training_data['train_data'])
                # train_data = np.load(dialog.training_data['train_data'])
                first_trail = train_data[0]
                # 状态栏更新
                self.statusBar().showMessage(
                    f'Current User: {self.user_name} | ID: {dialog.user_id} | Status: Training...'
                )
                self.eeg_canvas.plot_eeg(first_trail, user_id=self.user_id)
                # self.train_model(dialog.file_path)
                # 训练
                self.start_training(training_data, self.user_id)
                # self.start_training(dialog.training_data)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load data: {str(e)}')

    def start_training(self, training_data, user_id):
        try:
            # 创建进度对话框
            progress = QProgressDialog("Training...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 预处理数据
            os.makedirs("temp_process", exist_ok=True)  # 临时存储预处理后的四个文件
            train_processed, train_labels = preprocess_file(
                training_data['train_data'],
                training_data['train_label'],
                f"temp_process/temp_train_data.npy",
                f"temp_process/temp_train_labels.npy"  
            )  
            
            val_processed, val_labels = preprocess_file(
                training_data['val_data'],
                training_data['val_label'],
                f"temp_process/temp_val_data.npy",
                f"temp_process/temp_val_labels.npy"
            )
            
            # 提取特征
            train_features = extract_wavelet_features(train_processed)
            val_features = extract_wavelet_features(val_processed)
            # 数据加载器，特征及labels，转换为PyTorch张量
            train_X = torch.tensor(train_features, dtype=torch.float32)
            train_y = torch.tensor(train_labels - 1, dtype=torch.long)
            val_X = torch.tensor(val_features, dtype=torch.float32)
            val_y = torch.tensor(val_labels - 1, dtype=torch.long)

            train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle = True)
            val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=64)
            # 初始化与训练模型
            input_dim = train_X.shape[2]
            model = EEGWaveletLSTM(input_dim=input_dim)
            # model = train_model(train_X, train_y, val_X, val_y)
            model.apply(init_weights)   # 使用apply初始化模型参数
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            trained_model = train_model(model, train_loader, val_loader, device)

            # 训练完成，保存模型
            self.save_trained_model(trained_model, user_id)
            # self.save_trained_model(trained_model, training_data['user_id'])
            
            # 更新状态
            self.statusBar().showMessage(
                # f'Current User: {self.user_name} | ID: {training_data["user_id"]} | Status: Training Complete')
                f'Current User: {self.user_name} | ID: {user_id} | Status: Training Complete')
            QMessageBox.information(self, 'Success', 'Model training completed!')

            # 清理临时文件夹
            import shutil
            shutil.rmtree("temp_process", ignore_errors=True)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Training failed: {str(e)}')
            raise
    
    '''
    # Train模型调用
    def train_model(self, file_path):
        # 模型训练代码

        from .savemodel_dialog import SaveModelDialog
        save_dialog = SaveModelDialog(self.user_name, self.user_id, self)
        if save_dialog.exec_() == QDialog.Accepted:
            # 保存模型
            model_path = os.path.join(save_dialog.save_path, 
                                    f'{save_dialog.model_name}.pth')
            # ...save model code...
            QMessageBox.information(self, 'Success', 
                                  f'Model saved to:\n{model_path}')
    '''  

    # 保存训练好的模型      
    def save_trained_model(self, model, user_id):
        try:
            save_path = f'models/user_{user_id}_model.pth'
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), save_path)
        except Exception as e:
            QMessageBox.warning(self, '警告', f'模型保存失败: {str(e)}')

    def show_test_dialog(self):
        if not self.user_name:
            QMessageBox.warning(self, 'Error', 'Please register first')
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Data",
            "",
            "NPY Files (*.npy)"
        )
        
        if file_path:
            self.test_model(file_path)