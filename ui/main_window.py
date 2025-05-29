import sys
import numpy as np
import os
import torch
import re
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from torch.utils.data import DataLoader, TensorDataset

from utils.plot_utils import EEGCanvas
# # from eeg_wavelet_lstm_pipeline import *
# from eeg_wavelet_lstm_pipeline import preprocess_file, extract_wavelet_features, train_model, init_weights
# from eeg_wavelet_lstm_pipeline import EEGWaveletLSTM

from EWLP import preprocess_file, extract_wavelet_features, train_model, init_weights, set_seed
from EWLP import EEGWaveletLSTM

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

        # 字体等外观设计
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QWidget {
                font-family: 'Segoe UI';
                font-family: 'Microsoft YaHei';
                font-size: 20px;
            }
            QPushButton {
                # font-family: 'Consolas';
                font-family: 'Microsoft YaHei';
                font-size: 20px;
                padding: 5px 15px;
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QLabel {
                # font-family: 'Segoe UI';
                font-family: 'Microsoft YaHei';
                font-size: 20px;
            }
            QStatusBar {
                # font-family: 'Consolas';
                font-family: 'Microsoft YaHei';
                font-size: 20px;
            }
            QMenuBar {
                # font-family: 'Segoe UI';
                font-family: 'Microsoft YaHei';
                font-size: 20px;
            }
        """)
        
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
            val_data_path = f'EEG_DATASET/Validation_Sample{user_id}_preprocess/Data_Sample{user_id}_data_pre_0_1.npy'
            if not os.path.exists(val_data_path):
                raise FileNotFoundError(f"Validation data not found: {val_data_path}")
                
            val_data = np.load(val_data_path)
            first_trial = val_data[0]  # 获取第一个trial的数据  load进来的数据格式应为(trails, channels, samples)
            
            self.eeg_canvas.plot_eeg(first_trial, user_id=user_id)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to update EEG display: {str(e)}')
        
        # try:
        #     # 加载用户的训练数据
        #     # data_path = f'Training_set_npy/Data_Sample{user_id}_data.npy'
        #     data_path = f'EEG_DATASET/Validation_Sample{user_id}_preprocess/Data_Sample{user_id}_data_pre_0_1.npy'
        #     eeg_data = np.load(data_path)
            
        #     # 显示第一个trial的数据!
        #     first_trial = eeg_data[0]  # shape: (channels, samples)
        #     self.eeg_canvas.plot_eeg(first_trial, user_id=user_id)
            
        # except Exception as e:
        #     QMessageBox.warning(self, 'Error', f'Errors occurr while loading EEG data: {str(e)}')

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

                # train_data = np.load(training_data['train_data'])
                # # train_data = np.load(dialog.training_data['train_data'])
                # first_trail = train_data[0]

                # 状态栏更新
                self.statusBar().showMessage(
                    f'Current User: {self.user_name} | ID: {dialog.user_id} | Status: Training...'
                )
                # self.eeg_canvas.plot_eeg(first_trail, user_id=self.user_id)
                # self.train_model(dialog.file_path)

                self.update_eeg_display(self.user_id) # 显示validation set 图像

                # 训练
                self.start_training(
                    training_data,
                    model_name = dialog.model_name,
                    save_path = dialog.save_path
                )
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load data: {str(e)}')

    def start_training(self, training_data, model_name, save_path):
        try:
            # 创建进度对话框
            progress = QProgressDialog("Training...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Training Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(False)  # 不自动关闭
            progress.setAutoReset(False)  # 不自动重置
            progress.setMinimumDuration(0)
            progress.setStyleSheet("""
                QProgressDialog {
                    font-family: 'Microsoft YaHei UI';
                    font-size: 20px;
                    min-width: 500px;
                }
                QProgressBar {
                    text-align: center;
                    font-size: 16px;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    background: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
            progress.show()
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 数据导入
            train_data = np.load(training_data['train_data'])
            train_labels = np.load(training_data['train_label'])
            val_data = np.load(training_data['val_data'])
            val_labels = np.load(training_data['val_label'])
            # 特征提取
            progress.setLabelText("Extracting wavelet features...")
            train_features = extract_wavelet_features(train_data)
            val_features = extract_wavelet_features(val_data)

            # 训练模型
            TOTAL_SEEDS = 351
            # accuracy_train_list = []
            best_seed = None
            best_acc = 0
            best_model = None

            # 定义进度更新回调
            def update_training_progress(epoch_progress, message, current_seed):
                # 计算总体进度百分比 (0-100)
                total_progress = int((current_seed * 100 + epoch_progress) / TOTAL_SEEDS)
                progress_message = (
                    f"Seed {current_seed}/{TOTAL_SEEDS-1}\n"
                    f"{message}"
                )
                
                if progress.wasCanceled():
                    raise Exception("Training canceled by user")
                progress.setLabelText(progress_message)
                progress.setValue(total_progress)
                QApplication.processEvents()

            for seed in range(TOTAL_SEEDS):
                set_seed(seed)
                progress.setLabelText(f"Training with seed {seed}...")
                model = EEGWaveletLSTM(input_dim=train_features.shape[2])
                model.apply(init_weights)
                train_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(train_features, dtype=torch.float32),
                        torch.tensor(train_labels, dtype=torch.long)
                        ),
                    batch_size=64,
                    shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(val_features, dtype=torch.float32), 
                        torch.tensor(val_labels, dtype=torch.long)),
                    batch_size=64
                )
                trained_model, final_acc = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=1000,
                    patience=50,
                    progress_callback=lambda p, m: update_training_progress(p, m, seed)
                )   # lambda表达式，含有两个变量的匿名函数progress message
                # accuracy_train_list.append((seed, final_acc))
                if final_acc > best_acc:
                    best_acc = final_acc
                    best_seed = seed
                    best_model = trained_model

            if best_model is not None:
                best_model_save_path = save_path
                torch.save(best_model.state_dict(),best_model_save_path)

            # 保存model
            model_path = os.ath.join(save_path, f'{model_name}.pth')
            os.makedirs(save_path, exist_ok=True)
            torch.save(best_model.state_dict(), model_path)
            print(f"best model saved:{best_model_save_path}")


            progress.setLabelText("Training completed!")
            progress.setValue(100)
            progress.setCancelButtonText("OK") # 等待用户确认
            while not progress.wasCanceled():
                QApplication.processEvents()
            progress.close()

            # 保存成功
            QMessageBox.information(self, 'Success', 
                            f'Model training completed!\n\n'
                            f'Model Name: {model_name}\n'
                            f'Best Seed: {best_seed}\n'
                            f'Best Accuracy: {best_acc:.4f}\n'
                            f'Saved to: {model_path}')
            # 更新状态
            self.statusBar().showMessage(
                f'Current User: {self.user_name} | ID: {self.user_id} | Status: Model Saved')
            # QMessageBox.information(self, 'Success', 
            #                         f'Model training completed!\n Saved to: {model_path}')

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

    
    '''
    # 保存训练好的模型      
    def save_trained_model(self, model, user_id):
        try:
            save_path = f'models/user_{user_id}_model.pth'
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
        except Exception as e:
            QMessageBox.warning(self, '警告', f'模型保存失败: {str(e)}')
    '''
    

    def show_test_dialog(self):
        if not self.user_name:
            QMessageBox.warning(self, 'Error', 'Please register first')
            return
            
        if not self.user_id:
            QMessageBox.warning(self, 'Error', 'Please select a user ID first')
            return
        
        try:
            # 1. 选择模型文件
            model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Model File",
                "./models",  # 默认打开模型保存目录
                "PyTorch Models (*.pth)"
            )
            
            if not model_path:
                return
                
            # 2. 选择测试数据文件
            test_data_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Test Data File",
                "./Test_set_npy",  # 默认打开测试集目录
                "NPY Files (*.npy)"
            )
            
            if not test_data_path:
                return
            
            # 3. 验证测试文件的用户ID
            basename = os.path.basename(test_data_path)
            match = re.match(r'Data_Sample(\d{2})_data\.npy', basename)
            
            if not match:
                QMessageBox.warning(self, 'Error', 'Invalid test file format')
                return
                
            file_user_id = match.group(1)
            if file_user_id != self.user_id:
                QMessageBox.warning(
                    self, 
                    'Error', 
                    f'Test file user ID ({file_user_id}) does not match current user ID ({self.user_id})'
                )
                return
                
            # 4. 更新状态栏
            self.statusBar().showMessage(
                f'Current User: {self.user_name} | ID: {self.user_id} | Status: Testing...'
            )
            
            # 5. 开始测试
            self.run_test(model_path, test_data_path)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Test failed: {str(e)}')