import numpy as np
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from sklearn.metrics import *
from utils.plot_utils import EEGCanvas

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
        

    def create_control_panel(self):
        # 创建控制按钮面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        # 创建按钮
        register_btn = QPushButton('Register/Login')
        register_btn.clicked.connect(self.show_register_dialog)

        train_btn = QPushButton('Train')
        train_btn.clicked.connect(self.show_train_dialog)
        
        # model_list_btn = QPushButton('Model List')
        test_btn = QPushButton('Test')
        test_btn.clicked.connect(self.show_test_dialog)

        performance_btn = QPushButton('Performance') # model performance
        performance_btn.clicked.connect(self.show_performance_dialog)
        
        # 将按钮添加到布局
        for btn in [register_btn, train_btn, test_btn, performance_btn]:
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

                self.statusBar().showMessage(
                    f'Current User: {self.user_name} | ID: {dialog.user_id} | Status: Training...'
                )
                # 显示验证集对应图像
                val_data_path = f'EEG_DATASET/Validation_Sample{self.user_id}_preprocess/Data_Sample{self.user_id}_data_pre_0_1.npy'
                if os.path.exists(val_data_path):
                    self.update_eeg_display(self.user_id)
                else:
                    QMessageBox.warning(self, 'Warning', f'Validation data not found: {val_data_path}')
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load data: {str(e)}')

    
    def show_test_dialog(self):
        from .test_dialog import TestDialog
        if not self.user_name:
            QMessageBox.warning(self, 'Error', 'Please register first')
            return
        if not self.user_id:
            QMessageBox.warning(self, 'Error', 'Please select a user ID first')
            return
        dialog = TestDialog(self)
        dialog.exec_()
        
    
    def show_performance_dialog(self):
        from .performance_dialog import PerformanceDialog
        if not self.user_name:
            QMessageBox.warning(self, 'Error', 'Please register first')
            return 
        dialog = PerformanceDialog(self)
        dialog.exec_()