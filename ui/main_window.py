import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_user = None
        self.initUI()
        
    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('Imaged Speech Classification')
        self.setGeometry(100, 100, 800, 600) # 左上x,y,宽度，高度
        
        # 创建主widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建布局
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # 添加状态栏显示用户信息
        self.statusBar().showMessage('Current User: Group us! | Mode: Main')
        
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
        plot_label = QLabel('EEG Plot Area')
        plot_label.setStyleSheet('background-color: white; border: 1px solid black;')
        plot_label.setMinimumHeight(400)  # 设置最小高度
        plot_label.setAlignment(Qt.AlignCenter)
        self.centralWidget().layout().addWidget(plot_label)
        
    def create_control_panel(self):
        # 创建控制按钮面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        # 创建按钮
        register_btn = QPushButton('Register/Login')
        register_btn.clicked.connect(self.show_register_dialog)

        train_btn = QPushButton('Train')
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
            self.current_user = dialog.get_current_user()   # 当前user
            self.statusBar().showMessage(f'Current User: {self.current_user} | Mode: Logged In')
            # 可以在这里启用其他按钮
            self.enable_user_functions()

    def enable_user_functions(self):
        # 登录后的功能启用逻辑
        pass