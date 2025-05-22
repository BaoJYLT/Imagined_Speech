from PyQt5.QtWidgets import *
import os

class SaveModelDialog(QDialog):
    def __init__(self, user_name, user_id, parent=None):
        super().__init__(parent)    # 调用父类QDialog初始化方法
        self.user_name = user_name
        self.user_id = user_id
        self.model_name = None
        self.save_path = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Save Model')
        layout = QVBoxLayout()
        
        # 模型名称输入
        name_label = QLabel('Model Name:')
        self.name_input = QLineEdit()
        default_name = f'{self.user_name}_ID{self.user_id}_model'
        self.name_input.setText(default_name)   # 设置输入框的默认文本为
        
        # 保存路径选择
        path_label = QLabel('Save Location:')
        self.path_btn = QPushButton('Select Folder')
        self.path_btn.clicked.connect(self.select_path)
        self.path_label = QLabel('Default: ./models/')  # 创建标签显示默认保存路径
        
        # 确认按钮
        self.confirm_btn = QPushButton('Save')
        self.confirm_btn.clicked.connect(self.save_model)
        
        layout.addWidget(name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(path_label)
        layout.addWidget(self.path_btn)
        layout.addWidget(self.path_label)
        layout.addWidget(self.confirm_btn)
        
        self.setLayout(layout)
        
    def select_path(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Location",
            "./models/"     # 保存位置为当前这个models当前运行程序的文件夹下的models文件夹。 需要使用IIPLAB去运行ui_main.py
        )
        if folder:
            self.save_path = folder
            self.path_label.setText(folder)
            
    def save_model(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, 'Error', 'Please enter a model name')
            return
            
        self.model_name = name
        if not self.save_path:
            self.save_path = './models/'
            
        self.accept()