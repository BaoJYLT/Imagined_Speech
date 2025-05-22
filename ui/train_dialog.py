from PyQt5.QtWidgets import *
import os
import re
import numpy as np

class TrainDialog(QDialog):
    def __init__(self, user_name, parent=None):
        super().__init__(parent)
        self.user_name = user_name
        self.user_id = None
        self.file_path = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Train Model')
        layout = QVBoxLayout()
        
        # 文件选择按钮
        self.select_btn = QPushButton('Select Training Data', self)
        self.select_btn.clicked.connect(self.select_file)
        
        # 显示选中文件的标签
        self.file_label = QLabel('No file selected')
        
        layout.addWidget(self.select_btn)
        layout.addWidget(self.file_label)
        self.setLayout(layout)
        
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Data",
            "",     # 初始路径为空
            "NPY Files (*.npy)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            
            # 从文件名提取ID
            match = re.search(r'Sample(\d{2})_data', os.path.basename(file_path))
            if match:
                self.user_id = match.group(1)
                self.accept()
            else:
                QMessageBox.warning(self, 'Error', 
                                  'Invalid file format')