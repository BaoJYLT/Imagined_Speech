from PyQt5.QtWidgets import *
import os
import re
import numpy as np

class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.files = {
            'train_data': None,
            'train_label': None,
            'val_data': None,
            'val_label': None
        }
        self.user_id = None # 这个是子类的id，如果调用父类QDialog中可能定义的属性，则可以super().user_id 前提是有这个属性
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Train Model')
        layout = QVBoxLayout()
        
        # 文件选择组
        file_group = QGroupBox("Select files")
        file_layout = QGridLayout()
        
        # 创建文件选择按钮和标签
        self.labels = {}
        row = 0
        for file_type in self.files.keys():
            btn = QPushButton(f'Select {file_type} file')
            btn.clicked.connect(lambda checked, t=file_type: self.select_file(t))
            label = QLabel('no file selected')
            self.labels[file_type] = label
            file_layout.addWidget(btn, row, 0)
            file_layout.addWidget(label, row, 1)
            row += 1
            
        file_group.setLayout(file_layout)
        
        # 确认按钮
        self.confirm_btn = QPushButton('Start Train')
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.validate_files)
        
        layout.addWidget(file_group)
        layout.addWidget(self.confirm_btn)
        self.setLayout(layout)
        
    def select_file(self, file_type):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {file_type} file", "", "NPY Files (*.npy)")
            
        if file_path:
            # 检查文件命名
            basename = os.path.basename(file_path)
            match = re.match(r'Data_Sample(\d{2})_(data|labels)\.npy', basename)
            
            if not match:
                QMessageBox.warning(self, 'Error', 'Incorrect file name format!')       # 文件命名格式错误
                return
                
            sample_id = match.group(1)
            
            # 如果已经有其他文件，检查sample_id是否匹配
            if self.user_id and sample_id != self.user_id:
                QMessageBox.warning(self, 'Warning', 'Please select files from ONE user!')    # 选择同一个Sample的文件
                return
                
            self.user_id = sample_id
            self.files[file_type] = file_path
            self.labels[file_type].setText(basename)
            
            # 检查是否所有文件都已选择
            if all(self.files.values()):
                self.confirm_btn.setEnabled(True)
                
    def validate_files(self):
        """验证所有文件并准备训练"""
        '''
        try:
            # 加载并验证数据
            train_data = np.load(self.files['train_data'])
            train_label = np.load(self.files['train_label'])
            val_data = np.load(self.files['val_data'])
            val_label = np.load(self.files['val_label'])
            
            # 检查数据维度匹配
            if len(train_data) != len(train_label) or len(val_data) != len(val_label):
                raise ValueError("Sample size mismatch between datas and labels!")  # 数据和标签的样本数不匹配
                
            # 设置返回数据
            self.training_data = {
                'train_data': train_data,
                'train_label': train_label,
                'val_data': val_data,
                'val_label': val_label,
                'user_id': self.user_id
            }
        '''
        if not all(self.files.values()):
            QMessageBox.warning(self, 'Error', 'Please select all required files!')
            return
                    
        try:
            # 设置返回数据为文件路径
            self.training_data = {
                'train_data': self.files['train_data'],    # 保存文件路径
                'train_label': self.files['train_label'],
                'val_data': self.files['val_data'],
                'val_label': self.files['val_label'],
                'user_id': self.user_id
            }
            
            # 验证文件是否存在
            for file_type, file_path in self.files.items():
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{file_type} file not found: {file_path}")
                
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Data validation failure: {str(e)}')   # 数据验证失败