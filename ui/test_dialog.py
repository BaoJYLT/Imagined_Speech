import os
import re
import numpy as np
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from EWLP import extract_wavelet_features, EEGWaveletLSTM

class TestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.user_name = parent.user_name
        self.user_id = parent.user_id
        self.initUI()
    # 界面初始化-button响应、模型与文件打开-尝试运行
    def initUI(self):
        self.setWindowTitle('Test Model')
        self.setMinimumWidth(800)  # 设置最小宽度
        self.resize(600, 200)     # 设置默认大小
        layout = QVBoxLayout()
        
        model_btn = QPushButton('Select Model File')
        model_btn.clicked.connect(self.select_model_file)
        layout.addWidget(model_btn)
        
        test_btn = QPushButton('Select Test Data')
        test_btn.clicked.connect(self.select_test_file)
        layout.addWidget(test_btn)
        self.parent.statusBar().showMessage(
                f'Current User: {self.parent.user_name} | ID: {self.user_id} | Status: Test Mode'
            )
        self.setLayout(layout)

    def select_model_file(self):
        self.model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "./models",
            "PyTorch Models (*.pth)"
        )
        
        if self.model_path:
            self.check_and_run_test()

    def select_test_file(self):
        self.test_data_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Data File",
            f"EEG_DATASET/Test_Sample{self.user_id}_preprocess",
            "NPY Files (*.npy)"
        )
        
        if self.test_data_path:
            self.check_and_run_test()

    def check_and_run_test(self):
        if not hasattr(self, 'model_path') or not hasattr(self, 'test_data_path'):
            return
            
        try:
            # ID验证
            basename = os.path.basename(self.test_data_path)
            match = re.match(r'Data_Sample(\d{2})_data_pre\.npy', basename)
            if not match:
                QMessageBox.warning(self, 'Error', 'Invalid test file format')
                return

            file_user_id = match.group(1)
            if file_user_id != self.user_id:
                QMessageBox.warning(
                    self,
                    'Warning',
                    f'Test file user ID ({file_user_id}) does not match current user ID ({self.user_id})'
                )
            
            # 运行测试
            self.run_test()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Test failed: {str(e)}')

    def run_test(self): # self中已经有了name id model_path test_data_path
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            test_data = np.load(self.test_data_path)
            test_features = extract_wavelet_features(test_data)
            test_X = torch.tensor(test_features, dtype=torch.float32).to(device)
            
            model = EEGWaveletLSTM(input_dim=test_features.shape[2])
            model_info = torch.load(self.model_path, map_location=device)
            model_state = model_info['model_state']
            model.load_state_dict(model_state)
            model.to(device)
            model.eval()
            
            # PREDICTION
            with torch.no_grad():
                outputs = model(test_X)
                predictions = outputs.argmax(dim=1).cpu().numpy()
            # 交互 完成test
            QMessageBox.information(self, 'Success', 'Prediction finished!')
            
            # 显示预测结果
            self.show_predictions(predictions)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Test analysis failed: {str(e)}')

    def show_predictions(self, predictions):
        label_to_word = {0: "Hello", 1: "Stop", 2: "Yes"} # 标签单词映射
        predicted_words = [label_to_word[pred] for pred in predictions]
        
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle('Prediction Results')
        result_dialog.setMinimumWidth(600)
        
        layout = QVBoxLayout()
        
        # 预测结果文本显示
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        results_text.setStyleSheet("font-family: 'Microsoft YaHei'; font-size: 18px;")
        
        text = "Prediction Results:\n\n"
        for i, word in enumerate(predicted_words):
            text += f"Trial {i+1:02d}: {word}\n"
        results_text.setText(text)
        layout.addWidget(results_text)
        
        # OK按钮
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(result_dialog.accept)
        layout.addWidget(ok_button)
        
        result_dialog.setLayout(layout)
        result_dialog.exec_()