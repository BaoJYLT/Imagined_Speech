import os
import re
import numpy as np
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from EWLP import extract_wavelet_features, EEGWaveletLSTM, init_weights, set_seed, train_model
from torch.utils.data import DataLoader, TensorDataset

class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent  # 保存父窗口引用
        self.files = {
            'train_data': None,
            'train_label': None,
            'val_data': None,
            'val_label': None
        }
        self.model_name = ""
        self.save_path = ""
        self.user_id = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Training Setup')
        layout = QVBoxLayout()
        
        # 文件选择组GROUP
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

        # 模型保存设置区域GROUP
        save_group = QGroupBox("Model Save Settings")
        save_layout = QGridLayout()
        # 模型名称
        save_layout.addWidget(QLabel("Model Name:"), 0, 0)
        self.model_name_edit = QLineEdit()
        save_layout.addWidget(self.model_name_edit, 0, 1)
        # 保存路径
        save_layout.addWidget(QLabel("Save Path:"), 1, 0)
        self.save_path_edit = QLineEdit()
        self.save_path_btn = QPushButton("Browse")
        self.save_path_btn.clicked.connect(self.select_save_path)
        save_layout.addWidget(self.save_path_edit, 1, 1)
        save_layout.addWidget(self.save_path_btn, 1, 2)
        
        save_group.setLayout(save_layout)

        # 确认按钮
        self.confirm_btn = QPushButton('Start Train')
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.validate_files)
        
        layout.addWidget(file_group)
        layout.addWidget(save_group)
        layout.addWidget(self.confirm_btn)
        self.setLayout(layout)

    def select_file(self, file_type):
        """选择数据文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {file_type} file",
                "EEG_DATASET",  # 默认目录
                "NPY files (*.npy)"
            )
            
            if file_path:
                # user_id
                match = re.search(r'Sample(\d{2})', file_path)
                if match:
                    file_user_id = match.group(1)
                    if self.user_id is not None and self.user_id != file_user_id:
                        QMessageBox.warning(
                            self,
                            "Warning",
                            f"Selected file's user ID ({file_user_id}) does not match previously selected files ({self.user_id})"
                        )
                        return
                    self.user_id = file_user_id
                
                self.files[file_type] = file_path
                self.labels[file_type].setText(os.path.basename(file_path))
                self.check_enable_train()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to select file: {str(e)}')
    
    def check_enable_train(self):
        """检查是否可以启用训练按钮"""
        can_train = bool(
            all(self.files.values()) and  # 所有文件都已选择
            self.model_name_edit.text().strip() and  # 模型名称不为空
            self.save_path_edit.text().strip()  # 保存路径不为空
        )
        self.confirm_btn.setEnabled(can_train)

    def select_save_path(self):#模型保存路径
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            "./models"
        )
        if path:
            self.save_path_edit.setText(path)
            self.check_enable_train()

    def validate_files(self):
        if not self.validate_inputs():
            return
            
        try:
            self.parent.user_id = self.user_id  # 父窗口ID更新
            self.parent.statusBar().showMessage(
                f'Current User: {self.parent.user_name} | ID: {self.user_id} | Status: Training...'
            )# 状态栏更新
        
            self.parent.update_eeg_display(self.user_id)

            self.start_training()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Training failed: {str(e)}')

    def validate_inputs(self):
        if not all(self.files.values()):
            QMessageBox.warning(self, 'Error', 'Please select all required files!')
            return False
            
        if not self.model_name_edit.text().strip():
            QMessageBox.warning(self, 'Error', 'Please enter model name!')
            return False
            
        if not self.save_path_edit.text().strip():
            QMessageBox.warning(self, 'Error', 'Please select save path!')
            return False
            
        self.model_name = self.model_name_edit.text().strip()
        self.save_path = self.save_path_edit.text().strip()
        return True

    def start_training(self):
        try:
            progress = QProgressDialog("Training...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Training Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.setMinimumDuration(0)
            self.setup_progress_dialog_style(progress)
            progress.show()

            # 训练数据
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            train_data, train_labels, val_data, val_labels = self.load_data()
            train_features, val_features = self.extract_features(train_data, val_data)
            
            # 训练模型
            best_model, best_acc, best_seed, best_performance = self.train_with_multiple_seeds(
                train_features, train_labels,
                val_features, val_labels,
                device, progress
            )

            self.save_model_and_performance(best_model, best_acc, best_seed, best_performance)
            # 状态栏更新
            self.parent.statusBar().showMessage(
                f'Current User: {self.parent.user_name} | ID: {self.user_id} | Status: Model Saved'
            )
            self.finish_training(progress)
            
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Training failed: {str(e)}')
            raise

    def setup_progress_dialog_style(self, progress):
        """设置进度对话框样式"""
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

    def load_data(self):
        """加载训练数据"""
        train_data = np.load(self.files['train_data'])
        train_labels = np.load(self.files['train_label'])
        val_data = np.load(self.files['val_data'])
        val_labels = np.load(self.files['val_label'])
        return train_data, train_labels, val_data, val_labels

    def extract_features(self, train_data, val_data):
        """特征提取"""
        train_features = extract_wavelet_features(train_data)
        val_features = extract_wavelet_features(val_data)
        return train_features, val_features

    def train_with_multiple_seeds(self, train_features, train_labels, val_features, val_labels, device, progress):
        """使用多个种子训练模型"""
        TOTAL_SEEDS = 351
        best_seed = None
        best_acc = 0
        best_model = None
        best_performance = None

        def update_progress(epoch_progress, message, current_seed):
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
                    torch.tensor(val_labels, dtype=torch.long)
                ),
                batch_size=64
            )
            trained_model, final_acc, performance = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=1000,
                patience=50,
                progress_callback=lambda p, m: update_progress(p, m, seed)
            )
            if final_acc > best_acc:
                best_acc = final_acc
                best_seed = seed
                best_model = trained_model
                best_performance = performance

        return best_model, best_acc, best_seed, best_performance

    def save_model_and_performance(self, best_model, best_acc, best_seed, best_performance):
        """保存模型和性能指标"""
        model_info = {
            'model_state': best_model.state_dict(),
            'performance': best_performance,
            'best_seed': best_seed,
            'best_acc': best_acc
        }
        
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(self.save_path, f'{self.parent.user_name}_{self.model_name}_ID{self.user_id}.pth')
        torch.save(model_info, model_path)
        
        # 成功保存后 信息check
        QMessageBox.information(
            self, 'Success',
            f'Model training completed!\n\n'
            f'Model Name: {self.model_name}\n'
            f'Best Seed: {best_seed}\n'
            f'Best Accuracy: {best_acc:.4f}\n'
            f'Saved to: {model_path}'
        )

    def finish_training(self, progress):
        """完成训练"""
        progress.setLabelText("Training completed!")
        progress.setValue(100)
        progress.setCancelButtonText("OK")
        while not progress.wasCanceled():
            QApplication.processEvents()
        progress.close()

        # 状态栏更新
        self.parent.statusBar().showMessage(
            f'Current User: {self.parent.user_name} | ID: {self.user_id} | Status: Model Saved'
        )