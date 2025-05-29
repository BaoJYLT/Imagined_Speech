import os
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
from sklearn.metrics import f1_score

class PerformanceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent    # 其父窗口即为Main
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Model Performance')
        layout = QVBoxLayout()
        self.parent.statusBar().showMessage(
                f'Current User: {self.parent.user_name} | ID: {self.parent.user_id} | Status: Model performance Evaluation'
            )
        
         # 创建模型选择区域
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        select_btn = QPushButton('Select Model File')
        select_btn.clicked.connect(self.select_model_file)
        self.model_label = QLabel('No model selected')
        self.model_label.setStyleSheet("font-family: 'Microsoft YaHei'; font-size: 20px; color: #666;")
        
        model_layout.addWidget(select_btn)
        model_layout.addWidget(self.model_label)
        model_group.setLayout(model_layout)
        
        # 创建选项卡组件
        self.tab_widget = QTabWidget()
        layout.addWidget(model_group)
        layout.addWidget(self.tab_widget)
        
        self.setLayout(layout)
        self.resize(1200, 900)

    def select_model_file(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File to View Performance",
            "./models",
            "PyTorch Models (*.pth)"
        )
        
        if model_path:
            model_name = os.path.basename(model_path)
            self.model_label.setText(f"Selected model: {model_name}")
            self.load_and_show_performance(model_path)

    def load_and_show_performance(self, model_path):
        try:
            # 加载模型信息
            model_info = torch.load(model_path, map_location='cpu')
            performance = model_info['performance']
            
            self.tab_widget.clear()# 清除现有选项卡  
            self.add_metrics_tab(model_info)# 指标
            self.add_curves_tab(performance)# curve
            self.add_confusion_matrix_tab(performance)# 混淆矩阵
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load performance data: {str(e)}')

    def add_metrics_tab(self, model_info):
        metrics_tab = QWidget()
        layout = QVBoxLayout()
        
        performance = model_info['performance']
        f1_macro = f1_score(performance['y_true'], performance['y_pred'], average='macro')
        f1_micro = f1_score(performance['y_true'], performance['y_pred'], average='micro')
        f1_weighted = f1_score(performance['y_true'], performance['y_pred'], average='weighted')
        
        metrics_text = QTextEdit()
        metrics_text.setReadOnly(True)
        metrics_text.setStyleSheet("font-family: 'Consolas'; font-size: 20px;")
        metrics_text.setText(
            f"Model Performance Metrics:\n\n"
            f"Best Seed: {model_info['best_seed']}\n"
            f"Best Accuracy: {model_info['best_acc']:.4f}\n"
            f"F1 Score (Macro): {f1_macro:.4f}\n"
            f"F1 Score (Micro): {f1_micro:.4f}\n"
            f"F1 Score (Weighted): {f1_weighted:.4f}"
        )
        
        layout.addWidget(metrics_text)
        metrics_tab.setLayout(layout)
        self.tab_widget.addTab(metrics_tab, "Metrics")

    def add_curves_tab(self, performance):
        curves_tab = QWidget()
        layout = QVBoxLayout()
        
        figure = Figure(figsize=(8, 6))
        canvas = FigureCanvas(figure)
        
        # 训练损失曲线
        ax1 = figure.add_subplot(211)
        ax1.plot(performance['train_loss'], label='Training Loss')
        ax1.set_title('Training Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 验证准确率曲线
        ax2 = figure.add_subplot(212)
        ax2.plot(performance['val_acc'], label='Validation Accuracy')
        ax2.set_title('Validation Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        figure.tight_layout()
        layout.addWidget(canvas)
        
        curves_tab.setLayout(layout)
        self.tab_widget.addTab(curves_tab, "Learning Curves")

    def add_confusion_matrix_tab(self, performance):
        cm_tab = QWidget()
        layout = QVBoxLayout()
        
        figure = Figure(figsize=(6, 5))
        canvas = FigureCanvas(figure)
        
        ax = figure.add_subplot(111)
        sns.heatmap(
            performance['final_confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        layout.addWidget(canvas)
        cm_tab.setLayout(layout)
        self.tab_widget.addTab(cm_tab, "Confusion Matrix")