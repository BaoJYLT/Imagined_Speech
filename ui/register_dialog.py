from PyQt5.QtWidgets import *

class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 用户名输入
        self.username_input = QLineEdit()
        layout.addWidget(QLabel("Username:"))
        layout.addWidget(self.username_input)
        
        # 确认按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        
        layout.addWidget(buttons)
        self.setLayout(layout)
        
    def validate_and_accept(self):
        username = self.username_input.text()
        if self.check_user_data_exists(username):
            self.accept()
        else:
            QMessageBox.warning(self, "Error", 
                              "Sorry, service not available.")
            
    def check_user_data_exists(self, username):
        # 检查用户数据是否存在
        # 01-15范围内
        pass