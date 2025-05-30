from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton, 
                            QVBoxLayout, QMessageBox, QMainWindow)
import re

'''
initUI()    构建对话框界面，布局（标签、输入框、按钮）
validate_user() ID输入格式、对应文件存在性、用户注册状态设置
get_current_user()  返回当前登录的用户ID
'''

class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.user_name = None
        
    def initUI(self):
        self.setWindowTitle('Registeration')
        self.setModal(True) # 设置为模态窗口，阻塞主界面
        
        # 创建垂直布局，放置控件
        layout = QVBoxLayout()

        self.name_label = QLabel('Username (letters and numbers):', self)
        self.name_input = QLineEdit(self) # ID输入框 提示示例ID
        self.name_input.setPlaceholderText('e.g. user123')
        
        # 创建OK按钮，用connect将这个按钮绑定validate_user方法
        self.confirm_button = QPushButton('OK', self)
        self.confirm_button.clicked.connect(self.validate_user)
        
        # 添加组件到布局
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.confirm_button)
        
        self.setLayout(layout)

    
    def validate_user(self):
        name = self.name_input.text().strip()
        # 检查用户名格式：数字+字母
        if not re.match(r'^[a-zA-Z0-9]+$', name):
            QMessageBox.warning(self, 'Invalid Input', 
                              'Username can only contain letters and numbers')
            return
            
        self.user_name = name
        QMessageBox.information(self, 'Success', 
                              f'Welcome, {name}!')
        self.accept()

    def get_user_name(self):
        return self.user_name