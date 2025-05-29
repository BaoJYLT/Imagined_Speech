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
        # self.current_user = None
        
    def initUI(self):
        self.setWindowTitle('Registeration')
        self.setModal(True) # 设置为模态窗口，阻塞主界面
        
        # 创建垂直布局，放置控件
        layout = QVBoxLayout()
        
        # 创建输入框和标签
        # self.user_label = QLabel('User ID (01-99):', self)
        # self.user_input = QLineEdit(self) # ID输入框 提示示例ID
        # self.user_input.setPlaceholderText('ID: 01')

        self.name_label = QLabel('Username (letters and numbers):', self)
        self.name_input = QLineEdit(self) # ID输入框 提示示例ID
        self.name_input.setPlaceholderText('e.g. user123')
        
        # 创建OK按钮，用connect将这个按钮绑定validate_user方法
        self.confirm_button = QPushButton('OK', self)
        self.confirm_button.clicked.connect(self.validate_user)
        
        # 添加组件到布局
        # layout.addWidget(self.user_label)
        # layout.addWidget(self.user_input)
        # layout.addWidget(self.confirm_button)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.confirm_button)
        
        self.setLayout(layout)

    '''    
    def validate_user(self):
        user_id = self.user_input.text().strip() # 去除前后空白的用户输入
        
        # 正则表达式检查ID两位数字输入regular expression
        if not re.match(r'^[0-9]{2}$', user_id):
            QMessageBox.warning(self, 'Illegal Input', 
                              'Please enter a two-digit ID (01-99)')# “非法输入，请输入两位ID”
            return
        
        # 检查用户数据文件是否存在（基于输入的ID，构造训练集数据路径和label文件路径）
        train_data_path = f'Training_set_npy/Data_Sample{user_id}_data.npy'
        train_label_path = f'Training_set_npy/Data_Sample{user_id}_labels.npy'
        
        if not os.path.exists(train_data_path) or not os.path.exists(train_label_path):
            QMessageBox.warning(self, 'User files not found', 
                              'Sorry, service not available.')
            return
            
        # 用户验证成功
        self.current_user = user_id
        # 信息提示框，通知登录成功
        QMessageBox.information(self, 'Success', 
                              f'User {user_id} Log in successfully!')
        # 通知父窗口更新
        if isinstance(self.parent(), QMainWindow):
            self.parent().update_eeg_display(user_id)
        self.accept()
    '''
    
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

    # def get_current_user(self):
    #     return self.current_user

    def get_user_name(self):
        return self.user_name