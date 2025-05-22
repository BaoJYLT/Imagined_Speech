"""
UI模块初始化文件
界面组件的导入和版本信息
"""

from .main_window import MainWindow
from .register_dialog import RegisterDialog
from .train_dialog import TrainDialog
from .savemodel_dialog import SaveModelDialog

__version__ = '0.1.0'
__author__ = 'BaoJYLT'

# 导出主要的类，使其可以直接从ui包中导入'Class Name'
__all__ = [
    'MainWindow',
    'RegisterDialog',
    'TrainDialog',
    'SaveModelDialog',
]