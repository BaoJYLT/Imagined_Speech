"""
UI模块初始化文件
界面组件的导入和版本信息
"""

from .main_window import MainWindow
from .register_dialog import RegisterDialog
from .train_dialog import TrainDialog
from .test_dialog import TestDialog
from .performance_dialog import PerformanceDialog

__version__ = '1.0.1'
__author__ = 'BaoJYLT'

__all__ = [
    'MainWindow',
    'RegisterDialog',
    'TrainDialog',
    'TestDialog',
    'PerformanceDialog',
]