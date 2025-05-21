import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import cm

class EEGCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=12, height=8, dpi=100): 
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
    def plot_eeg(self, data, fs=256, user_id=None):
        """绘制EEG数据
        Args:
            data: shape (channels, samples) 的 numpy 数组
            fs: 采样率
            user_id: 用户id
        """
        self.axes.clear()
        n_channels = data.shape[0]
        t = np.arange(data.shape[1]) / fs
        
        # 设置颜色映射
        cmap = cm.get_cmap('viridis')  # 你可以选择其他颜色映射，比如 'plasma', 'inferno', 'jet'
        
        # 为每个通道绘制一条线，并应用颜色映射
        for i in range(n_channels):
            # 添加偏移以区分不同通道
            offset = i * 100
            self.axes.plot(t, data[i, :] + offset, label=f'Channel {i+1}', color=cmap(i / n_channels))  # 使用颜色映射
            
        # 设置纵坐标刻度
        self.axes.set_yticks(np.arange(n_channels) * 100)  # 设置每个通道的刻度位置
        self.axes.set_yticklabels([f'Channel {i+1}' for i in range(n_channels)])  # 设置每个通道的标签
        
        # 设置标题和标签
        self.axes.set_xlabel('Time (s)', fontsize=12)
        self.axes.set_ylabel('Channel', fontsize=12)
        
        title = 'EEG Signal Display'
        if user_id:
            title = f'User {user_id} EEG Signal Display'
        self.axes.set_title(title, fontsize=14, pad=5)
        self.axes.grid(True)
        # self.axes.legend(fontsize=10)
        # 重新绘制画布
        self.draw()
