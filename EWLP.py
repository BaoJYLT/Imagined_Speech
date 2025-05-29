import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import random
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import *
import pandas as pd

# %%预处理

# 萤火虫优化算法（优化软阈值）——用于数据预处理
def firefly_algorithm(detail_coeffs, n_fireflies=15, max_iter=20, alpha=0.2, beta0=1, gamma=1):
    def fitness(threshold):
        thresholded = pywt.threshold(detail_coeffs, threshold, mode='soft')
        return -np.var(thresholded)  # 越平滑越好（负方差）

    thresholds = np.random.uniform(0.01, np.max(np.abs(detail_coeffs)), n_fireflies)
    fitness_vals = np.array([fitness(th) for th in thresholds])

    for t in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness_vals[j] > fitness_vals[i]:
                    r = np.abs(thresholds[i] - thresholds[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    thresholds[i] = thresholds[i] + beta * (thresholds[j] - thresholds[i]) + alpha * (np.random.rand() - 0.5)
        fitness_vals = np.array([fitness(th) for th in thresholds])

    best_idx = np.argmax(fitness_vals)
    return thresholds[best_idx]

# 小波去噪（每个通道应用 FOA-soft 阈值）
def wavelet_denoise(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    thresholded_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        th = firefly_algorithm(detail)
        thresholded = pywt.threshold(detail, th, mode='soft')
        thresholded_coeffs.append(thresholded)
    return pywt.waverec(thresholded_coeffs, wavelet)

# EEG 信号预处理函数（滤波 + FOA-DWT 去噪 + Z-score）
# 修改 preprocess_file 函数以添加打印过程与更多可视化，返回的是processed的数据和labels
# EEG 信号预处理函数（滤波 + FOA-DWT 去噪 + Z-score）
def preprocess_file(data_path, label_path, out_data_path, out_label_path, labels_to_select, label_mapping, fs=256, lowcut=0.5, highcut=50.0):
    print(f"加载数据：{data_path}")
    data = np.load(data_path)  # (n_trials, C, T)
    labels = np.load(label_path)
    print(f"原始数据形状：{data.shape}，标签形状：{labels.shape}")

    # ===== 标签筛选 =====
    selected_mask = np.isin(labels, labels_to_select)
    data = data[selected_mask]
    labels = labels[selected_mask]

    # 标签映射
    labels = np.vectorize(label_mapping.get)(labels)
    print(f"筛选后的样本数：{len(labels)}")

    b, a = butter(N=4, Wn=[lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    start_idx, end_idx = int(1.0 * fs), int(3.0 * fs)

    n_trials, C, T = data.shape
    processed = np.zeros((n_trials, C, end_idx - start_idx), dtype=np.float32)

    for i in range(n_trials):
        print(f"预处理第 {i+1}/{n_trials} 个试次...")
        segment = data[i][:, start_idx:end_idx]
        filtered = filtfilt(b, a, segment, axis=1)
        denoised = np.array([wavelet_denoise(filtered[ch])[:end_idx-start_idx] for ch in range(C)])
        zscored = (denoised - denoised.mean()) / (denoised.std() + 1e-8)
        processed[i] = zscored

    print(f"保存预处理后的数据到：{out_data_path}")
    np.save(out_data_path, processed)
    np.save(out_label_path, labels)
    print("处理完成。\n")
    return processed, labels

# 可视化函数（默认展示 Trial 0, Channel 10）
def plot_example_signal(data, trial_idx=0, channel_idx=10, title="Preprocessed EEG with FOA-DWT"):
    plt.figure(figsize=(10, 4))
    plt.plot(data[trial_idx, channel_idx], label=f"Trial {trial_idx}, Channel {channel_idx}")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (z-scored)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %% 数据增强 
# 【数据增强！！设置好输入输出的data和label路径之后，实现数据的增强和存储】
def augment_data(train_data_path, train_labels_path, output_data_path, output_labels_path, noise_std=0.05, plot_channel=10):

    # 加载训练数据和标签
    train_data = np.load(train_data_path)  # shape: (N, 64, 512)
    train_labels = np.load(train_labels_path)

    # 初始化增强容器
    data_augmented = []
    labels_augmented = []

    print("🔧 正在按 trial 扩展每个样本（三倍增强）")

    for i in range(len(train_data)):
        original = train_data[i]
        label = train_labels[i]

        # 原始数据
        data_augmented.append(original)
        labels_augmented.append(label)

        # 加性噪声
        noisy = original + np.random.normal(0, noise_std, size=original.shape).astype(np.float32)
        data_augmented.append(noisy)
        labels_augmented.append(label)

        if i < 2:
            print(f"✅ Trial {i+1}: 已生成原始+噪声共2份")

    # 转换为 numpy 格式
    data_augmented = np.stack(data_augmented)
    labels_augmented = np.array(labels_augmented)

    print(f"🎉 原始样本数: {len(train_data)} → 增强后样本数: {len(data_augmented)}")

    # 保存增强数据和标签
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    np.save(output_data_path, data_augmented)
    np.save(output_labels_path, labels_augmented)
    print("📁 已保存三倍增强数据与标签")

    # 可视化样本增强效果（通道 10）
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(data_augmented[0][plot_channel])
    plt.title("原始样本")

    plt.subplot(3, 1, 2)
    plt.plot(data_augmented[1][plot_channel])
    plt.title("加性噪声样本")
    
    plt.tight_layout()
    plt.show()

# %% 训练相关
# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# %% 特征提取 小波
def extract_wavelet_features(data, wavelet='db4', level=2):
    n_trials, n_channels, n_samples = data.shape
    features = []
    for trial in data:
        trial_features = []
        for ch_signal in trial:
            coeffs = pywt.wavedec(ch_signal, wavelet=wavelet, level=level)
            feature = np.concatenate([coeffs[0], coeffs[1]])  # cA2 + cD2
            trial_features.append(feature)
        features.append(trial_features)
    return np.array(features)

# %% LSTM模型相关（结构与初始化）
'''初始化+forward
使用LSTM网络处理时间序列
两层全连接进行分类
使用Dropout防止过拟合
'''
class EEGWaveletLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
# %% train_model 模型训练
# 训练流程 返回一个model 和best_acc
# 一个train_model一个二分类
def train_model_save(model, train_loader, val_loader, device, epochs=1000, patience=50, model_save_path="best_model.pth", progress_callback = None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    train_loss_list = []
    val_acc_list = []

    best_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        progress = int((epoch + 1) * 100 / epochs)
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(1).cpu().numpy()
                y_true.extend(yb.numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        val_acc_list.append(acc)
        print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.4f} - Train Loss: {avg_train_loss:.4f}")

        # 更新进度条信息
        if progress_callback:
            message = (f"Epoch {epoch+1}/{epochs}\n"
                      f"Validation Accuracy: {acc:.4f}\n"
                      f"Training Loss: {avg_train_loss:.4f}")
            progress_callback(progress, message)


        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            torch.save(best_state, model_save_path)  # 保存最佳模型的参数
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback(100, "Early stopping triggered!")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_acc

def train_model(model, train_loader, val_loader, device, epochs=1000, patience=5, progress_callback = None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 用来保存每个epoch的损失和验证准确率
    train_loss_list = []
    val_acc_list = []
    best_acc = 0
    best_state = None
    patience_counter = 0

    # best performance
    best_performance = {
        'train_loss':[],
        'val_acc':[],
        'final_confusion_matrix': None,
        'y_true': None,
        'y_pred': None
    }

    for epoch in range(epochs):
        progress = int((epoch+1)* 100 / epochs)
        model.train()
        total_train_loss = 0  # 用于记录一个epoch的训练损失
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)  # 累加每个batch的损失

        avg_train_loss = total_train_loss / len(train_loader.dataset)  # 计算平均训练损失
        train_loss_list.append(avg_train_loss)  # 保存该epoch的训练损失

        # 评估阶段
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(1).cpu().numpy()
                y_true.extend(yb.numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        val_acc_list.append(acc)
        print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.4f} - Train Loss: {avg_train_loss:.4f}")

        # 更新进度信息
        if progress_callback:
            message = (f"Epoch {epoch+1}/{epochs}\n"
                      f"Validation Accuracy: {acc:.4f}\n"
                      f"Training Loss: {avg_train_loss:.4f}")
            progress_callback(progress, message)

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            # 保存最佳的性能指标
            best_performance['train_loss'] = train_loss_list
            best_performance['val_acc'] = val_acc_list
            best_performance['final_confusion_matrix'] = confusion_matrix(y_true=y_true, y_pred=y_pred)
            best_performance['y_true'] = y_true
            best_performance['y_pred'] = y_pred

            # torch.save(best_state, "best_model_3fenlei_01.pth")  # ✅ 保存模型参数
            print(f"Best model saved with accuracy: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered")
                break

    if best_state:
        model.load_state_dict(best_state)

    # 训练完成时的最终进度更新
    if progress_callback:
        progress_callback(100, "Training completed!\nClick OK to continue...")


    return model, best_acc, best_performance

# %% 测试集预处理
def preprocess_file_test(data_path,  out_data_path,  fs=256, lowcut=0.5, highcut=50.0):
    
    data = data_path  # (n_trials, C, T)

    b, a = butter(N=4, Wn=[lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    start_idx, end_idx = int(1.0 * fs), int(3.0 * fs)

    n_trials, C, T = data.shape
    processed = np.zeros((n_trials, C, end_idx - start_idx), dtype=np.float32)

    for i in range(n_trials):
        print(f"预处理第 {i+1}/{n_trials} 个试次...")
        segment = data[i][:, start_idx:end_idx]
        filtered = filtfilt(b, a, segment, axis=1)
        denoised = np.array([wavelet_denoise(filtered[ch])[:end_idx-start_idx] for ch in range(C)])
        zscored = (denoised - denoised.mean()) / (denoised.std() + 1e-8)
        processed[i] = zscored

    print(f"保存预处理后的数据到：{out_data_path}")
    np.save(out_data_path, processed)

    print("处理完成。\n")
    return processed

# %% 