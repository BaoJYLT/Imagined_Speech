# %% [markdown]
# # 所有函数放这里

# %%
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



plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# %% [markdown]
# ## 预处理

# %%

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

# %% [markdown]
# ## 数据增强

# %%
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


# %% [markdown]
# ## 训练相关

# %%
# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
# ================== 小波特征提取 ===================
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

# ================== LSTM 模型定义 ===================
'''初始化+forward
使用LSTM网络处理时间序列
两层全连接进行分类
使用Dropout防止过拟合
'''
class EEGWaveletLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.5):
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
    
# ================== 自定义权重初始化 ===================
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

# 训练流程 返回一个model 和best_acc
# 一个train_model一个二分类
def train_model(model, train_loader, val_loader, device, epochs=1000, patience=50, model_save_path="best_model.pth"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    train_loss_list = []
    val_acc_list = []

    best_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
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

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            torch.save(best_state, model_save_path)  # ✅ 保存模型参数
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered")
                break

    if best_state:
        model.load_state_dict(best_state)

    '''
    # 绘制训练损失和验证准确率曲线
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss", marker='o')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(val_acc_list, marker='o')
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    '''

    return model, best_acc


# %% [markdown]
# ## 测试集

# %%
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

# %% [markdown]
# # 开始吧

# %% [markdown]
# ## 生成预处理之后的数据

# %%
# 创建保存路径
os.makedirs("Training_Sample01_preprocess", exist_ok=True)
os.makedirs("Validation_Sample01_preprocess", exist_ok=True)
os.makedirs("Test_Sample01_preprocess", exist_ok=True)
# 生成不同的二分类模型数据集

# 生成 0-1 类别的数据集
train_data_0_1, train_labels_0_1 = preprocess_file(
    "Training_set_npy_3fenlei/Data_Sample01_data.npy",
    "Training_set_npy_3fenlei/Data_Sample01_labels.npy",
    "Training_Sample01_preprocess/Data_Sample01_data_pre_0_1.npy",
    "Training_Sample01_preprocess/Data_Sample01_labels_pre_0_1.npy",
    labels_to_select=[0, 2],
    label_mapping={0: 0, 2: 1}
)

val_data_0_1, val_labels_0_1 = preprocess_file(
    "Validation_set_npy_3fenlei/Data_Sample01_data.npy",
    "Validation_set_npy_3fenlei/Data_Sample01_labels.npy",
    "Validation_Sample01_preprocess/Data_Sample01_data_pre_0_1.npy",
    "Validation_Sample01_preprocess/Data_Sample01_labels_pre_0_1.npy",
    labels_to_select=[0, 2],
    label_mapping={0: 0, 2: 1}
)

# 生成 0-2 类别的数据集
train_data_0_2, train_labels_0_2 = preprocess_file(
    "Training_set_npy_3fenlei/Data_Sample01_data.npy",
    "Training_set_npy_3fenlei/Data_Sample01_labels.npy",
    "Training_Sample01_preprocess/Data_Sample01_data_pre_0_2.npy",
    "Training_Sample01_preprocess/Data_Sample01_labels_pre_0_2.npy",
    labels_to_select=[0, 4],
    label_mapping={0: 0, 4: 1}
)

val_data_0_2, val_labels_0_2 = preprocess_file(
    "Validation_set_npy_3fenlei/Data_Sample01_data.npy",
    "Validation_set_npy_3fenlei/Data_Sample01_labels.npy",
    "Validation_Sample01_preprocess/Data_Sample01_data_pre_0_2.npy",
    "Validation_Sample01_preprocess/Data_Sample01_labels_pre_0_2.npy",
    labels_to_select=[0, 4],
    label_mapping={0: 0, 4: 1}
)

# 生成 1-2 类别的数据集
train_data_1_2, train_labels_1_2 = preprocess_file(
    "Training_set_npy_3fenlei/Data_Sample01_data.npy",
    "Training_set_npy_3fenlei/Data_Sample01_labels.npy",
    "Training_Sample01_preprocess/Data_Sample01_data_pre_1_2.npy",
    "Training_Sample01_preprocess/Data_Sample01_labels_pre_1_2.npy",
    labels_to_select=[2, 4],
    label_mapping={2: 0, 4: 1}
)

val_data_1_2, val_labels_1_2 = preprocess_file(
    "Validation_set_npy_3fenlei/Data_Sample01_data.npy",
    "Validation_set_npy_3fenlei/Data_Sample01_labels.npy",
    "Validation_Sample01_preprocess/Data_Sample01_data_pre_1_2.npy",
    "Validation_Sample01_preprocess/Data_Sample01_labels_pre_1_2.npy",
    labels_to_select=[2, 4],
    label_mapping={2: 0, 4: 1}
)

test_data = preprocess_file_test(
    "Test_set_npy/Data_Sample01_data.npy",
    "Test_Sample01_preprocess/Data_Sample01_data_pre.npy",
)

# 可视化预处理后的样本（例如0-1数据集）
plot_example_signal(val_data_0_1)

# %% [markdown]
# ## 数据增强 基于前面的preprocess生成

# %%
augment_data(
    train_data_path="Training_Sample01_preprocess/Data_Sample01_data_pre_0_1.npy",
    train_labels_path="Training_Sample01_preprocess/Data_Sample01_labels_pre_0_1.npy",
    output_data_path="Training_Sample01_preprocess/Data_Sample01_data_aug_0_1.npy",
    output_labels_path="Training_Sample01_preprocess/Data_Sample01_labels_aug_0_1.npy"
)

augment_data(
    train_data_path="Training_Sample01_preprocess/Data_Sample01_data_pre_0_2.npy",
    train_labels_path="Training_Sample01_preprocess/Data_Sample01_labels_pre_0_2.npy",
    output_data_path="Training_Sample01_preprocess/Data_Sample01_data_aug_0_2.npy",
    output_labels_path="Training_Sample01_preprocess/Data_Sample01_labels_aug_0_2.npy"
)

augment_data(
    train_data_path="Training_Sample01_preprocess/Data_Sample01_data_pre_1_2.npy",
    train_labels_path="Training_Sample01_preprocess/Data_Sample01_labels_pre_1_2.npy",
    output_data_path="Training_Sample01_preprocess/Data_Sample01_data_aug_1_2.npy",
    output_labels_path="Training_Sample01_preprocess/Data_Sample01_labels_aug_1_2.npy"
)

# %% [markdown]
# ## 先获得最佳种子数

# %%
# ================== 尝试多个种子并输出最终准确率 ===================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据
train_data_0_1 = np.load("Training_Sample01_preprocess/Data_Sample01_data_aug_0_1.npy")
train_labels_0_1 = np.load("Training_Sample01_preprocess/Data_Sample01_labels_aug_0_1.npy")
train_data_0_2 = np.load("Training_Sample01_preprocess/Data_Sample01_data_aug_0_2.npy")
train_labels_0_2 = np.load("Training_Sample01_preprocess/Data_Sample01_labels_aug_0_2.npy")
train_data_1_2 = np.load("Training_Sample01_preprocess/Data_Sample01_data_aug_1_2.npy")
train_labels_1_2 = np.load("Training_Sample01_preprocess/Data_Sample01_labels_aug_1_2.npy")

val_data_0_1 = np.load("Validation_Sample01_preprocess/Data_Sample01_data_pre_0_1.npy")
val_labels_0_1 = np.load("Validation_Sample01_preprocess/Data_Sample01_labels_pre_0_1.npy")
val_data_0_2 = np.load("Validation_Sample01_preprocess/Data_Sample01_data_pre_0_2.npy")
val_labels_0_2 = np.load("Validation_Sample01_preprocess/Data_Sample01_labels_pre_0_2.npy")
val_data_1_2 = np.load("Validation_Sample01_preprocess/Data_Sample01_data_pre_1_2.npy")
val_labels_1_2 = np.load("Validation_Sample01_preprocess/Data_Sample01_labels_pre_1_2.npy")

# 提取特征
train_features_0_1 = extract_wavelet_features(train_data_0_1)
val_features_0_1 = extract_wavelet_features(val_data_0_1)
train_features_0_2 = extract_wavelet_features(train_data_0_2)
val_features_0_2 = extract_wavelet_features(val_data_0_2)
train_features_1_2 = extract_wavelet_features(train_data_1_2)
val_features_1_2 = extract_wavelet_features(val_data_1_2)




# %%
# 训练每个模型
accuracies_0_1 = []
best_seed = None
best_acc = 0
best_model = None
for seed in range(0, 351):  # 选择种子范围 0-350
    set_seed(seed)
    model = EEGWaveletLSTM(input_dim=train_features_0_1.shape[2])
    model.apply(init_weights)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_features_0_1, dtype=torch.float32), torch.tensor(train_labels_0_1, dtype=torch.long)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_features_0_1, dtype=torch.float32), torch.tensor(val_labels_0_1, dtype=torch.long)), batch_size=64)

    trained_model, final_acc = train_model(model, train_loader, val_loader, device, epochs=1000, patience=50)# model_save_path="best_model_0_1_seed_{seed}.pth"
    accuracies_0_1.append((seed, final_acc))

    # 保存当前最好的模型
    if final_acc > best_acc:
        best_acc = final_acc
        best_seed = seed
        best_model = trained_model

# 输出Top 5准确率的种子
accuracies_0_1.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 Seed Accuracies (0-1):")
for seed, acc in accuracies_0_1[:5]:
    print(f"Seed: {seed}, Accuracy: {acc:.4f}")

# 在找到最佳种子后保存最佳模型
if best_model is not None:
    best_model_save_path = f"best_model_Sample01_0_1.pth"
    torch.save(best_model.state_dict(), best_model_save_path)
    print(f"最佳模型已保存：{best_model_save_path}")
    
# 训练每个模型（0-2）
accuracies_0_2 = []
best_seed_0_2 = None
best_acc_0_2 = 0
best_model_0_2 = None
for seed in range(0, 351):  # 选择种子范围 0-350
    set_seed(seed)
    model = EEGWaveletLSTM(input_dim=train_features_0_2.shape[2])
    model.apply(init_weights)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_features_0_2, dtype=torch.float32), torch.tensor(train_labels_0_2, dtype=torch.long)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_features_0_2, dtype=torch.float32), torch.tensor(val_labels_0_2, dtype=torch.long)), batch_size=64)

    trained_model, final_acc = train_model(model, train_loader, val_loader, device, epochs=1000, patience=50)
    accuracies_0_2.append((seed, final_acc))

    # 保存当前最好的模型
    if final_acc > best_acc_0_2:
        best_acc_0_2 = final_acc
        best_seed_0_2 = seed
        best_model_0_2 = trained_model

# 输出Top 5准确率的种子（0-2）
accuracies_0_2.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 Seed Accuracies (0-2):")
for seed, acc in accuracies_0_2[:5]:
    print(f"Seed: {seed}, Accuracy: {acc:.4f}")

# 在找到最佳种子后保存最佳模型（0-2）
if best_model_0_2 is not None:
    best_model_save_path_0_2 = f"best_model_Sample01_0_2.pth"
    torch.save(best_model_0_2.state_dict(), best_model_save_path_0_2)
    print(f"最佳模型已保存：{best_model_save_path_0_2}")


# 训练每个模型（1-2）
accuracies_1_2 = []
best_seed_1_2 = None
best_acc_1_2 = 0
best_model_1_2 = None
for seed in range(0, 351):  # 选择种子范围 0-350
    set_seed(seed)
    model = EEGWaveletLSTM(input_dim=train_features_1_2.shape[2])
    model.apply(init_weights)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_features_1_2, dtype=torch.float32), torch.tensor(train_labels_1_2, dtype=torch.long)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_features_1_2, dtype=torch.float32), torch.tensor(val_labels_1_2, dtype=torch.long)), batch_size=64)

    trained_model, final_acc = train_model(model, train_loader, val_loader, device, epochs=1000, patience=50)
    accuracies_1_2.append((seed, final_acc))

    # 保存当前最好的模型
    if final_acc > best_acc_1_2:
        best_acc_1_2 = final_acc
        best_seed_1_2 = seed
        best_model_1_2 = trained_model

# 输出Top 5准确率的种子（1-2）
accuracies_1_2.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 Seed Accuracies (1-2):")
for seed, acc in accuracies_1_2[:5]:
    print(f"Seed: {seed}, Accuracy: {acc:.4f}")

# 在找到最佳种子后保存最佳模型（1-2）
if best_model_1_2 is not None:
    best_model_save_path_1_2 = f"best_model_Sample01_1_2.pth"
    torch.save(best_model_1_2.state_dict(), best_model_save_path_1_2)
    print(f"最佳模型已保存：{best_model_save_path_1_2}")

# %% [markdown]
# ## 带结果的训练

# %%
# =============== 7. 训练函数（加早停机制 + 梯度裁剪）只是没有模型的保存位置 ===============
def train_model(model, train_loader, val_loader, device, epochs=1000, patience=50):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 用来保存每个epoch的损失和验证准确率
    train_loss_list = []
    val_acc_list = []

    best_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
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

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
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

    # 绘制验证准确率曲线
    plt.figure()
    plt.plot(val_acc_list, marker='o')
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 绘制训练损失曲线
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss", marker='o')
    plt.title("Train Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return model



# %%
print(best_seed, best_acc, best_seed_0_2, best_acc_0_2, best_seed_1_2, best_acc_1_2)

# %%
best_seed = 211
best_seed_0_2 = 102
best_seed_1_2 = 121

set_seed(best_seed)
model = EEGWaveletLSTM(input_dim=train_features_0_1.shape[2])
model.apply(init_weights)
train_loader = DataLoader(TensorDataset(torch.tensor(train_features_0_1, dtype=torch.float32), torch.tensor(train_labels_0_1, dtype=torch.long)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(val_features_0_1, dtype=torch.float32), torch.tensor(val_labels_0_1, dtype=torch.long)), batch_size=64)

trained_model = train_model(model, train_loader, val_loader, device, epochs=1000, patience=50)# model_save_path="best_model_0_1_seed_{seed}.pth"


# %%
best_seed = 211
best_seed_0_2 = 102
best_seed_1_2 = 121

set_seed(best_seed_0_2)
model = EEGWaveletLSTM(input_dim=train_features_0_2.shape[2])
model.apply(init_weights)
train_loader = DataLoader(TensorDataset(torch.tensor(train_features_0_2, dtype=torch.float32), torch.tensor(train_labels_0_2, dtype=torch.long)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(val_features_0_2, dtype=torch.float32), torch.tensor(val_labels_0_2, dtype=torch.long)), batch_size=64)

trained_model = train_model(model, train_loader, val_loader, device, epochs=1000, patience=50)# model_save_path="best_model_0_1_seed_{seed}.pth"


# %%
best_seed = 211
best_seed_0_2 = 102
best_seed_1_2 = 121

set_seed(best_seed_1_2)
model = EEGWaveletLSTM(input_dim=train_features_1_2.shape[2])
model.apply(init_weights)
train_loader = DataLoader(TensorDataset(torch.tensor(train_features_1_2, dtype=torch.float32), torch.tensor(train_labels_1_2, dtype=torch.long)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(val_features_1_2, dtype=torch.float32), torch.tensor(val_labels_1_2, dtype=torch.long)), batch_size=64)

trained_model = train_model(model, train_loader, val_loader, device, epochs=1000, patience=50)# model_save_path="best_model_0_1_seed_{seed}.pth"


# %% [markdown]
# ## 使用这三个模型

test_data =  np.load("Test_Sample01_preprocess/Data_Sample01_data_pre.npy") 
# 3. 小波特征提取
test_features = extract_wavelet_features(test_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 构造模型输入并预测
test_X = torch.tensor(test_features, dtype=torch.float32).to(device)

temp_input_dim = test_X.shape[2]
test_model = EEGWaveletLSTM(input_dim=temp_input_dim)
test_model.load_state_dict(torch.load("best_model_Sample01_0_1.pth", map_location=device))
test_model.to(device)
test_model.eval()

with torch.no_grad():
    outputs = test_model(test_X)
    predictions = outputs.argmax(dim=1).cpu().numpy()
    probs = torch.softmax(outputs, dim=1).cpu().numpy()

'''
# 5. 打印预测结果
for i, (pred, prob) in enumerate(zip(predictions, probs)):
    print(f"🧪 Trial {i}: Predicted Class = {pred}, Probabilities = {np.round(prob, 3)}")

# 6. （可选）计算准确率
if 'test_labels' in locals():
    test_acc = accuracy_score(test_labels, predictions)
    print(f"✅ 测试集准确率: {test_acc:.4f}")
'''

# ========== 读取人工标注标签（从图中整理） ==========
manual_true_labels = [
    4,1,5,3,2,5,5,3,2,4,4,4,2,4,3,5,2,2,4,3,
    3,4,1,3,4,4,5,1,2,1,3,3,5,2,1,1,2,5,1,1,
    5,3,4,5,1,5,2,2,3,1
]

true_labels = np.array(manual_true_labels) - 1  # 转为0-based

# 使用前面联合模型的 final_preds 作为 predictions
predictions = predictions  # 这在前面的推理流程中应已生成

# 仅保留标签为 0, 2, 4 的样本用于评估
target_labels = {0, 2}
indices_024 = [i for i, label in enumerate(true_labels) if label in target_labels]

print("以下是仅针对实际为 1/3/5 的样本：")
print("Trial\tTrue(0/1/2)\tPred(0/1/2)")

filtered_true = true_labels[indices_024]
filtered_pred = predictions[indices_024]



# 混淆矩阵（标签映射为 0/1/2）
# 🔁 将 true_labels 的值 [0,2,4] → 映射到 [0,1,2]
label_mapping = {0: 0, 2: 1}  # 显式映射
mapped_true = np.array([label_mapping[label] for label in filtered_true])
mapped_pred = filtered_pred  # 你的模型已经是 0/1/2 输出，直接用

# 计算准确率（仅对0/2/4）
acc_024 = accuracy_score(mapped_true, mapped_pred)
print(f"仅对标签0/2的预测准确率：{acc_024:.4f}")

for idx_in_filtered, idx_original in enumerate(indices_024):
    true_label = mapped_true[idx_in_filtered]
    pred_label = mapped_pred[idx_in_filtered]
    print(f"{idx_original:5d}\t{true_label}\t\t{pred_label}")
    

# 混淆矩阵可视化
cm = confusion_matrix(mapped_true, mapped_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix (labels 0,2,4 only)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# 计算 F1 分数
f1_macro = f1_score(mapped_true, mapped_pred, average='macro')
f1_micro = f1_score(mapped_true, mapped_pred, average='micro')
f1_weighted = f1_score(mapped_true, mapped_pred, average='weighted')
f1_per_class = f1_score(mapped_true, mapped_pred, average=None)

report = {
    "Acc (0/2/4)": round(acc_024, 4),
    "F1 Macro": round(f1_macro, 2),
    "F1 Micro": round(f1_micro, 2),
    "F1 Weighted": round(f1_weighted, 2),
    "F1 Per Class (0,2,4)": [round(x, 2) for x in f1_per_class.tolist()]
}
report

