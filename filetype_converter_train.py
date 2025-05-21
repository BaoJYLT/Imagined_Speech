import os
import numpy as np
import h5py
import scipy.io as sio
from glob import glob

def unpack(obj):
    """递归拆包 MATLAB HDF5 dataset 或 cell，使其变成 ndarray 或 Python list。"""
    if isinstance(obj, h5py.Dataset):
        return obj[()]  # 直接读出为 ndarray
    elif isinstance(obj, h5py.Group):
        # 假设 cell 数组存成 Group，按索引键排序取
        keys = sorted(obj.keys(), key=lambda k: int(k))
        arrs = [unpack(obj[k]) for k in keys]
        return np.array(arrs)
    else:
        return obj

def process_mat_file(mat_path):
    """处理 .mat 文件，返回拆包后的数据。"""
    try:
        with h5py.File(mat_path, 'r') as f:
            # 1. 自动匹配 epo_train 组
            epo_key = next((k for k in f.keys() if k.startswith('epo')), None)
            if epo_key is None:
                print(f"  ⚠️ 未找到 epo* 组，跳过")
                return None
            grp = f[epo_key]

            # 2. 拆包原始数据 (time, chan, trials)
            raw = unpack(grp['x'])  # 可能是 (time,chan,trials)
            raw = raw.astype(np.float32)

            # 4. 拆包 className 确定类别数
            cls = unpack(grp['className'])
            n_classes = len(cls)

        # 6. 转置 raw -> (trials, chan, time)
        data = raw.transpose(2, 1, 0)

        return data
    except Exception as e:
        print(f"  ⚠️ 无法处理 HDF5 文件 {mat_path}: {e}")
        return None

def process_old_mat_file(mat_path):
    """处理旧版 .mat 文件 (非 HDF5)，返回拆包后的数据。"""
    try:
        mat = sio.loadmat(mat_path)
        epo = mat['epo_validation'][0, 0]
        
        # 拆包 EEG 数据 (time, chan, trials)
        raw = unpack(epo['x'])           # 得到形如 (time_steps, n_channels, n_trials) 的 ndarray
        raw = raw.astype(np.float32)
        
        # 转置为 (n_trials, n_channels, time_steps)
        data = raw.transpose(2, 1, 0)

        return data
    except Exception as e:
        print(f"  ⚠️ 无法处理旧版 .mat 文件 {mat_path}: {e}")
        return None

mat_dir = "Validation_set"
out_dir = "Validation_set_npy"
os.makedirs(out_dir, exist_ok=True)

for mat_path in glob(os.path.join(mat_dir, "*.mat")):
    print("Processing", mat_path)

    # 尝试处理 HDF5 格式文件
    data = process_mat_file(mat_path)
    
    # 如果是旧版 .mat 文件，使用 scipy 加载
    if data is None:
        data = process_old_mat_file(mat_path)

    if data is not None:
        # 保存为 npy 文件
        base = os.path.splitext(os.path.basename(mat_path))[0]
        npy_file_path = os.path.join(out_dir, f"{base}_data.npy")
        np.save(npy_file_path, data)
        print(f"  ✅ Saved {base}_data.npy {data.shape}")
    else:
        print(f"  ⚠️ {mat_path} 转换失败")

print("All done. Outputs in", out_dir)
