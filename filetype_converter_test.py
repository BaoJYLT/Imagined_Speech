import os
import numpy as np
import h5py
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

mat_dir = "Test_set"
out_dir = "Test_set_npy"
os.makedirs(out_dir, exist_ok=True)

for mat_path in glob(os.path.join(mat_dir, "*.mat")):
    print("Processing", mat_path)
    with h5py.File(mat_path, 'r') as f:
        # 1. 自动匹配 epo_train 组
        epo_key = next((k for k in f.keys() if k.startswith('epo')), None)
        if epo_key is None:
            print(f"  ⚠️ 未找到 epo* 组，跳过")
            continue
        grp = f[epo_key]

        # 2. 拆包原始数据 (time, chan, trials)
        raw = unpack(grp['x'])           # 可能是 (time,chan,trials)
        raw = raw.astype(np.float32)

        # 4. 拆包 className 确定类别数
        cls = unpack(grp['className'])
        n_classes = len(cls)

    # 6. 转置 raw -> (trials, chan, time)
    data = raw.transpose(2, 1, 0)

    # 7. 保存到 npy
    base = os.path.splitext(os.path.basename(mat_path))[0]
    np.save(os.path.join(out_dir, f"{base}_data.npy"),   data)
    print(f"  ✅ Saved {base}_data.npy {data.shape}")

print("All done. Outputs in", out_dir)
