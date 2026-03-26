import os
import glob
import torch
import cv2  # 新增导入 cv2
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat

class LOSDataset(Dataset):
    """
    单光子成像数据集类
    读取.mat文件，包含spad、albedo_hr、dist_hr数组。
    支持自动 Resize 至 1024x1024 并进行全局 Min-Max 归一化。
    """
    def __init__(self, dataset_dir, split='train', train_ratio=0.9, transform=None, seed=42):
        self.dataset_dir = dataset_dir
        self.split = split
        self.train_ratio = train_ratio
        self.transform = transform
        
        # 找到所有.mat文件并排序
        self.file_list = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.mat'), recursive=True))
        
        if len(self.file_list) == 0:
            raise RuntimeError(f"在 {dataset_dir} 中未找到.mat文件")
        
        print(f"找到 {len(self.file_list)} 个.mat文件")
        
        # 按比例划分训练集和测试集
        total_files = len(self.file_list)
        train_size = int(total_files * train_ratio)
        
        # 设置随机种子以确保可重复性
        generator = torch.Generator().manual_seed(seed)
        
        # 划分数据集
        indices = torch.randperm(total_files, generator=generator).tolist()
        
        if split == 'train':
            self.indices = indices[:train_size]
            print(f"训练集: {len(self.indices)} 个文件")
        elif split == 'test':
            self.indices = indices[train_size:]
            print(f"测试集: {len(self.indices)} 个文件")
        else:
            raise ValueError(f"split参数必须是 'train' 或 'test', 得到 {split}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_path = self.file_list[self.indices[idx]]
        mat_data = loadmat(file_path)
        
        # 1. 提取 SPAD 数据并自适应稀疏矩阵
        spad_raw = mat_data['spad']
        if hasattr(spad_raw, 'toarray'): 
            spad_raw = spad_raw.toarray()
            
        # 如果 MATLAB 中将其 reshape 成了二维，在这里恢复为三维 (128, 128, 512)
        if spad_raw.ndim == 2:
            res = int(np.sqrt(spad_raw.shape[0]))
            spad_raw = spad_raw.reshape((res, res, spad_raw.shape[1]))
            
        spad = spad_raw.astype(np.float32)                   # (128, 128, 512)
        albedo_hr = mat_data['albedo_hr'].astype(np.float32) # (512, 512, 3)
        dist_hr = mat_data['dist_hr'].astype(np.float32)     # (512, 512)
        
        # ================= 修改核心区域开始 =================
        
        # 2. 在 Numpy 层面先进行归一化 (防除零)
        albedo_max = albedo_hr.max()
        if albedo_max > 0:
            albedo_hr = albedo_hr / albedo_max
            
        dist_max = dist_hr.max()
        if dist_max > 0:
            dist_hr = dist_hr / dist_max

        # 3. 使用 cv2 进行 Resize (默认双线性插值)
        # 注意 cv2.resize 接收的目标尺寸为 (width, height)
        target_size = (1024, 1024)
        albedo_hr = cv2.resize(albedo_hr, target_size, interpolation=cv2.INTER_LINEAR)
        dist_hr = cv2.resize(dist_hr, target_size, interpolation=cv2.INTER_LINEAR)

        # 4. 转换为 PyTorch 张量并调整通道顺序
        # Albedo: (1024, 1024, 3) -> (3, 1024, 1024)
        albedo_hr = torch.from_numpy(albedo_hr).permute(2, 0, 1)
        # Dist: (1024, 1024) -> 增加通道维度变为 (1, 1024, 1024)
        dist_hr = torch.from_numpy(dist_hr).unsqueeze(0)
        
        # ================= 修改核心区域结束 =================
        
        # 5. 处理 SPAD (保持原有逻辑)
        spad = torch.from_numpy(spad)
        spad = spad / torch.max(spad)
        # 将 SPAD 转换为4维结构 (1, 512, 128, 128)
        spad = spad.permute(2, 0, 1).unsqueeze(0)
        
        if self.transform:
            spad, albedo_hr, dist_hr = self.transform([spad, albedo_hr, dist_hr])
        
        return {"meas": spad, "inten": albedo_hr, "dep": dist_hr}

# 使用示例
if __name__ == "__main__":
    dataset_dir = "/data/ImageData/LOS/" # 替换为你的路径
    
    # 临时创建测试数据以供调试通过（如果本地没有文件，记得替换回你自己的测试路径）
    test_dataset = LOSDataset(dataset_dir, split='test', train_ratio=0.9)
    print(f"测试集大小: {len(test_dataset)}")
    
    sample = test_dataset[0]
    spad = sample["meas"]
    albedo = sample["inten"]
    dist = sample["dep"]
    
    print(f"SPAD shape: {spad.shape}, Min: {spad.min():.2f}, Max: {spad.max():.2f}")
    print(f"Albedo shape: {albedo.shape}, Min: {albedo.min():.2f}, Max: {albedo.max():.2f}")
    print(f"Distance shape: {dist.shape}, Min: {dist.min():.2f}, Max: {dist.max():.2f}")