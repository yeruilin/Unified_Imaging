# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResnetBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             # 分组归一化，注意如果通道数小于32，需调整 num_groups
#             nn.GroupNorm(min(32, in_channels), in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(min(32, out_channels), out_channels),
#             nn.SiLU(),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         )
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.block(x) + self.shortcut(x)

# class LOSEncoder(nn.Module):
#     def __init__(self, in_channels=1, out_dim=16):
#         """
#         单光子成像特征提取网络
#         :param in_channels: 输入通道数，通常单光子数据为 1
#         :param out_dim: 输出特征图的通道数
#         """
#         super().__init__()

#         # 1. 初始 3D 特征提取 (保持空间 N*N 和时间 M 不变)
#         self.feature_3d = nn.Sequential(
#             nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
#             nn.BatchNorm3d(16),
#             nn.SiLU(),
#             nn.Conv3d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm3d(32),
#             nn.SiLU(),
#             nn.Conv3d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm3d(64),
#             nn.SiLU()
#         )
        
#         # 2. 3D 到 2D 的投影映射 (压缩时间维度 M)
#         # 空间维度 H 和 W 的 kernel 和 stride 均为 1，确保 N*N 不变
#         self.project_spatial = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=(8, 1, 1), stride=(8, 1, 1)),
#             nn.BatchNorm3d(128),
#             nn.SiLU()
#         )

#         # 3. 强度特征提取分支 (保持分辨率为 N*N)
#         self.intensity_head = nn.Sequential(
#             ResnetBlock2D(128, 128),
#             ResnetBlock2D(128, 64),
#             nn.GroupNorm(16, 64),
#             nn.SiLU(),
#             nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
#         )
        
#         # 4. 深度特征提取分支 (保持分辨率为 N*N)
#         self.depth_head = nn.Sequential(
#             ResnetBlock2D(128, 128),
#             ResnetBlock2D(128, 64),
#             nn.GroupNorm(16, 64),
#             nn.SiLU(),
#             nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
#         )

#     def forward(self, inputs):
#         """
#         :param inputs: 形状为 (B, M, N, N) 或 (B, 1, M, N, N)
#         :return: intensity_feat (B, out_dim, N, N), depth_feat (B, out_dim, N, N)
#         """
#         # 补齐通道维度
#         if inputs.dim() == 4:
#             inputs = inputs.unsqueeze(1) # (B, 1, M, N, N)

#         # A. 提取 3D 时空特征
#         feat3d = self.feature_3d(inputs) # (B, 64, M, N, N)
        
#         # B. 投影到 2D 空间 (大幅压缩时间维度)
#         feat2d = self.project_spatial(feat3d) # (B, 128, M//8, N, N)
        
#         # C. 处理残余的时间维度：使用自适应平均池化将其严格压平为 1
#         # 这使得网络能够接受任意大小的 M (例如 M=512 或 M=1024 都可以直接跑)
#         if feat2d.shape[2] > 1:
#             feat2d = F.adaptive_avg_pool3d(feat2d, (1, feat2d.shape[3], feat2d.shape[4]))
        
#         # 挤压掉时间维度：(B, 128, 1, N, N) -> (B, 128, N, N)
#         feat2d = feat2d.squeeze(2) 
        
#         # D. 双分支输出
#         feat_intensity = self.intensity_head(feat2d) # (B, out_dim, N, N)
#         feat_depth = self.depth_head(feat2d)         # (B, out_dim, N, N)
        
#         return feat_intensity, feat_depth


######### Version 2

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResnetBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.GroupNorm(min(32, in_channels), in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(min(32, out_channels), out_channels),
#             nn.SiLU(),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         )
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.block(x) + self.shortcut(x)

# class LOSEncoder(nn.Module):
#     def __init__(self, in_channels=1, out_dim=16):
#         """
#         单光子成像特征提取网络 (改进版)
#         :param in_channels: 输入通道数，通常单光子数据为 1
#         :param out_dim: 输出特征图的通道数
#         """
#         super().__init__()

#         # 1. 改进的 3D 特征提取：在时间维度(M)进行阶梯式降采样，扩大时间感受野
#         # 空间维度(N,N)保持不变。kernel_size=(时间, 空间, 空间)
#         self.feature_3d = nn.Sequential(
#             nn.Conv3d(in_channels, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
#             nn.BatchNorm3d(16),
#             nn.SiLU(),
            
#             # stride=(2, 1, 1) 仅压缩时间维度，保留空间分辨率
#             nn.Conv3d(16, 32, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
#             nn.BatchNorm3d(32),
#             nn.SiLU(),
            
#             nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
#             nn.BatchNorm3d(64),
#             nn.SiLU(),
            
#             nn.Conv3d(64, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
#             nn.BatchNorm3d(128),
#             nn.SiLU()
#         )

#         # 2. 核心改进：时间维度注意力层 (Temporal Attention / Learned Soft-Argmax)
#         # 用来替代原来的 adaptive_avg_pool3d，它能自适应寻找峰值并保留深度信息
#         self.temporal_attention = nn.Conv3d(128, 128, kernel_size=1)

#         # 3. 强度特征提取分支
#         self.intensity_head = nn.Sequential(
#             ResnetBlock2D(128, 128),
#             ResnetBlock2D(128, 64),
#             nn.GroupNorm(16, 64),
#             nn.SiLU(),
#             nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
#         )
        
#         # 4. 深度特征提取分支
#         self.depth_head = nn.Sequential(
#             ResnetBlock2D(128, 128),
#             ResnetBlock2D(128, 64),
#             nn.GroupNorm(16, 64),
#             nn.SiLU(),
#             nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
#         )

#     def forward(self, inputs):
#         """
#         :param inputs: 形状为 (B, M, N, N) 或 (B, 1, M, N, N)
#         :return: intensity_feat (B, out_dim, N, N), depth_feat (B, out_dim, N, N)
#         """
#         # 补齐通道维度
#         if inputs.dim() == 4:
#             inputs = inputs.unsqueeze(1) # (B, 1, M, N, N)

#         # A. 提取 3D 时空特征 (同时时间维度M会被压缩为 M/8)
#         feat3d = self.feature_3d(inputs) # (B, 128, M', N, N)
        
#         # B. 核心步骤：通过注意力机制（软求导）融合时间维度，而不是简单的平均池化
#         # 在时间维度(dim=2)上计算 Softmax 权重
#         attn_weights = F.softmax(self.temporal_attention(feat3d), dim=2)
        
#         # 使用注意力权重对时间维度进行加权求和，将 3D 特征无损压缩为 2D 特征
#         # (B, 128, M', N, N) -> (B, 128, N, N)
#         feat2d = torch.sum(feat3d * attn_weights, dim=2) 
        
#         # C. 双分支输出 2D Latent 特征
#         feat_intensity = self.intensity_head(feat2d) # (B, out_dim, N, N)
#         feat_depth = self.depth_head(feat2d)         # (B, out_dim, N, N)
        
#         return feat_intensity, feat_depth


##### version 3.0
import torch
import torch.nn as nn
import torch.nn.functional as F

class WienerFilter3D(nn.Module):
    """
    自适应 3D 维纳滤波层
    原理: y = μ + ( (σ^2 - ν^2) / σ^2 ) * (x - μ)
    其中 μ 是局部均值，σ^2 是局部方差，ν^2 是噪声方差（可学习或设定）
    """
    def __init__(self, channels, kernel_size=(5, 3, 3)):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
        # 可学习的噪声估计参数，初始化为一个较小的值
        self.noise_var = nn.Parameter(torch.tensor([0.01] * channels).view(1, channels, 1, 1, 1))

    def forward(self, x):
        # 计算局部均值 mu
        mu = F.avg_pool3d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        
        # 计算局部平方的均值
        mu_sq = F.avg_pool3d(x**2, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        
        # 计算局部方差 sigma^2
        sigma_sq = mu_sq - mu**2
        sigma_sq = torch.clamp(sigma_sq, min=1e-6) # 防止除以0
        
        # 维纳滤波公式
        # 如果噪声方差比信号方差大，缩放系数趋近于0（输出均值）；反之趋近于1（保留原信号）
        scaling_factor = torch.clamp((sigma_sq - self.noise_var) / sigma_sq, min=0)
        output = mu + scaling_factor * (x - mu)
        return output


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class LOSEncoder(nn.Module):
    def __init__(self, in_channels=1, out_dim=16):
        super().__init__()

        # 1. 初始 3D 输入层：将极稀疏的数据映射到高维特征空间
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=[5,3,3], padding=[2,1,1],bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU()
        )

        # 2. 核心改进：3D 维纳滤波层
        # 在特征提取初期进行自适应去噪，稳定稀疏信号
        self.wiener_filter = WienerFilter3D(channels=16)

        # 3. 骨干 3D 特征提取：进行时间维度降采样
        self.feature_3d = nn.Sequential(
            # (B, 16, M, N, N) -> (B, 32, M/2, N, N)
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            
            # (B, 32, M/2, N, N) -> (B, 64, M/4, N, N)
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            
            # (B, 64, M/4, N, N) -> (B, 128, M/8, N, N)
            nn.Conv3d(64, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(128),
            nn.SiLU()
        )

        # 4. 时间维度注意力
        self.temporal_attention = nn.Conv3d(128, 128, kernel_size=1)
        
        # 3. 强度特征提取分支 (保持分辨率为 N*N)
        self.intensity_head = nn.Sequential(
            ResnetBlock2D(128, 128),
            ResnetBlock2D(128, 64),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
        )
        
        # 4. 深度特征提取分支 (保持分辨率为 N*N)
        self.depth_head = nn.Sequential(
            ResnetBlock2D(128, 128),
            ResnetBlock2D(128, 64),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1) # (B, 1, M, N, N)

        # 前向传播
        x = self.stem(inputs)
        x = self.wiener_filter(x)           # 执行维纳去噪
        feat3d = self.feature_3d(x)
        
        # 时间维度加权融合
        attn_weights = F.softmax(self.temporal_attention(feat3d), dim=2)
        feat2d = torch.sum(feat3d * attn_weights, dim=2)
        
        # 分支输出
        feat_intensity = self.intensity_head(feat2d)
        feat_depth = self.depth_head(feat2d)
        
        return feat_intensity, feat_depth

# ### version 4.0: 使用肖忠培版本的维纳滤波，效果比较糟糕
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResnetBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.GroupNorm(min(32, in_channels), in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(min(32, out_channels), out_channels),
#             nn.SiLU(),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         )
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.block(x) + self.shortcut(x)


# class PhotonCore(nn.Module):
#     def __init__(self, channels=16, init_sigma_xy=2.0, init_sigma_z=2.0, init_snr=0.1):
#         """
#         基于 3D 维纳滤波的可学习物理先验层 (多通道独立 SNR 版)
#         :param channels: 特征通道数，确保每个通道有独立的 SNR 学习能力
#         """
#         super().__init__()
#         # 空间和时间维度的模糊核参数 (通常所有通道共享同一套光学系统的 PSF 先验)
#         self.sigma_xy = nn.Parameter(torch.tensor(init_sigma_xy, dtype=torch.float32))
#         self.sigma_z  = nn.Parameter(torch.tensor(init_sigma_z, dtype=torch.float32))
        
#         # 核心修改：将 snr 设为长度为 channels 的可学习向量
#         self.snr = nn.Parameter(torch.ones(channels, dtype=torch.float32) * init_snr)

#     def forward(self, x):
#         """
#         :param x: 输入张量，shape = [B, C, D, H, W]
#         """
#         B, C, D, H, W = x.shape
#         dev = x.device
        
#         # 1. 动态生成网格坐标
#         xs = torch.linspace(-(W-1)/2, (W-1)/2, steps=W, device=dev)
#         ys = torch.linspace(-(H-1)/2, (H-1)/2, steps=H, device=dev)
#         zs = torch.linspace(-(D-1)/2, (D-1)/2, steps=D, device=dev)
#         grid_y, grid_x, grid_z = torch.meshgrid(ys, xs, zs, indexing='ij')

#         # 保证参数恒正
#         sigma_xy = F.softplus(self.sigma_xy) + 1e-3  
#         sigma_z  = F.softplus(self.sigma_z) + 1e-3  
        
#         # shape 转换: [C] -> [1, C, 1, 1, 1] 以匹配 [B, C, D, H, W] 进行广播
#         snr_val  = F.softplus(self.snr) + 1e-6
#         snr_val  = snr_val.view(1, C, 1, 1, 1)

#         # 2. 构造 3D 高斯 PSF
#         gauss_xy = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_xy**2))
#         gauss_z  = torch.exp(-(grid_z**2) / (2 * sigma_z**2))
#         psf = (gauss_xy * gauss_z).permute(2, 0, 1)  # [D, H, W]
#         psf = psf / (psf.sum() + 1e-8)               # 能量归一化

#         # 3. 计算 PSF 的 FFT
#         H_f = torch.fft.fftn(psf, dim=(0, 1, 2))     # [D, H, W]
#         H_f = H_f.unsqueeze(0).unsqueeze(0)          # 扩展为 [1, 1, D, H, W]

#         # 4. 构造多通道维纳滤波器
#         H_conj = torch.conj(H_f)
#         H_abs2 = H_f.real**2 + H_f.imag**2
        
#         # 由于 snr_val 是 [1, C, 1, 1, 1]，W_filter 会自动广播成 [1, C, D, H, W]
#         # 这样每个通道就拥有了独立强度的频域滤波器
#         W_filter = H_conj / (H_abs2 + 1.0 / snr_val) 

#         # 5. 数据的 FFT -> 频域滤波 -> IFFT
#         Xf = torch.fft.fftn(x, dim=(2, 3, 4))        # [B, C, D, H, W]
#         Yf = Xf * W_filter                           # [B, C, D, H, W]
#         re = torch.fft.ifftn(Yf, dim=(2, 3, 4)).real # [B, C, D, H, W]

#         # 6. 逐样本、逐通道的归一化
#         min_val = re.amin(dim=(2, 3, 4), keepdim=True)
#         max_val = re.amax(dim=(2, 3, 4), keepdim=True)
#         output = (re - min_val) / (max_val - min_val + 1e-8)
        
#         return output


# class LOSEncoder(nn.Module):
#     def __init__(self, in_channels=1, out_dim=16):
#         super().__init__()

#         # 1. 第一层输入 3D 卷积 (保持空间和时间分辨率)
#         self.conv_in = nn.Sequential(
#             nn.Conv3d(in_channels, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1),bias=False),
#             nn.BatchNorm3d(16),
#             nn.SiLU()
#         )
        
#         # 2. 核心插入点：加入可学习的 3D 维纳滤波
#         self.wiener_filter = PhotonCore(init_sigma_xy=2.0, init_sigma_z=2.0, init_snr=1.0)

#         # 3. 剩余的 3D 特征提取：阶梯式降采样时间维度(M)
#         self.feature_3d_rest = nn.Sequential(
#             nn.Conv3d(16, 32, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
#             nn.BatchNorm3d(32),
#             nn.SiLU(),
            
#             nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
#             nn.BatchNorm3d(64),
#             nn.SiLU(),
            
#             nn.Conv3d(64, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
#             nn.BatchNorm3d(128),
#             nn.SiLU()
#         )

#         # 4. 时间维度注意力层
#         self.temporal_attention = nn.Conv3d(128, 128, kernel_size=1)

#         # 5. 强度特征提取分支
#         self.intensity_head = nn.Sequential(
#             ResnetBlock2D(128, 128),
#             ResnetBlock2D(128, 64),
#             nn.GroupNorm(16, 64),
#             nn.SiLU(),
#             nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
#         )
        
#         # 6. 深度特征提取分支
#         self.depth_head = nn.Sequential(
#             ResnetBlock2D(128, 128),
#             ResnetBlock2D(128, 64),
#             nn.GroupNorm(16, 64),
#             nn.SiLU(),
#             nn.Conv2d(64, out_dim, kernel_size=3, padding=1)
#         )

#     def forward(self, inputs):
#         if inputs.dim() == 4:
#             inputs = inputs.unsqueeze(1) # (B, 1, M, N, N)

#         # A. 初始特征提取
#         x = self.conv_in(inputs)             # (B, 16, M, N, N)
        
#         # B. 物理先验滤波 (增强稀疏信号，抑制泊松/背景噪声)
#         x = self.wiener_filter(x)            # (B, 16, M, N, N)
        
#         # C. 后续下采样特征提取
#         feat3d = self.feature_3d_rest(x)     # (B, 128, M', N, N)
        
#         # D. 注意力融合 (3D 转 2D)
#         attn_weights = F.softmax(self.temporal_attention(feat3d), dim=2)
#         feat2d = torch.sum(feat3d * attn_weights, dim=2) 
        
#         # E. 双分支输出
#         feat_intensity = self.intensity_head(feat2d) 
#         feat_depth = self.depth_head(feat2d)         
        
#         return feat_intensity, feat_depth


# ================= 测试代码 =================
if __name__ == "__main__":
    # 模拟输入参数: Batch=2, TimeBins(M)=512, Spatial(N*N)=128*128
    B, M, N = 2, 512, 128
    
    # 构建模拟的单光子 3D 测量结果
    dummy_input = torch.rand(B, 1, M, N, N)
    
    # 实例化网络 (设定输出特征通道数为 16)
    model = LOSEncoder(in_channels=1, out_dim=16)
    
    # 前向传播
    intensity_features, depth_features = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"强度特征输出形状: {intensity_features.shape}") # [B, out_dim, N, N]
    print(f"深度特征输出形状: {depth_features.shape}") # [B, out_dim, N, N]