# Model.py
import scipy.io
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import time
import numpy as np

def save_output(output):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    torch.save(output, f"nlos_output_{timestamp}.pt")


def definePsf_Batchtorch(sigma,sptial_grid, temprol_grid, slope,device='cuda'):
    # slop is time_range / wall_size
    dev=device

    x_2N = torch.linspace(-1, 1, steps=2 * sptial_grid, dtype=torch.float32).to(dev)
    y_2N = x_2N.clone()
    z_2M = torch.linspace(0, 2, steps=2 * temprol_grid, dtype=torch.float32).to(dev)
    # grid axis, also in hxwxt
    # that's why x is the second axis
    # y is the first axis
    # [gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M] = np.meshgrid(x_2N, y_2N, z_2M)
    gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M = torch.meshgrid(y_2N, x_2N, z_2M, indexing="ij")

    # dst
    a_2Nx2NX2M = (4 * slope) ** 2 * (gridx_2Nx2Nx2M ** 2 + gridy_2Nx2Nx2M ** 2) - gridz_2Nx2Nx2M
    b_2Nx2NX2M = torch.abs(a_2Nx2NX2M)

    # should be a ellipse
    c_2Nx2NX2M = torch.min(b_2Nx2NX2M, dim=2, keepdim=True)[0]  # min along z-axis

    k=1000
    d_2Nx2NX2M = torch.sigmoid(-k * (b_2Nx2NX2M - c_2Nx2NX2M))
    # d_2Nx2NX2M = d_2Nx2NX2M.to(torch.float32)  # Convert to float32

    d_2Nx2NX2M = d_2Nx2NX2M.to(torch.float32)

    # norm
    e_2Nx2NX2M = d_2Nx2NX2M / torch.sqrt(torch.sum(d_2Nx2NX2M))

    # shift
    f1_2Nx2NX2M = torch.roll(e_2Nx2NX2M, shifts=sptial_grid, dims=0)
    f2_2Nx2NX2M = torch.roll(f1_2Nx2NX2M, shifts=sptial_grid, dims=1)

    # psf_2Mx2Nx2N = np.transpose(f2_2Nx2NX2M, [2, 0, 1])
    psf_2Mx2Nx2N = f2_2Nx2NX2M.permute(2, 0, 1) 
    psf_2Mx2Nx2N = psf_2Mx2Nx2N.unsqueeze(0).unsqueeze(0)  # → [1, 1, T, H, W]
    return psf_2Mx2Nx2N


def resamplingOperator_Batchtorch(temprol_grid,device='cuda'):
    dev=device
    M = temprol_grid
    row = M ** 2
    col = M
    assert 2 ** int(np.log2(M)) == M

    x = torch.arange(1, row + 1, dtype=torch.float32,device=dev)  # 从 1 到 M^2

    rowidx = torch.arange(row,device=dev)
    # 0 to M-1
    colidx = torch.ceil(torch.sqrt(x)) - 1
    data = torch.ones_like(rowidx, dtype=torch.float32,device=dev)
    mtx1 = ssp.csr_matrix((data.cpu().numpy(), (rowidx.cpu().numpy(), colidx.cpu().numpy())), shape=(row, col), dtype=np.float32)
    mtx2 = ssp.spdiags(data=[1.0 / np.sqrt(x.cpu().numpy())], diags=[0], m=row, n=row)

    mtx = mtx2.dot(mtx1)
    mtx = torch.tensor(mtx.toarray(), dtype=torch.float32).to(dev)
    K = int(torch.log2(torch.tensor(M)))
    for _ in range(K):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2])

    mtxi = mtx.T
    return mtx,mtxi


# 基于LCT实现域转换
class NlosCore(nn.Module):
    def __init__(self, spatial=64, crop=512, bin_resolution=33e-12, wall_size=2.0, snr=0.8,sigma=0.02):
        super(NlosCore, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop

        assert 2 ** int(torch.log2(torch.tensor(crop))) == crop

        self.has_saved = False
        self.wall_size=wall_size
        self.bin_resolution = bin_resolution
        self.snr=snr
        self.sigma=sigma

    def forward(self, rect_data):
        """
        rect_data: [B, D, T, H, W]
        """
        decay=4
        device=rect_data.device
        c = 3e8
        half_width = self.wall_size / 2.0
        bin_resolution = self.bin_resolution
        crop=self.crop
        # assert 2 ** int(torch.log2(crop)) == crop

        ###########################################
        rect_data = rect_data[:,:,:crop, :, :]  # [B, D, T, H, W]
        B, D,  T, H, W = rect_data.shape
        trange = T * c * bin_resolution

        # 计算维纳滤波核
        gridz_TxHxW= torch.linspace(0, 1, steps=T).to(device).view(1, 1, T,1, 1)
        slope = half_width / trange
        psf = definePsf_Batchtorch(self.sigma, H, T, slope, device=device)
        fpsf = torch.fft.fftn(psf) # [1,1,T,W,H],后三维傅里叶变换和五维一起傅里叶变换完全一样
        invpsf = torch.conj(fpsf) / (1 / self.snr + torch.real(fpsf) ** 2 + torch.imag(fpsf) ** 2)
        mtx_MxM, mtxi_MxM = resamplingOperator_Batchtorch(T,device=device)

        # mtx重采样
        data_TxHxW = rect_data * (gridz_TxHxW ** decay)
        datapad_2Tx2Hx2W = torch.zeros((B,D,2 * T, 2 * H, 2 * H), device=device)
        left = mtx_MxM.unsqueeze(0).expand(B * D, -1, -1).float()
        right = data_TxHxW.reshape(B * D, T, H * W).float()  # (BD, T, HW)
        tmp = torch.bmm(left, right).view(B, D, T, H, W)
        datapad_2Tx2Hx2W[:,:,:T, :H, :H] = tmp

        # 傅里叶变换
        datafre = torch.fft.fftn(datapad_2Tx2Hx2W,dim=(-3,-2,-1)) # [B,D,T,W,H]，只有后三维傅里叶变换
        # 维纳滤波
        filtered_datafre = datafre * invpsf
        # 傅里叶逆变换
        volumn_2Mx2Nx2N = torch.fft.ifftn(filtered_datafre,dim=(-3,-2,-1)) # [B,D,T,W,H]，只有后三维傅里叶逆变换

        volumn_2Mx2Nx2N = torch.real(volumn_2Mx2Nx2N)
        volumn_ZxYxX = volumn_2Mx2Nx2N[:,:,:T, :H, :H]

        # 逆mtx重采样
        left = mtxi_MxM.unsqueeze(0).expand(B * D, -1, -1).float()
        right = volumn_ZxYxX.reshape(B * D, T, H * W).float()
        volumn_ZxYxX = torch.bmm(left, right).view(B, D, T, H, W)
        volumn_ZxYxX = F.relu(volumn_ZxYxX)

        min_val = volumn_ZxYxX.min()
        max_val = volumn_ZxYxX.max()
        volumn_ZxYxX = (volumn_ZxYxX - min_val) / (max_val - min_val + 1e-8)

        return volumn_ZxYxX

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

# class Downsample(nn.Module):
#     def __init__(self, in_channels: int):
#         super().__init__()
#         # no asymmetric padding in torch conv, must do it ourselves
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

#     def forward(self, x):
#         pad = (0, 1, 0, 1)
#         x = nn.functional.pad(x, pad, mode="constant", value=0)
#         x = self.conv(x)
#         return x
    

# class NlosEncoder(nn.Module):
#     def __init__(self, ch_in=1, spatial=128, crop=512, bin_resolution=33e-12, wall_size=2.0):
#         super().__init__()

#         self.in_chans = ch_in
#         self.spatial = spatial
#         self.crop = crop
#         self.bin_resolution = bin_resolution

#         # 1. 初始三维卷积提取特征：保持最后三个维度 (H, W, T) 大小不变
#         # 输入: (B, 1, H, W, T) -> 输出: (B, 16, H, W, T)
#         self.input_layer = nn.Sequential(
#             nn.Conv3d(ch_in, 16, kernel_size=3, padding=1),
#             nn.BatchNorm3d(16),
#             nn.SiLU()
#         )

#         # 2. 域转换核心模块：将 (B, D, H, W, T) 映射到 3D 体素空间 (B, D, T, H, W)
#         self.tra2vol = NlosCore(spatial=self.spatial, crop=self.crop, 
#                                 bin_resolution=bin_resolution, wall_size=wall_size)
        
#         # 3. 三维特征进一步提取
#         # 此时输入维度可能是 (B, 16, T, H, W)
#         self.feature_layer = nn.Sequential(
#             nn.Conv3d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm3d(32),
#             nn.SiLU(),
#             nn.Conv3d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm3d(64),
#             nn.SiLU()
#         )
        
#         # 4. 3D 到 2D 的投影映射 (避免使用简单的 max-pooling)
#         # 通过步长卷积逐步压缩深度(T)维度，同时提取空间特征
#         # 目标是将 (B, 64, T, H, W) 映射为 (B, 128, H, W)
#         self.project_spatial = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=(8, 1, 1), stride=(8, 1, 1)), # 压缩 T
#             nn.SiLU()
#         )

#         # 5. 专门提取强度(RGB)和深度特征的 Head (模仿 JointDiT Encoder 结构)
#         # 目标输出分辨率: (H//8, W//8)，通道: 16
#         # 需要 3 次降采样 (8 = 2^3)
#         self.rgb_feature = nn.Sequential(
#             ResnetBlock2D(128, 256),
#             # Downsample(256),      # 1/2
#             ResnetBlock2D(256, 512),
#             # Downsample(512),      # 1/4
#             ResnetBlock2D(512, 512),
#             # Downsample(512),      # 1/8
#             nn.GroupNorm(32, 512),
#             nn.Conv2d(512, 16, kernel_size=3, padding=1)
#         )
        
#         self.depth_feature = nn.Sequential(
#             ResnetBlock2D(128, 256),
#             # Downsample(256),      # 1/2
#             ResnetBlock2D(256, 512),
#             # Downsample(512),      # 1/4
#             ResnetBlock2D(512, 512),
#             # Downsample(512),      # 1/8
#             nn.GroupNorm(32, 512),
#             nn.Conv2d(512, 16, kernel_size=3, padding=1)
#         )

#     def forward(self, inputs):
#         # inputs shape: (B, D, T, H, W)
#         if inputs.dim() == 4:
#             inputs = inputs.unsqueeze(1) # 补上通道维度 D

#         # A. 初始 3D 特征提取
#         x = self.input_layer(inputs)  # (B, 16, T, H, W)
        
#         # B. 通过 LCT 物理层进行域转换
#         # tra2vol 内部处理维度转换，输出 (B, 16, T, H, W)
#         vol = self.tra2vol(x)
        
#         # C. 深度 3D 特征提取
#         feat3d = self.feature_layer(vol)
        
#         # D. 空间投影：将 3D 体素压缩为 2D 基础特征图
#         # 假设 T 被压缩为 1，然后 squeeze 掉
#         feat2d = self.project_spatial(feat3d)
        
#         # 如果 T 没被压缩干净，可以使用自适应池化辅助
#         if feat2d.shape[2] > 1:
#             feat2d = F.adaptive_avg_pool3d(feat2d, (1, feat2d.shape[3], feat2d.shape[4]))
#         feat2d = feat2d.squeeze(2) # (B, 128, H, W)
        
#         # E. 分支处理：生成 RGB 特征和深度特征
#         feature_rgb = self.rgb_feature(feat2d)     # (B, 16, H, W)
#         feature_depth = self.depth_feature(feat2d) # (B, 16, H, W)
        
#         return feature_rgb, feature_depth


# ==========================================
# 引入 Flux 风格的组件 (2D & 3D)
# ==========================================
from einops import rearrange
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# 仿 Flux 2D AttnBlock
class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


class ResConv3D(nn.Module):
    """3D Residual Convolution Block."""
    def __init__(self, channels, inplace=False):
        super(ResConv3D, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=[3, 3, 3], 
                      padding=0, stride=[1, 1, 1], bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=[3, 3, 3], 
                      padding=0, stride=[1, 1, 1], bias=True),
        )
        self.inplace = inplace

    def forward(self, x):
        return F.leaky_relu(self.conv(x) + x, negative_slope=0.2, inplace=self.inplace)


class TransientStem(nn.Module):
    """Modified Transient2volumn without downsampling.
    Maps [B, 1, T, H, W] -> [B, 16, T, H, W]
    """
    def __init__(self, in_channels=1, out_channels=16):
        super(TransientStem, self).__init__()
        
        assert in_channels == 1, "Input channels must be 1 for the physical prior weights."
        
        # 1. 物理先验分支的权重初始化 (保持不变)
        weights = np.zeros((1, 1, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        self.weights = nn.Parameter(tfweights) # 默认 requires_grad=True
        
        # 2. 计算特征分支需要的通道数：总通道 16 - 先验通道 1 = 15
        conv_channels = out_channels - 1 
        
        # 3. 特征提取分支 (将 stride 改为 1 以保持分辨率)
        self.conv1 = nn.Sequential(
            nn.ReplicationPad3d(1),
            # 修改 stride=[1, 1, 1] 以取消降采样
            nn.Conv3d(in_channels, conv_channels, kernel_size=[3, 3, 3], 
                      padding=0, stride=[1, 1, 1], bias=True),
            ResConv3D(conv_channels, inplace=False),
            ResConv3D(conv_channels, inplace=False)
        )
    
    def forward(self, x0):
        # Path 1: 物理先验卷积，修改 stride=1 保持尺寸不变
        # 输出尺寸: [B, 1, T, H, W]
        x0_conv = F.conv3d(x0, self.weights, bias=None, stride=1, # 这个bias设置为None是很重要的，不然会改变初始输入
                           padding=1, dilation=1, groups=1)
        
        # Path 2: 深度特征提取
        # 输出尺寸: [B, 15, T, H, W]
        x1 = self.conv1(x0)
        
        # 拼接维度 1 (通道维度): 1 + 15 = 16
        # 输出尺寸: [B, 16, T, H, W]
        re = torch.cat([x0_conv, x1], dim=1)
        return re

# ==========================================
# 重构后的 NlosEncoder
# ==========================================
class NlosEncoder(nn.Module):
    def __init__(self, ch_in=1, spatial=128, crop=512, bin_resolution=33e-12, wall_size=2.0):
        super().__init__()

        self.in_chans = ch_in
        self.spatial = spatial
        self.crop = crop
        self.bin_resolution = bin_resolution

        # 1. 初始三维卷积：(B, 1, T, H, W) -> (B, 16, T, H, W)
        # self.input_layer = nn.Sequential(
        #     # Stem: 先用基础卷积将物理空间的 raw input 平滑投影到特征空间
        #     nn.Conv3d(ch_in, 16, kernel_size=3, padding=1),
        #     # ResBlock: 紧跟残差块，完美契合你的思路，增强初始特征表达
        #     ResnetBlock3D(16, 16),
        #     ResnetBlock3D(16, 16)
        # )
        self.input_layer = TransientStem(in_channels=ch_in, out_channels=16)

        # 2. 域转换核心模块：将 (B, D, H, W, T) 映射到 3D 体素空间 (B, D, T, H, W)
        self.tra2vol = NlosCore(spatial=self.spatial, crop=self.crop, 
                                bin_resolution=bin_resolution, wall_size=wall_size)
        
        # 3. 三维特征进一步提取
        # 此时输入维度可能是 (B, 16, T, H, W)
        self.feature_layer = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU()
        )
        
        # 4. 3D 到 2D 的投影映射 (避免使用简单的 max-pooling)
        # 通过步长卷积逐步压缩深度(T)维度，同时提取空间特征
        # 目标是将 (B, 64, T, H, W) 映射为 (B, 128, H, W)
        self.project_spatial = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(8, 1, 1), stride=(8, 1, 1)), # 压缩 T
            nn.SiLU()
        )

        # 5. 专门提取强度(RGB)和深度特征的 Head (模仿 JointDiT Encoder 结构)
        # 目标输出分辨率: (H//8, W//8)，通道: 16
        # 需要 3 次降采样 (8 = 2^3)
        self.rgb_feature = nn.Sequential(
            ResnetBlock2D(128, 256),
            # Downsample(256),      # 1/2
            ResnetBlock2D(256, 512),
            # Downsample(512),      # 1/4
            ResnetBlock2D(512, 512),
            # Downsample(512),      # 1/8
            nn.GroupNorm(32, 512),
            nn.Conv2d(512, 16, kernel_size=3, padding=1)
        )
        
        self.depth_feature = nn.Sequential(
            ResnetBlock2D(128, 256),
            # Downsample(256),      # 1/2
            ResnetBlock2D(256, 512),
            # Downsample(512),      # 1/4
            ResnetBlock2D(512, 512),
            # Downsample(512),      # 1/8
            nn.GroupNorm(32, 512),
            nn.Conv2d(512, 16, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        # inputs shape: (B, D, T, H, W)
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1) # 补上通道维度 D

        # A. 初始 3D 特征提取
        x = self.input_layer(inputs)  # (B, 16, T, H, W)
        
        # B. 通过 LCT 物理层进行域转换
        # tra2vol 内部处理维度转换，输出 (B, 16, T, H, W)
        vol = self.tra2vol(x)
        
        # C. 深度 3D 特征提取
        feat3d = self.feature_layer(vol)
        
        # D. 空间投影：将 3D 体素压缩为 2D 基础特征图
        # 假设 T 被压缩为 1，然后 squeeze 掉
        feat2d = self.project_spatial(feat3d)
        
        # 如果 T 没被压缩干净，可以使用自适应池化辅助
        if feat2d.shape[2] > 1:
            feat2d = F.adaptive_avg_pool3d(feat2d, (1, feat2d.shape[3], feat2d.shape[4]))
        feat2d = feat2d.squeeze(2) # (B, 128, H, W)
        
        # E. 分支处理：生成 RGB 特征和深度特征
        feature_rgb = self.rgb_feature(feat2d)     # (B, 16, H, W)
        feature_depth = self.depth_feature(feat2d) # (B, 16, H, W)
        
        return feature_rgb, feature_depth




if __name__ == "__main__":
    # 简单测试
    B, D, H, W, T = 1, 1, 128, 128, 512
    device = "cuda:1"
    
    # 1. 清空缓存并重置显存峰值统计
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats(device=device)

    dummy_input = torch.randn(B, D, T, H, W).to(device)
    model = NlosEncoder().to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # 2. 运行前向传播
    feat_rgb, feat_depth = model(dummy_input)
    print("RGB Feature Shape:", feat_rgb.shape)       # 期望: (B, 16, H, W)
    print("Depth Feature Shape:", feat_depth.shape)   # 期望: (B, 16, H, W)
    
    # 3. 获取并打印最大显存占用
    # Allocated: 张量实际占用的显存
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    # Reserved: PyTorch 缓存分配器保留的显存总量（通常在 nvidia-smi 中看到的更接近这个值）
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    
    print("-" * 30)
    print(f"Max Memory Allocated: {max_memory_allocated:.2f} GB")
    print(f"Max Memory Reserved:  {max_memory_reserved:.2f} GB")
    print("-" * 30)