
import torch
from torch import nn
from .extractor import UnetExtractor, ResidualBlock
from einops import rearrange


class GaussianNetwork(nn.Module):
    def __init__(self, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = [64, 64, 128]
        self.depth_dims = [32, 48, 96]
        self.decoder_dims = [48, 64, 96]
        self.head_dim = 32

        self.sh_degree = 4
        self.d_sh = (self.sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0]+rgb_dim+1, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.sh_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3 * self.d_sh, kernel_size=1),
        )

    def forward(self, img, depth, img_feat):
        # img_feat1: [4, 64, 176, 320]
        # img_feat2: [4, 64, 88, 160]
        # img_feat3: [4, 128, 44, 80]
        img_feat1, img_feat2, img_feat3 = img_feat
        # depth_feat1: [4, 32, 176, 320]
        # depth_feat2: [4, 48, 88, 160]
        # depth_feat3: [4, 96, 44, 80]
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)

        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)

        # sh head
        sh_out = self.sh_head(out)
        # sh_out: [(b * v), C, H, W]

        sh_out = rearrange(
            sh_out, "n c h w -> n (h w) c",
        )
        sh_out = rearrange(
            sh_out,
            "... (srf c) -> ... srf () c",
            srf=1,
        )

        sh_out = rearrange(sh_out, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        # [(b * v), (H * W), 1, 1 3, 25]

        # sh_out = sh_out.broadcast_to(sh_out.shape) * self.sh_mask
        sh_out = sh_out * self.sh_mask
        # import pdb; pdb.set_trace()
        

        return rot_out, scale_out, opacity_out, sh_out


# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# from .extractor import UnetExtractor, ResidualBlock
# from einops import rearrange

# # =========================================================================
# #  核心模块: Dilated Isotropic Diffusion (DID) Module
# #  基于空洞级联扩散的几何势场计算
# # =========================================================================
# class GeometricPotentialModule(nn.Module):
#     """
#     实现基于 "Varadhan's Formula" 的可微距离变换。
    
#     技术亮点:
#     1. 全分辨率 (Full-Resolution): 不做任何下采样，保留细长物体(如杆状物)的拓扑结构。
#     2. 空洞级联 (Dilated Cascade): 利用多尺度空洞卷积模拟热扩散过程，显存消耗为线性 O(N)，而非平方 O(N^2)。
#     3. 结构张量 (Structure Tensor): 自动过滤纹理噪声，只对真实的几何边缘产生反应。
#     """
#     def __init__(self, beta=0.3):
#         super().__init__()
#         self.beta = beta  # 控制势场衰减速度 (Temperature parameter)
        
#         # 1. 结构张量核 (局部 3x3)
#         self.register_buffer('struct_kernel', self._make_gaussian_kernel(3, sigma=1.0))

#         # 2. 扩散核 (3x3 固定高斯核)
#         # 用于级联空洞卷积，模拟物理热传导
#         self.register_buffer('diff_kernel', self._make_gaussian_kernel(3, sigma=1.0))

#     def _make_gaussian_kernel(self, k_size, sigma):
#         x = torch.arange(k_size) - k_size // 2
#         k1d = torch.exp(-x**2 / (2 * sigma**2))
#         k2d = k1d.view(-1, 1) * k1d.view(1, -1)
#         k2d = k2d / k2d.sum()
#         return k2d.view(1, 1, k_size, k_size)

#     def compute_edge_source(self, depth):
#         """全分辨率计算几何边缘源，利用结构张量过滤噪声"""
#         dtype = depth.dtype
#         device = depth.device
        
#         # 3x3 Sobel (Tiny memory footprint)
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1,1,3,3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1,1,3,3)
        
#         Ix = F.conv2d(depth, sobel_x, padding=1)
#         Iy = F.conv2d(depth, sobel_y, padding=1)
        
#         # 结构张量平滑
#         Ixx, Iyy, Ixy = Ix*Ix, Iy*Iy, Ix*Iy
#         G = self.struct_kernel.type_as(depth)
#         Sxx = F.conv2d(Ixx, G, padding=1)
#         Syy = F.conv2d(Iyy, G, padding=1)
#         Sxy = F.conv2d(Ixy, G, padding=1)
        
#         # 几何相干性 (Coherence)
#         trace = Sxx + Syy
#         det = Sxx*Syy - Sxy*Sxy
#         delta = torch.sqrt((trace**2 - 4*det).clamp(min=1e-8))
#         l1 = (trace + delta) / 2
#         l2 = (trace - delta) / 2
#         coherence = (l1 - l2) / (l1 + l2 + 1e-6)
        
#         grad_mag = torch.sqrt(Ix**2 + Iy**2 + 1e-8)
#         # 生成边缘源: 1=Clean Edge, 0=Noise/Flat
#         edge_source = coherence * torch.sigmoid(grad_mag * 5.0 - 2.0)
#         return edge_source, coherence

#     def forward(self, scale_raw, depth, intrinsics):
#         # Step 1: 物理限制计算 (完全不保留梯度，节省巨大显存)
#         with torch.no_grad():
#             # 获取边缘源 (Full Resolution)
#             edge_source, coherence = self.compute_edge_source(depth)
            
#             # 多尺度空洞扩散 (Multi-Scale Dilated Diffusion)
#             # 模拟热传导: U_total = sum( w_i * Conv_dilated(Source) )
#             K = self.diff_kernel.type_as(depth)
            
#             # Layer 1: Dilation=1 (近距离)
#             u1 = F.conv2d(edge_source, K, padding=1, dilation=1)
#             # Layer 2: Dilation=2 (中距离)
#             u2 = F.conv2d(u1, K, padding=2, dilation=2)
#             # Layer 3: Dilation=4 (远距离)
#             u3 = F.conv2d(u2, K, padding=4, dilation=4)
#             # Layer 4: Dilation=8 (超远距离，适配高分图)
#             u4 = F.conv2d(u3, K, padding=8, dilation=8)
            
#             # 加权融合不同尺度的势能 (近处权重高，远处权重低)
#             potential_field = 0.4 * u1 + 0.3 * u2 + 0.2 * u3 + 0.1 * u4
#             potential_field = potential_field.clamp(min=1e-6, max=1.0)
            
#             # 物理距离反解 (Varadhan's Inverse): d ~ -log(U)
#             pixel_dist = - (1.0 / self.beta) * torch.log(potential_field)
            
#             # 物理单位转换
#             focal = intrinsics[:, 0, 0].view(-1, 1, 1, 1)
#             metric_per_pixel = depth / (focal + 1e-5)
            
#             # Limit 计算 (1.2 是重叠系数)
#             limit = 1.2 * pixel_dist * metric_per_pixel
            
#             # 保底机制: 至少保留 0.8 个像素宽，防止极细物体数值消失
#             safe_limit = limit + 0.8 * metric_per_pixel
            
#         # Step 2: 软融合 (Soft Clamp)
#         # 这一步在计算图中，梯度可以回传给 scale_raw
#         scale_constrained = (scale_raw * safe_limit) / (scale_raw + safe_limit + 1e-6)
        
#         return scale_constrained, coherence


# # =========================================================================
# #  主网络: GaussianNetwork
# # =========================================================================
# class GaussianNetwork(nn.Module):
#     def __init__(self, rgb_dim=3, depth_dim=1, norm_fn='group', use_agpf=True):
#         super().__init__()
        
#         self.use_agpf = use_agpf
#         print(f"[GaussianNetwork] Initialized. AGPF Physics Constraint: {self.use_agpf}")

#         self.rgb_dims = [64, 64, 128]
#         self.depth_dims = [32, 48, 96]
#         self.decoder_dims = [48, 64, 96]
#         self.head_dim = 32

#         self.sh_degree = 4
#         self.d_sh = (self.sh_degree + 1) ** 2

#         self.register_buffer(
#             "sh_mask",
#             torch.ones((self.d_sh,), dtype=torch.float32),
#             persistent=False,
#         )
#         for degree in range(1, self.sh_degree + 1):
#             self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

#         self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

#         self.decoder3 = nn.Sequential(
#             ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
#             ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
#         )
#         self.decoder2 = nn.Sequential(
#             ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
#             ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
#         )
#         self.decoder1 = nn.Sequential(
#             ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
#             ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
#         )
#         self.up = nn.Upsample(scale_factor=2, mode="bilinear")
#         self.out_conv = nn.Conv2d(self.decoder_dims[0]+rgb_dim+1, self.head_dim, kernel_size=3, padding=1)
#         self.out_relu = nn.ReLU(inplace=True)

#         self.rot_head = nn.Sequential(
#             nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.head_dim, 4, kernel_size=1),
#         )
        
#         # =====================================================================
#         # 【关键修改】 Uncertainty-Aware Scale Head
#         # 输出通道 = 3 (Scale xyz) + 1 (Semantic Confidence Beta)
#         # =====================================================================
#         self.scale_head = nn.Sequential(
#             nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.head_dim, 4, kernel_size=1) 
#             # 移除 Softplus，我们在 forward 里手动处理
#         )
        
#         self.opacity_head = nn.Sequential(
#             nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.head_dim, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.sh_head = nn.Sequential(
#             nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.head_dim, 3 * self.d_sh, kernel_size=1),
#         )
        
#         # 初始化 AGPF 模块
#         self.agpf_module = GeometricPotentialModule(beta=0.3)

#     def forward(self, img, depth, img_feat, intrinsics=None):
#         img_feat1, img_feat2, img_feat3 = img_feat
#         depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)
#         feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
#         feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
#         feat1 = torch.concat([img_feat1, depth_feat1], dim=1)
#         up3 = self.decoder3(feat3)
#         up3 = self.up(up3)
#         up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
#         up2 = self.up(up2)
#         up1 = self.decoder1(torch.cat([up2, feat1], dim=1))
#         up1 = self.up(up1)
#         out = torch.cat([up1, img, depth], dim=1)
#         out = self.out_conv(out)
#         out = self.out_relu(out)

#         rot_out = self.rot_head(out)
#         rot_out = torch.nn.functional.normalize(rot_out, dim=1)
#         opacity_out = self.opacity_head(out)
#         sh_out = self.sh_head(out)
#         sh_out = rearrange(sh_out, "n c h w -> n (h w) c")
#         sh_out = rearrange(sh_out, "... (srf c) -> ... srf () c", srf=1)
#         sh_out = rearrange(sh_out, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
#         sh_out = sh_out * self.sh_mask

#         # =====================================================================
#         #  【核心修复】 Scale 防爆盾 (Anti-Explosion Guard)
#         # =====================================================================
#         raw_out = self.scale_head(out)
#         scale_raw_logits, beta_logits = torch.split(raw_out, [3, 1], dim=1)
        
#         # 1. 基础预测 (Softplus 保证非负)
#         scale_raw = F.softplus(scale_raw_logits, beta=100)
#         semantic_gate = torch.sigmoid(beta_logits)

#         # 2. 深度安全处理 (防止 0 深度导致除零或投影无限大)
#         # 这一步非常重要！如果有 0.01m 的噪点深度，投影会无穷大
#         depth_safe = torch.clamp(depth, min=0.5, max=200.0)

#         # 3. 计算最终 scale
#         if self.use_agpf and intrinsics is not None:
#             # Checkpoint 节省显存
#             scale_math, geo_coherence = checkpoint.checkpoint(
#                 self.agpf_module, 
#                 scale_raw, depth_safe, intrinsics, 
#                 use_reentrant=False
#             )
#             # 融合
#             final_gate = geo_coherence * semantic_gate
#             scale_out = final_gate * scale_math + (1 - final_gate) * scale_raw
#         else:
#             scale_out = scale_raw # 先不 clamp，后面统一 clamp

#         # =====================================================================
#         #  【三重硬约束】 解决 OOM 的关键
#         # =====================================================================
        
#         # 约束 A: 绝对数值上限
#         # Baseline 是 0.01 (1cm)，我们稍微放宽到 0.05 (5cm)。
#         # 5cm 足够让近处的电线杆变得饱满，但绝不会大到爆显存。
#         # 只要这个值够小，光栅化器就是安全的。
#         scale_out = torch.clamp_max(scale_out, 0.05) 

#         # 约束 B: 相对深度约束 (投影几何保护)
#         # 规则: 一个球的半径绝对不能超过它距离相机深度的一半。
#         # 如果 radius > depth/2，球会遮挡极大视场。
#         # 比如 depth=1m，radius 最大 0.2m。
#         max_rel_scale = depth_safe * 0.2
#         scale_out = torch.minimum(scale_out, max_rel_scale)

#         # 约束 C: 远景回退 (Distance Culling)
#         # 超过 60米 的地方，强制回退到 0.01，和 Baseline 保持一致
#         # 这样保证天空和远景不会有任何显存风险
#         far_mask = depth_safe > 60.0
#         scale_out = torch.where(far_mask, torch.clamp_max(scale_out, 0.01), scale_out)

#         # 4. 极小值保底 (防止数值错误)
#         scale_out = torch.clamp_min(scale_out, 1e-6)

#         return rot_out, scale_out, opacity_out, sh_out