"""
GS补全网络 - 方法B：每像素预测 + 有效性分数

动机：如果没有前景遮挡，背景也会完整呈现在图片中。
所以现有像素能对应到完整的背景被遮挡的GS球。

设计：
- 每个像素预测一个潜在的被遮挡背景GS球
- 网络学习"有效性分数"来判断哪些像素对应被遮挡的背景
- 输出：额外的GS参数（xyz偏移、旋转、缩放、不透明度、球谐函数）+ 有效性mask
"""

import torch
from torch import nn
from .extractor import UnetExtractor, ResidualBlock
from einops import rearrange


class GSCompletionNetwork(nn.Module):
    #claude: 添加debug参数
    def __init__(self, rgb_dim=3, depth_dim=1, norm_fn='group', debug=False):
        super().__init__()
        self.rgb_dims = [64, 64, 128]
        self.depth_dims = [32, 48, 96]
        self.decoder_dims = [48, 64, 96]
        self.head_dim = 32

        #claude: 保存debug标志
        self.debug = debug

        self.sh_degree = 4
        self.d_sh = (self.sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        # 深度编码器（与GaussianNetwork共享架构）
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        # 解码器
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

        # 预测旋转
        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )

        # 预测缩放
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )

        # 预测不透明度
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 预测球谐函数
        self.sh_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3 * self.d_sh, kernel_size=1),
        )

        # === Student不确定性建模Head ===
        # Head A: 存在性先验（Occlusion-aware Existence Prior）
        # 预测每个像素方向上是否可能存在被遮挡的结构
        self.existence_prior_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()  # 输出[0,1]概率
        )

        # Head B: 深度偏移分布（Depth Offset Distribution）
        # 预测新增GS相对于当前深度的偏移量分布（均值和标准差）
        self.depth_offset_mean_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Softplus()  # 确保>0，因为偏移在深度之后
        )

        self.depth_offset_std_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Softplus()  # 确保>0
        )

    def forward(self, img, depth, img_feat):
        """
        预测额外的GS球用于补全被遮挡的背景

        Args:
            img: RGB图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]（包含0值的遮挡区域）
            img_feat: RGB特征 (feat1, feat2, feat3)

        Returns:
            dict: {
                'rotation': 旋转四元数 [B, 4, H, W]
                'scale': 缩放 [B, 3, H, W]
                'opacity': 不透明度 [B, 1, H, W]
                'sh': 球谐函数 [B, H*W, 1, 3, d_sh]
                'existence_prior': 存在性先验 [B, 1, H, W]
                'depth_offset_mean': 深度偏移均值 [B, 1, H, W]
                'depth_offset_std': 深度偏移标准差 [B, 1, H, W]
            }
        """
        # 提取特征
        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)

        # 解码
        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        # 预测各个参数
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)  # [B, 4, H, W]

        scale_out = torch.clamp_max(self.scale_head(out), 0.01)  # [B, 3, H, W]

        opacity_out = self.opacity_head(out)  # [B, 1, H, W]

        sh_out = self.sh_head(out)  # [B, 3*d_sh, H, W]
        sh_out = rearrange(sh_out, "n c h w -> n (h w) c")
        sh_out = rearrange(sh_out, "... (srf c) -> ... srf () c", srf=1)
        sh_out = rearrange(sh_out, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh_out = sh_out * self.sh_mask  # [B, H*W, 1, 3, d_sh]

        # Student不确定性建模输出
        existence_prior = self.existence_prior_head(out)  # [B, 1, H, W]
        depth_offset_mean = self.depth_offset_mean_head(out)  # [B, 1, H, W]
        depth_offset_std = self.depth_offset_std_head(out) + 1e-6  # [B, 1, H, W]，避免0

        #claude: 添加调试输出（仅第一个样本）
        if self.debug:
            # 只输出第一个样本的关键指标（注释掉避免刷屏）
            pass
            # print("\n[补全网络预测] 样本0:")
            # print(f"  尺度范围: {scale_out[0].min().item():.4f}~{scale_out[0].max().item():.4f}")
            # print(f"  不透明度: {opacity_out[0].mean().item():.3f}±{opacity_out[0].std().item():.3f}")
            # print(f"  存在性先验: {existence_prior[0].mean().item():.3f}±{existence_prior[0].std().item():.3f}")
            # print(f"  深度偏移均值: {depth_offset_mean[0].mean().item():.3f}±{depth_offset_mean[0].std().item():.3f}")
            # print(f"  深度偏移标准差: {depth_offset_std[0].mean().item():.3f}±{depth_offset_std[0].std().item():.3f}")

        return {
            'rotation': rot_out,
            'scale': scale_out,
            'opacity': opacity_out,
            'sh': sh_out,
            'existence_prior': existence_prior,
            'depth_offset_mean': depth_offset_mean,
            'depth_offset_std': depth_offset_std
        }
