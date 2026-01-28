from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import upsample, conv2d, pack_cam_feat, unpack_cam_feat
from .volumetric_fusionnet import VFNet
from external.layers import ResnetEncoder

class FeatureMetricRefinement(nn.Module):
    def __init__(self, iterations=5, scale_factor=10.0, gravity_bias=2.0):
        super(FeatureMetricRefinement, self).__init__()
        self.iterations = iterations
        self.scale_factor = scale_factor
        self.gravity_bias = gravity_bias

    def forward(self, depth_map, feature_map):
        """
        depth_map: [B, 1, H, W] (Disparity or Depth)
        feature_map: [B, C, H, W]
        """
        # 确保 feature_map 和 depth_map 尺寸一致
        if feature_map.shape[-2:] != depth_map.shape[-2:]:
            feature_map = F.interpolate(feature_map, size=depth_map.shape[-2:], mode='bilinear', align_corners=True)

        refined_depth = depth_map.clone()
        
        # 1. 计算特征差异 (Feature Difference)
        # 使用 padding 保证 shift 后尺寸一致
        # Up (v-1), Down (v+1), Left (u-1), Right (u+1)
        feat_up = F.pad(feature_map, (0, 0, 0, 1))[..., 1:, :]
        feat_down = F.pad(feature_map, (0, 0, 1, 0))[..., :-1, :]
        feat_left = F.pad(feature_map, (0, 1, 0, 0))[..., :, 1:]
        feat_right = F.pad(feature_map, (1, 0, 0, 0))[..., :, :-1]

        # 计算亲和力权重 (Affinity Weights)
        # 垂直方向 (Gravity Bias)
        w_up = torch.exp(-torch.mean((feature_map - feat_up)**2, dim=1, keepdim=True) * self.scale_factor) * self.gravity_bias
        w_down = torch.exp(-torch.mean((feature_map - feat_down)**2, dim=1, keepdim=True) * self.scale_factor) * self.gravity_bias
        # 水平方向
        w_left = torch.exp(-torch.mean((feature_map - feat_left)**2, dim=1, keepdim=True) * self.scale_factor)
        w_right = torch.exp(-torch.mean((feature_map - feat_right)**2, dim=1, keepdim=True) * self.scale_factor)

        # 归一化权重
        w_sum = w_up + w_down + w_left + w_right + 1e-6
        
        # 2. 迭代扩散 (Diffusion)
        for _ in range(self.iterations):
            # 获取邻域深度
            d_up = F.pad(refined_depth, (0, 0, 0, 1))[..., 1:, :]
            d_down = F.pad(refined_depth, (0, 0, 1, 0))[..., :-1, :]
            d_left = F.pad(refined_depth, (0, 1, 0, 0))[..., :, 1:]
            d_right = F.pad(refined_depth, (1, 0, 0, 0))[..., :, :-1]
            
            # 加权平均更新 (Weighted Average Update)
            # D_new = (D_orig + sum(w * D_neighbor)) / (1 + sum(w)) 
            # 注意：这里我们使用一种简化的扩散形式，保留部分原始数据保真度
            neighbor_sum = w_up * d_up + w_down * d_down + w_left * d_left + w_right * d_right
            
            # 数据保真项权重 (lambda_data)，防止过度平滑
            # 这里设为 1.0，即原始深度占一份权重
            refined_depth = (depth_map + neighbor_sum) / (1.0 + w_sum)

        return refined_depth


class SupervisedDepthNetwork(nn.Module):
    """
    Depth fusion module (supervised with external depth)
    与原 depth_network 接口保持一致，输出同样的 disp/img_feat 结构。
    """

    def __init__(self, cfg):
        super(SupervisedDepthNetwork, self).__init__()
        self.read_config(cfg)

        # feature encoder
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)
        self.encoder = ResnetEncoder(self.num_layers, self.weights_init, 1)
        del self.encoder.encoder.fc
        enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:])
        self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode='reflect')

        # fusion net
        fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level]
        self.fusion_net = VFNet(cfg, self.fusion_feat_in_dim, fusion_feat_out_dim, model='depth')

        # depth decoder
        num_ch_enc = self.encoder.num_ch_enc[: (self.fusion_level + 1)]
        num_ch_dec = [16, 32, 64, 128, 256]
        self.decoder = DepthDecoder(self.fusion_level, num_ch_enc, num_ch_dec, self.scales, use_skips=self.use_skips)

        # === 【修改 1】 初始化 Refinement 模块 ===
        # iterations 可以根据显存和速度需求调整，通常 3-5 次即可
        self.refinement_net = FeatureMetricRefinement(iterations=5, scale_factor=10.0, gravity_bias=2.0)

    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def forward(self, inputs):
        outputs = {}

        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        lev = self.fusion_level

        # packed images for surrounding view
        sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        if self.novel_view_mode == 'MF':
            sf_images_last = torch.stack([inputs[('color_aug', -1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
            sf_images_next = torch.stack([inputs[('color_aug', 1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        packed_input = pack_cam_feat(sf_images)
        if self.novel_view_mode == 'MF':
            packed_input_last = pack_cam_feat(sf_images_last)
            packed_input_next = pack_cam_feat(sf_images_next)

        packed_feats = self.encoder(packed_input)
        if self.novel_view_mode == 'MF':
            packed_feats_last = self.encoder(packed_input_last)
            packed_feats_next = self.encoder(packed_input_next)

        _, _, up_h, up_w = packed_feats[lev].size()

        packed_feats_list = packed_feats[lev: lev + 1] + [
            F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev + 1:]
        ]
        if self.novel_view_mode == 'MF':
            packed_feats_last_list = packed_feats_last[lev: lev + 1] + [
                F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True)
                for feat in packed_feats_last[lev + 1:]
            ]
            packed_feats_next_list = packed_feats_next[lev: lev + 1] + [
                F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True)
                for feat in packed_feats_next[lev + 1:]
            ]

        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))
        if self.novel_view_mode == 'MF':
            packed_feats_agg_last = self.conv1x1(torch.cat(packed_feats_last_list, dim=1))
            packed_feats_agg_next = self.conv1x1(torch.cat(packed_feats_next_list, dim=1))

        feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams)
        if self.novel_view_mode == 'MF':
            feats_agg_last = unpack_cam_feat(packed_feats_agg_last, self.batch_size, self.num_cams)
            feats_agg_next = unpack_cam_feat(packed_feats_agg_next, self.batch_size, self.num_cams)

        fusion_dict = self.fusion_net(inputs, feats_agg)
        if self.novel_view_mode == 'MF':
            fusion_dict_last = self.fusion_net(inputs, feats_agg_last)
            fusion_dict_next = self.fusion_net(inputs, feats_agg_next)

        feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']]
        img_feat = []
        for i in range(len(feat_in)):
            img_feat.append(unpack_cam_feat(feat_in[i], self.batch_size, self.num_cams))

        if self.novel_view_mode == 'MF':
            feat_in_last = packed_feats_last[:lev] + [fusion_dict_last['proj_feat']]
            img_feat_last = []
            for i in range(len(feat_in_last)):
                img_feat_last.append(unpack_cam_feat(feat_in_last[i], self.batch_size, self.num_cams))

            feat_in_next = packed_feats_next[:lev] + [fusion_dict_next['proj_feat']]
            img_feat_next = []
            for i in range(len(feat_in_next)):
                img_feat_next.append(unpack_cam_feat(feat_in_next[i], self.batch_size, self.num_cams))

        packed_depth_outputs = self.decoder(feat_in)
        if self.novel_view_mode == 'MF':
            packed_depth_outputs_last = self.decoder(feat_in_last)
            packed_depth_outputs_next = self.decoder(feat_in_next)

        depth_outputs = unpack_cam_feat(packed_depth_outputs, self.batch_size, self.num_cams)
        if self.novel_view_mode == 'MF':
            depth_outputs_last = unpack_cam_feat(packed_depth_outputs_last, self.batch_size, self.num_cams)
            depth_outputs_next = unpack_cam_feat(packed_depth_outputs_next, self.batch_size, self.num_cams)

        # === 【修改 2】 在此处插入 Refinement 逻辑 ===
        
        # 我们使用 img_feat[0] 作为语义特征引导。
        # img_feat 是 unpacking 后的特征列表，img_feat[0] 对应 Encoder 的第0层输出（分辨率最高）。
        # 注意：depth_outputs 是 unpacked 的 [B, N_Cam, 1, H, W]，refinement 需要处理 [B*N_Cam, ...]
        # 但 unpack_cam_feat 已经把 B 和 N_Cam 分开了。为了并行处理，我们可以先 reshape 归并 B 和 Cam 维度，或者在循环里做。
        # 为了效率，我们直接对 Tensor 操作 (View folding)。

        # 获取当前帧的预测 disparity (scale 0)
        # shape: [B, Num_Cams, 1, H, W]
        raw_disp = depth_outputs[('disp', 0)]
        B, N_Cams, _, H, W = raw_disp.shape
        
        # 获取引导特征 (img_feat[0])
        # shape: [B, Num_Cams, C, H_feat, W_feat]
        guidance_feat = img_feat[0]

        # Reshape 为 [B*N, C, H, W] 以进行批量处理
        raw_disp_flat = raw_disp.view(B * N_Cams, 1, H, W)
        guidance_feat_flat = guidance_feat.view(B * N_Cams, -1, guidance_feat.shape[-2], guidance_feat.shape[-1])

        # 执行 Refinement
        refined_disp_flat = self.refinement_net(raw_disp_flat, guidance_feat_flat)

        # 恢复形状并覆盖原始输出
        depth_outputs[('disp', 0)] = refined_disp_flat.view(B, N_Cams, 1, H, W)

        # 如果开启了 MF (多帧) 模式，也需要对 last 和 next 帧进行同样的优化
        if self.novel_view_mode == 'MF':
            # Last frame
            raw_disp_last = depth_outputs_last[('disp', 0)]
            guidance_feat_last = img_feat_last[0]
            
            raw_disp_last_flat = raw_disp_last.view(B * N_Cams, 1, H, W)
            guidance_feat_last_flat = guidance_feat_last.view(B * N_Cams, -1, guidance_feat_last.shape[-2], guidance_feat_last.shape[-1])
            
            refined_disp_last = self.refinement_net(raw_disp_last_flat, guidance_feat_last_flat)
            depth_outputs_last[('disp', 0)] = refined_disp_last.view(B, N_Cams, 1, H, W)

            # Next frame
            raw_disp_next = depth_outputs_next[('disp', 0)]
            guidance_feat_next = img_feat_next[0]
            
            raw_disp_next_flat = raw_disp_next.view(B * N_Cams, 1, H, W)
            guidance_feat_next_flat = guidance_feat_next.view(B * N_Cams, -1, guidance_feat_next.shape[-2], guidance_feat_next.shape[-1])
            
            refined_disp_next = self.refinement_net(raw_disp_next_flat, guidance_feat_next_flat)
            depth_outputs_next[('disp', 0)] = refined_disp_next.view(B, N_Cams, 1, H, W)

        # === 修改结束，继续打包输出 ===
        
        for cam in range(self.num_cams):
            for k in depth_outputs.keys():
                outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]
            outputs[('cam', cam)][('img_feat', 0, 0)] = [feat[:, cam, ...] for feat in img_feat]
            if self.novel_view_mode == 'MF':
                outputs[('cam', cam)][('img_feat', -1, 0)] = [feat[:, cam, ...] for feat in img_feat_last]
                outputs[('cam', cam)][('img_feat', 1, 0)] = [feat[:, cam, ...] for feat in img_feat_next]
                outputs[('cam', cam)][('disp', -1, 0)] = depth_outputs_last[('disp', 0)][:, cam, ...]
                outputs[('cam', cam)][('disp', 1, 0)] = depth_outputs_next[('disp', 0)][:, cam, ...]

        return outputs


class DepthDecoder(nn.Module):
    def __init__(self, level_in, num_ch_enc, num_ch_dec, scales=range(2), use_skips=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = 1
        self.scales = scales
        self.use_skips = use_skips

        self.level_in = level_in
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        self.convs = OrderedDict()
        for i in range(self.level_in, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin='ELU')

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin='ELU')

        for s in self.scales:
            self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin=None)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        x = input_features[-1]
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in self.scales:
                outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))
        return outputs
