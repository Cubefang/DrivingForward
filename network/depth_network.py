# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .blocks import upsample, conv2d, pack_cam_feat, unpack_cam_feat
# from .volumetric_fusionnet import VFNet

# from external.layers import ResnetEncoder

# class DepthNetwork(nn.Module):
#     """
#     Depth fusion module
#     """    
#     def __init__(self, cfg):
#         super(DepthNetwork, self).__init__()
#         self.read_config(cfg)
        
#         # feature encoder        
#         # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)        
#         self.encoder = ResnetEncoder(self.num_layers, self.weights_init, 1) # number of layers, pretrained, number of input images
#         del self.encoder.encoder.fc # del fc in weights_init
#         enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:]) 
#         self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode = 'reflect') 

#         # fusion net
#         fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level] 
#         self.fusion_net = VFNet(cfg, self.fusion_feat_in_dim, fusion_feat_out_dim, model ='depth')
        
#         # depth decoder
#         num_ch_enc = self.encoder.num_ch_enc[:(self.fusion_level+1)] 
#         num_ch_dec = [16, 32, 64, 128, 256]
#         self.decoder = DepthDecoder(self.fusion_level, num_ch_enc, num_ch_dec, self.scales, use_skips = self.use_skips)
    
#     def read_config(self, cfg):
#         for attr in cfg.keys(): 
#             for k, v in cfg[attr].items():
#                 setattr(self, k, v)

#     def forward(self, inputs):
#         '''
#         dict_keys(['idx', 'sensor_name', 'filename', 'extrinsics', 'mask', 
#         ('K', 0), ('inv_K', 0), ('color', 0, 0), ('color_aug', 0, 0), 
#         ('K', 1), ('inv_K', 1), ('color', 0, 1), ('color_aug', 0, 1), 
#         ('K', 2), ('inv_K', 2), ('color', 0, 2), ('color_aug', 0, 2), 
#         ('K', 3), ('inv_K', 3), ('color', 0, 3), ('color_aug', 0, 3), 
#         ('color', -1, 0), ('color_aug', -1, 0), ('color', 1, 0), ('color_aug', 1, 0), 'extrinsics_inv'])
#         '''

#         outputs = {}
        
#         # dictionary initialize
#         for cam in range(self.num_cams): # self.num_cames = 6
#             outputs[('cam', cam)] = {} # outputs = {('cam', 0): {}, ..., ('cam', 5): {}}
        
#         lev = self.fusion_level # 2
        
#         # packed images for surrounding view
#         sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1) 
#         if self.novel_view_mode == 'MF':
#             sf_images_last = torch.stack([inputs[('color_aug', -1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
#             sf_images_next = torch.stack([inputs[('color_aug', 1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
#         packed_input = pack_cam_feat(sf_images) 
#         if self.novel_view_mode == 'MF':
#             packed_input_last = pack_cam_feat(sf_images_last)
#             packed_input_next = pack_cam_feat(sf_images_next)
        
#         # feature encoder
#         packed_feats = self.encoder(packed_input) 
#         if self.novel_view_mode == 'MF':
#             packed_feats_last = self.encoder(packed_input_last)
#             packed_feats_next = self.encoder(packed_input_next)
#         # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
#         _, _, up_h, up_w = packed_feats[lev].size() 
        
#         packed_feats_list = packed_feats[lev:lev+1] \
#                         + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev+1:]]        
#         if self.novel_view_mode == 'MF':
#             packed_feats_last_list = packed_feats_last[lev:lev+1] \
#                             + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_last[lev+1:]]
#             packed_feats_next_list = packed_feats_next[lev:lev+1] \
#                             + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_next[lev+1:]]                

#         packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1)) 
#         if self.novel_view_mode == 'MF':
#             packed_feats_agg_last = self.conv1x1(torch.cat(packed_feats_last_list, dim=1))
#             packed_feats_agg_next = self.conv1x1(torch.cat(packed_feats_next_list, dim=1))

#         feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams) 
#         if self.novel_view_mode == 'MF':
#             feats_agg_last = unpack_cam_feat(packed_feats_agg_last, self.batch_size, self.num_cams)
#             feats_agg_next = unpack_cam_feat(packed_feats_agg_next, self.batch_size, self.num_cams)

#         # fusion_net, backproject each feature into the 3D voxel space
#         fusion_dict = self.fusion_net(inputs, feats_agg)
#         if self.novel_view_mode == 'MF':
#             fusion_dict_last = self.fusion_net(inputs, feats_agg_last)
#             fusion_dict_next = self.fusion_net(inputs, feats_agg_next)

#         feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']]   
#         img_feat = []
#         for i in range(len(feat_in)):
#             img_feat.append(unpack_cam_feat(feat_in[i], self.batch_size, self.num_cams))
        
#         if self.novel_view_mode == 'MF':
#             feat_in_last = packed_feats_last[:lev] + [fusion_dict_last['proj_feat']] 
#             img_feat_last = []
#             for i in range(len(feat_in_last)):
#                 img_feat_last.append(unpack_cam_feat(feat_in_last[i], self.batch_size, self.num_cams))
            
#             feat_in_next = packed_feats_next[:lev] + [fusion_dict_next['proj_feat']] 
#             img_feat_next = []
#             for i in range(len(feat_in_next)):
#                 img_feat_next.append(unpack_cam_feat(feat_in_next[i], self.batch_size, self.num_cams))

#         packed_depth_outputs = self.decoder(feat_in)  
#         if self.novel_view_mode == 'MF':      
#             packed_depth_outputs_last = self.decoder(feat_in_last)
#             packed_depth_outputs_next = self.decoder(feat_in_next)

#         depth_outputs = unpack_cam_feat(packed_depth_outputs, self.batch_size, self.num_cams) 
#         if self.novel_view_mode == 'MF':
#             depth_outputs_last = unpack_cam_feat(packed_depth_outputs_last, self.batch_size, self.num_cams)
#             depth_outputs_next = unpack_cam_feat(packed_depth_outputs_next, self.batch_size, self.num_cams)

#         for cam in range(self.num_cams):
#             for k in depth_outputs.keys():
#                 # depth_outputs.keys() -> dict_keys([('disp', 0)])
#                 import pdb; pdb.set_trace()
#                 outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]
#             outputs[('cam', cam)][('img_feat', 0, 0)] = [feat[:, cam, ...] for feat in img_feat] 
#             if self.novel_view_mode == 'MF':
#                 outputs[('cam', cam)][('img_feat', -1, 0)] = [feat[:, cam, ...] for feat in img_feat_last] 
#                 outputs[('cam', cam)][('img_feat', 1, 0)] = [feat[:, cam, ...] for feat in img_feat_next]
#                 outputs[('cam', cam)][('disp', -1, 0)] = depth_outputs_last[('disp', 0)][:, cam, ...] 
#                 outputs[('cam', cam)][('disp', 1, 0)] = depth_outputs_next[('disp', 0)][:, cam, ...] 

#         return outputs
    
        
# class DepthDecoder(nn.Module):
#     def __init__(self, level_in, num_ch_enc, num_ch_dec, scales=range(2), use_skips=False):
#         super(DepthDecoder, self).__init__()

#         self.num_output_channels = 1
#         self.scales = scales
#         self.use_skips = use_skips
        
#         self.level_in = level_in
#         self.num_ch_enc = num_ch_enc
#         self.num_ch_dec = num_ch_dec

#         self.convs = OrderedDict()
#         for i in range(self.level_in, -1, -1):
#             num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

#             num_ch_in = self.num_ch_dec[i]
#             if self.use_skips and i > 0:
#                 num_ch_in += self.num_ch_enc[i - 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

#         for s in self.scales:
#             self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin = None)

#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input_features):
#         outputs = {}
        
#         # decode
#         x = input_features[-1]
#         for i in range(self.level_in, -1, -1):
#             x = self.convs[('upconv', i, 0)](x)
#             x = [upsample(x)]
#             if self.use_skips and i > 0:
#                 x += [input_features[i - 1]]
#             x = torch.cat(x, 1)
#             x = self.convs[('upconv', i, 1)](x)
#             if i in self.scales:
#                 outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))                
#         return outputs

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import upsample, conv2d, pack_cam_feat, unpack_cam_feat
from .volumetric_fusionnet import VFNet

from external.layers import ResnetEncoder

class DepthNetwork(nn.Module):
    """
    RGB-D 特征融合模块 (替代原有的 DepthNetwork)
    
    新角色的功能:
    1. 接收 RGB 图像 和 深度真值 (Ground Truth Depth) 作为输入。
    2. 融合两者以生成更强的语义特征 (img_feat)。
    3. 直接透传 GT Depth 作为“预测”的深度/视差，不再进行解码预测。
    """    
    def __init__(self, cfg):
        super(DepthNetwork, self).__init__()
        self.read_config(cfg)
        
        # --- 修改点 1: 输入适配层 (Input Adaptation Layer) ---
        # 我们现在需要处理 4 个通道 (3 RGB + 1 Depth)。
        # 我们添加一个小型的卷积层，在送入 ResnetEncoder 之前将它们融合回 3 个通道。
        # 这样做的好处是我们可以保持标准的 ResnetEncoder 代码不变，无需修改底层库。
        self.input_fusion_layer = conv2d(4, 3, kernel_size=3, stride=1)
        
        # 特征编码器 (Feature Encoder)
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)        
        # 标准编码器现在接收的是经过适配层处理后的 "RGBD融合" 特征
        self.encoder = ResnetEncoder(self.num_layers, self.weights_init, 1) 
        del self.encoder.encoder.fc 
        enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:]) 
        self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode = 'reflect') 

        # 融合网络 (Fusion Net) - 保留用于多视角特征聚合
        fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level] 
        self.fusion_net = VFNet(cfg, self.fusion_feat_in_dim, fusion_feat_out_dim, model ='depth')
        
        # --- 修改点 2: 移除深度解码器 (Removed DepthDecoder) ---
        # 我们不再需要解码深度，因为我们已经有了真值 (GT)。
        # self.decoder = DepthDecoder(...) # 已移除
    
    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def forward(self, inputs):
        """
        保留了原始的输入字典结构。
        现在我们会主动利用输入的深度真值 (如果存在)。
        """
        outputs = {}
        
        # 字典初始化
        for cam in range(self.num_cams): 
            outputs[('cam', cam)] = {} 
        
        lev = self.fusion_level # 通常为 2
        
        # --- 修改点 3: 准备 RGB-D 输入数据 ---
        
        # 辅助函数：用于准备 [B, N, 4, H, W] 格式的输入张量
        def prepare_rgbd_input(image_key, depth_key_hint='depth_gt'):
            # 堆叠图像: [B, N_cam, 3, H, W]
            imgs = torch.stack([inputs[image_key][:, cam, ...] for cam in range(self.num_cams)], 1)
            
            # 堆叠深度: [B, N_cam, 1, H, W]
            # 注意：你需要确保数据加载器 (DataLoader) 的 inputs 中包含 'depth_gt' 键。
            # 如果你的数据加载器使用特定的键名 (例如 ('depth_gt', 0, 0))，请在此处调整。
            
            # 检查常见的深度键名
            if (depth_key_hint, 0, 0) in inputs:
                depths = torch.stack([inputs[(depth_key_hint, 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
            else:
                # 如果没找到深度，为了防止报错，使用全0填充 (虽然这违背了初衷，用于调试)
                # print("警告: 未找到 GT Depth，使用全0代替")
                depths = torch.zeros_like(imgs[:, :, 0:1, ...])

            # 深度归一化 (Normalization)
            # 这对 Encoder 的数值稳定性至关重要。假设最大深度约为 100米 (自动驾驶典型值)。
            # 将其归一化到 0-1 范围。
            norm_depths = torch.clamp(depths / 100.0, 0, 1)
            
            # 在通道维度拼接: [B, N_cam, 4, H, W]
            return torch.cat([imgs, norm_depths], dim=2), depths

        # 准备当前帧输入
        rgbd_current, raw_depths_current = prepare_rgbd_input(('color_aug', 0, 0))
        packed_input = pack_cam_feat(rgbd_current)
        
        # 处理 MF (多帧/时序) 模式
        if self.novel_view_mode == 'MF':
            rgbd_last, raw_depths_last = prepare_rgbd_input(('color_aug', -1, 0))
            rgbd_next, raw_depths_next = prepare_rgbd_input(('color_aug', 1, 0))
            packed_input_last = pack_cam_feat(rgbd_last)
            packed_input_next = pack_cam_feat(rgbd_next)
        
        # --- 修改点 4: 特征编码 (包含融合层) ---
        
        # 将 4通道输入通过适配层 -> 变为 3通道
        x_curr = self.input_fusion_layer(packed_input)
        packed_feats = self.encoder(x_curr) 
        
        if self.novel_view_mode == 'MF':
            x_last = self.input_fusion_layer(packed_input_last)
            x_next = self.input_fusion_layer(packed_input_next)
            packed_feats_last = self.encoder(x_last)
            packed_feats_next = self.encoder(x_next)

        # ... (标准的特征聚合逻辑保持不变) ...
        _, _, up_h, up_w = packed_feats[lev].size() 
        
        packed_feats_list = packed_feats[lev:lev+1] \
                        + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev+1:]]        
        if self.novel_view_mode == 'MF':
            packed_feats_last_list = packed_feats_last[lev:lev+1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_last[lev+1:]]
            packed_feats_next_list = packed_feats_next[lev:lev+1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_next[lev+1:]]                

        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1)) 
        if self.novel_view_mode == 'MF':
            packed_feats_agg_last = self.conv1x1(torch.cat(packed_feats_last_list, dim=1))
            packed_feats_agg_next = self.conv1x1(torch.cat(packed_feats_next_list, dim=1))

        feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams) 
        if self.novel_view_mode == 'MF':
            feats_agg_last = unpack_cam_feat(packed_feats_agg_last, self.batch_size, self.num_cams)
            feats_agg_next = unpack_cam_feat(packed_feats_agg_next, self.batch_size, self.num_cams)

        # fusion_net 逻辑 (体积融合)
        fusion_dict = self.fusion_net(inputs, feats_agg)
        if self.novel_view_mode == 'MF':
            fusion_dict_last = self.fusion_net(inputs, feats_agg_last)
            fusion_dict_next = self.fusion_net(inputs, feats_agg_next)

        # 构建最终的 img_feat 列表
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

        # --- 修改点 5: 绕过解码器，直接使用 GT 深度 ---
        # 我们跳过了 self.decoder() 的调用。
        # 注意：下游的 GSNet (Gaussian Splatting Network) 通常期望的是 "视差 (disparity)" 或 "深度 (depth)"。
        # 如果原始代码使用的是 'disp'，我们在这里需要进行转换 (disp = 1/depth)。
        
        # --- 修改点 6: 最终输出组装 ---
        for cam in range(self.num_cams):
            # 1. 传递图像特征 (这是融合了深度信息后的"增强版"特征)
            outputs[('cam', cam)][('img_feat', 0, 0)] = [feat[:, cam, ...] for feat in img_feat] 
            
            # 2. 传递 GT 深度/视差
            # 重要：如果你的 GSNet 期望视差 (1/depth)，必须在这里转换。
            # 假设标准行为是 disp = 1 / depth
            curr_depth = raw_depths_current[:, cam, ...]
            # 加一个极小值 epsilon 防止除以零
            curr_disp = 1.0 / (curr_depth + 1e-6) 
            
            # 使用键名 ('disp', 0)，因为这通常是解码器输出的默认键名
            outputs[('cam', cam)][('disp', 0)] = curr_disp

            if self.novel_view_mode == 'MF':
                outputs[('cam', cam)][('img_feat', -1, 0)] = [feat[:, cam, ...] for feat in img_feat_last] 
                outputs[('cam', cam)][('img_feat', 1, 0)] = [feat[:, cam, ...] for feat in img_feat_next]
                
                # 处理 上一帧/下一帧 的深度
                last_depth = raw_depths_last[:, cam, ...]
                next_depth = raw_depths_next[:, cam, ...]
                
                outputs[('cam', cam)][('disp', -1, 0)] = 1.0 / (last_depth + 1e-6)
                outputs[('cam', cam)][('disp', 1, 0)] = 1.0 / (next_depth + 1e-6)

        return outputs