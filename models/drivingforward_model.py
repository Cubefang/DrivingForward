from collections import defaultdict
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
torch.manual_seed(0)

from dataset import construct_dataset
from network import *
from network.depth_network_supervised import SupervisedDepthNetwork

from .base_model import BaseModel
from .geometry import Pose, ViewRendering
from .losses import MultiCamLoss, SingleCamLoss

from .gaussian import GaussianNetwork, depth2pc, pts2render, focal2fov, getProjectionMatrix, getWorld2View2, rotate_sh
from .gaussian.gs_completion_network import GSCompletionNetwork
from .gaussian.depth_utils import depth_dilation, merge_gaussian_params
from .teachers import warp_future_to_current
from einops import rearrange

_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']


class DrivingForwardModel(BaseModel):
    def __init__(self, cfg, rank):
        super(DrivingForwardModel, self).__init__(cfg)
        self.rank = rank
        self.read_config(cfg)
        # ensure boolean flags are real bool (avoid YAML typos like "Flase")
        self.use_dvgt_supervision = bool(self.use_dvgt_supervision)
        self.enable_self_supervised_depth = bool(self.enable_self_supervised_depth)
        self.prepare_dataset(cfg, rank)
        self.models = self.prepare_model(cfg, rank)
        self.losses = self.init_losses(cfg, rank)
        self.view_rendering, self.pose = self.init_geometry(cfg, rank)
        self.set_optimizer()
        self.dvgt_teacher = None

        #claude: 添加debug计数器（用于控制每个epoch只输出一次）
        self.debug_batch_count = 0

        if self.pretrain and rank == 0:
            self.load_weights()
        if self.use_dvgt_supervision:
            self.init_dvgt_teacher()

        self.left_cam_dict = {2:0, 0:1, 4:2, 1:3, 5:4, 3:5}
        self.right_cam_dict = {0:2, 1:0, 2:4, 3:1, 4:5, 5:3}
        
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def init_dvgt_teacher(self):
        """
        初始化DVGT教师模型（启用有监督深度时调用）。
        """
        if self.dvgt_teacher is not None:
            return
        if '/home/lfliang/project/DVGT' not in sys.path:
            sys.path.append('/home/lfliang/project/DVGT')
        try:
            from dvgt.models.dvgt import DVGT
        except ImportError as e:
            raise ImportError("DVGT 模型导入失败，请确认路径 /home/lfliang/project/DVGT 可用") from e

        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        self.dvgt_teacher = DVGT().to(device).eval()
        for p in self.dvgt_teacher.parameters():
            p.requires_grad = False

        ckpt_path = getattr(self, 'dvgt_ckpt_path', None)
        if ckpt_path is None:
            raise ValueError("dvgt_ckpt_path 未设置，无法启用DVGT监督")
        state = torch.load(ckpt_path, map_location='cpu')
        self.dvgt_teacher.load_state_dict(state)
        print(f"[DVGT] loaded teacher checkpoint from {ckpt_path}")
                
    def init_geometry(self, cfg, rank):
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
        
    def init_losses(self, cfg, rank):
        if self.spatio_temporal or self.spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else:
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model
        
    def prepare_model(self, cfg, rank):
        models = {}
        if self.use_pose_net:
            models['pose_net'] = self.set_posenet(cfg)
        depth_net_cls = SupervisedDepthNetwork if getattr(self, 'use_supervised_depth_net', False) else DepthNetwork
        models['depth_net'] = self.set_depthnet(cfg, depth_net_cls)
        if self.gaussian:
            models['gs_net'] = self.set_gaussiannet(cfg)
            # 添加GS补全网络
            if getattr(self, 'use_gs_completion', False):
                models['gs_completion_net'] = self.set_gs_completion_net(cfg)

        return models
        if self.use_pose_net:
            models['pose_net'] = self.set_posenet(cfg)
        depth_net_cls = SupervisedDepthNetwork if getattr(self, 'use_supervised_depth_net', False) else DepthNetwork
        models['depth_net'] = self.set_depthnet(cfg, depth_net_cls)
        if self.gaussian:
            models['gs_net'] = self.set_gaussiannet(cfg)

        return models

    def set_posenet(self, cfg):
        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        return PoseNetwork(cfg).to(device)
        
    def set_depthnet(self, cfg, net_cls=DepthNetwork):
        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        return net_cls(cfg).to(device)

    def set_gaussiannet(self, cfg):
        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        return GaussianNetwork(rgb_dim=3, depth_dim=1).to(device)

    #claude: 添加debug参数
    def set_gs_completion_net(self, cfg):
        """初始化GS补全网络"""
        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        debug = getattr(self, 'gs_completion_debug', False)
        return GSCompletionNetwork(rgb_dim=3, depth_dim=1, debug=debug).to(device)

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')
        
        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            if rank == 0 :
                self.set_val_dataloader(cfg)
                
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)

    def set_train_dataloader(self, cfg, rank):
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # === 快速调试：交换train和val ===
        swap_train_val = cfg.get('data', {}).get('swap_train_val', False)
        split_name = 'val' if swap_train_val else 'train'

        if swap_train_val:
            print(f"[DEBUG] 快速调试模式：使用 '{split_name}' 数据集作为训练集")

        # construct train dataset
        train_dataset = construct_dataset(cfg, split_name, **_augmentation)

        # === 快速调试模式：不打乱数据，保持每个epoch顺序一致 ===
        use_shuffle = not swap_train_val  # 调试模式下不打乱，正常训练时打乱

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': use_shuffle,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        if swap_train_val:
            print(f"[DEBUG] 数据加载器设置: shuffle={use_shuffle} (保持每个epoch顺序一致)")

        self._dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_val_dataloader(self, cfg):
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # === 快速调试：交换train和val ===
        swap_train_val = cfg.get('data', {}).get('swap_train_val', False)
        split_name = 'train' if swap_train_val else 'val'
        max_val_samples = cfg.get('data', {}).get('max_val_samples', None)

        if swap_train_val:
            print(f"[DEBUG] 快速调试模式：使用 '{split_name}' 数据集作为验证集")
            if max_val_samples:
                print(f"[DEBUG] 验证集限制为前 {max_val_samples} 个样本")

        # construct validation dataset
        val_dataset = construct_dataset(cfg, split_name, **_augmentation)

        # === 限制验证集样本数量 ===
        if max_val_samples and len(val_dataset) > max_val_samples:
            from torch.utils.data import Subset
            indices = list(range(max_val_samples))
            val_dataset = Subset(val_dataset, indices)
            print(f"[DEBUG] 验证集从 {len(val_dataset.dataset)} 减少到 {len(val_dataset)} 个样本")

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['val']  = DataLoader(val_dataset, **dataloader_opts)
    
    def set_eval_dataloader(self, cfg):  
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        eval_dataset = construct_dataset(cfg, 'eval', **_augmentation)

        self._dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)

    def set_optimizer(self):
        parameters_to_train = []
        for v in self.models.values():
            parameters_to_train += list(v.parameters())

        self.optimizer = optim.Adam(
        parameters_to_train, 
            self.learning_rate
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            self.scheduler_step_size,
            0.1
        )
    
    def process_batch(self, inputs, rank):
        """
        Pass a minibatch through the network and generate images, depth maps, and losses.
        """
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                if 'ego_pose' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(rank)   

        outputs = self.estimate(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses  

    def estimate(self, inputs):
        """
        This function estimates the outputs of the network.
        """          
        # pre-calculate inverse of the extrinsic matrix
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        if self.use_pose_net:
            pose_pred = self.predict_pose(inputs)
        depth_feats = self.predict_depth(inputs)
        # 若使用弱监督深度网络，则不再用外部深度覆盖预测（保留梯度）
        use_ext_depth = getattr(self, 'use_dvgt_depth', False) and (not getattr(self, 'use_supervised_depth_net', False))
        if use_ext_depth and 'depth' in inputs:
            inputs, depth_feats = self.override_depth_feats_with_external(inputs, depth_feats)

        for cam in range(self.num_cams):       
            if self.mode != 'train' or (not self.use_pose_net):
                outputs[('cam', cam)].update({('cam_T_cam', 0, 1): inputs[('cam_T_cam', 0, 1)][:, cam, ...]})
                outputs[('cam', cam)].update({('cam_T_cam', 0, -1): inputs[('cam_T_cam', 0, -1)][:, cam, ...]}) 
            elif self.mode == 'train':
                outputs[('cam', cam)].update(pose_pred[('cam', cam)])                
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])
            
        self.compute_depth_maps(inputs, outputs)
        # 若使用外部深度，则直接用原始外部深度覆盖 outputs 中的 depth（保持 raw 值域）
        if use_ext_depth and 'depth' in inputs:
            ext_depth = inputs['depth']
            if ext_depth.dim() == 5:
                ext_depth = ext_depth.squeeze(2)
            source_scale = 0
            for cam in range(self.num_cams):
                depth_raw = ext_depth[:, cam:cam+1, ...]
                depth_resized = F.interpolate(depth_raw, [self.height, self.width], mode='bilinear', align_corners=False)
                outputs[('cam', cam)][('depth', 0, 0)] = depth_resized
                for scale in self.scales:
                    if scale == 0:
                        continue
                    h_s = self.height // (2 ** scale)
                    w_s = self.width // (2 ** scale)
                    d_s = F.interpolate(depth_resized, [h_s, w_s], mode='bilinear', align_corners=False)
                    outputs[('cam', cam)][('depth', 0, scale)] = d_s
        if getattr(self, 'use_dvgt_supervision', False):
            self.inject_teacher_depth(inputs, outputs)
        return outputs

    def predict_pose(self, inputs):      
        """
        This function predicts poses.
        """          
        net = self.models['pose_net']
        
        pose = self.pose.compute_pose(net, inputs)
        return pose

    def predict_depth(self, inputs):
        """
        This function predicts disparity maps.
        """                  
        net = self.models['depth_net']

        depth_feats = net(inputs)
        return depth_feats
    
    def compute_depth_maps(self, inputs, outputs):     
        """
        This function computes depth map for each viewpoint.
        """                  
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', 0, scale)] = self.to_depth(disp, ref_K)
                if self.novel_view_mode == 'MF':
                    disp_last = outputs[('cam', cam)][('disp', -1, scale)]
                    outputs[('cam', cam)][('depth', -1, scale)] = self.to_depth(disp_last, ref_K)
                    disp_next = outputs[('cam', cam)][('disp', 1, scale)]
                    outputs[('cam', cam)][('depth', 1, scale)] = self.to_depth(disp_next, ref_K)
    
    def to_depth(self, disp_in, K_in):        
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale

    @torch.no_grad()
    def inject_teacher_depth(self, inputs, outputs):
        """
        使用DVGT教师模型生成监督深度，存入outputs供后续loss使用。
        """
        if self.dvgt_teacher is None:
            return
        imgs = inputs[('color', 0, 0)]  # [B, V, C, H, W]
        b, v, _, h, w = imgs.shape

        teacher_in = imgs.unsqueeze(1)  # [B, 1, V, 3, H, W]
        teacher_out = self.dvgt_teacher(teacher_in)
        if 'world_points' not in teacher_out:
            return

        pts = teacher_out['world_points'][:, 0]  # [B, V, H, W, 3]
        depth = torch.norm(pts, dim=-1, keepdim=True)  # [B, V, H, W, 1]
        conf = teacher_out.get('world_points_conf', None)
        if conf is not None:
            conf = conf[:, 0].unsqueeze(2)  # [B, V, 1, H, W]
            conf = (conf >= getattr(self, 'dvgt_conf_threshold', 0.0)).float()

        depth = depth.permute(0, 1, 4, 2, 3).contiguous()  # [B, V, 1, H, W]
        depth = depth.view(b * v, 1, h, w)
        depth_resized = F.interpolate(depth, [self.height, self.width], mode='bilinear', align_corners=False)
        depth_resized = depth_resized.view(b, v, 1, self.height, self.width)

        if conf is not None:
            conf = conf.permute(0, 1, 2, 3, 4).contiguous()
            conf = conf.view(b * v, 1, h, w)
            conf_resized = F.interpolate(conf, [self.height, self.width], mode='nearest')
            conf_resized = conf_resized.view(b, v, 1, self.height, self.width)
        else:
            conf_resized = None

        for cam in range(self.num_cams):
            outputs[('cam', cam)][('teacher_depth', 0, 0)] = depth_resized[:, cam, ...]
            if conf_resized is not None:
                outputs[('cam', cam)][('teacher_conf', 0, 0)] = conf_resized[:, cam, ...]
            for scale in self.scales:
                if scale == 0:
                    continue
                h_s = self.height // (2 ** scale)
                w_s = self.width // (2 ** scale)
                d_s = F.interpolate(depth_resized[:, cam, ...], [h_s, w_s], mode='bilinear', align_corners=False)
                outputs[('cam', cam)][('teacher_depth', 0, scale)] = d_s
                if conf_resized is not None:
                    m_s = F.interpolate(conf_resized[:, cam, ...], [h_s, w_s], mode='nearest')
                    outputs[('cam', cam)][('teacher_conf', 0, scale)] = m_s

    def override_depth_feats_with_external(self, inputs, depth_feats):
        """
        在 depth_feats 尚未写入 outputs 前，用外部深度替换其中的 disp。
        """
        ext_depth = inputs['depth']
        # 兼容 [B, V, H, W] 或 [B, V, 1, H, W]
        if ext_depth.dim() == 5:
            ext_depth = ext_depth.squeeze(2)
        if ext_depth.dim() != 4:
            raise ValueError(f"external depth expected 4D or 5D, got shape {ext_depth.shape}")
        b, v, h0, w0 = ext_depth.shape
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        disp_range = max_disp - min_disp
        source_scale = 0
        debug_info = {}

        for cam in range(self.num_cams):
            if ('cam', cam) not in depth_feats:
                continue
            depth_raw = ext_depth[:, cam:cam+1, ...]  # [B,1,H,W]
            depth_resized = F.interpolate(depth_raw, [self.height, self.width], mode='bilinear', align_corners=False)

            K = inputs[('K', source_scale)][:, cam, ...]
            focal = K[:, 0, 0].view(-1, 1, 1, 1)

            disp_target = focal / self.focal_length_scale / torch.clamp(depth_resized, min=1e-6)
            disp_in = (disp_target - min_disp) / disp_range
            disp_in = torch.clamp(disp_in, 0.0, 1.0)

            # 更新 depth_feats 的 disp
            for scale in self.scales:
                h_s = self.height // (2 ** scale)
                w_s = self.width // (2 ** scale)
                disp_scaled = F.interpolate(disp_in, [h_s, w_s], mode='bilinear', align_corners=False)
                depth_feats[('cam', cam)][('disp', scale)] = disp_scaled
        return inputs, depth_feats

    def get_gaussian_data(self, inputs, outputs, cam):
        """
        This function computes gaussian data for each viewpoint.
        """
        bs, _, height, width = inputs[('color', 0, 0)][:, cam, ...].shape
        zfar = self.max_depth
        znear = 0.01

        if self.novel_view_mode == 'MF':
            for frame_id in self.frame_ids:
                if frame_id == 0:
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
                    FovX_list = []
                    FovY_list = []
                    world_view_transform_list = []
                    full_proj_transform_list = []
                    camera_center_list = []
                    for i in range(bs):
                        intr = inputs[('K', 0)][:, cam, ...][i,:]
                        extr = inputs['extrinsics_inv'][:, cam, ...][i,:]
                        FovX = focal2fov(intr[0, 0], width)
                        FovY = focal2fov(intr[1, 1], height)
                        device = inputs[('color', 0, 0)].device
                        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).to(device)
                        world_view_transform = torch.tensor(extr).transpose(0, 1).to(device)
                        # full_proj_transform: (E^T K^T) = (K E)^T
                        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                        camera_center = world_view_transform.inverse()[3, :3] 

                        FovX_list.append(FovX)
                        FovY_list.append(FovY)
                        world_view_transform_list.append(world_view_transform.unsqueeze(0))
                        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                        camera_center_list.append(camera_center.unsqueeze(0))

                    device = inputs[('color', 0, 0)].device
                    outputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).to(device)
                    outputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).to(device)
                    outputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
                    outputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
                    outputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)
                else:
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                        torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                        torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))
                outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
                valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
                outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
                rot_maps, scale_maps, opacity_maps, sh_maps = \
                    self.gs_net(
                        inputs[('color', frame_id, 0)][:, cam, ...],
                        outputs[('cam', cam)][('depth', frame_id, 0)],
                        outputs[('cam', cam)][('img_feat', frame_id, 0)],
                        # intrinsics=inputs[('K', 0)][:, cam, ...]
                    )

                c2w_rotations = rearrange(outputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3], "k i j -> k () () () i j")
                sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
                outputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
                outputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
                outputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
                outputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps
        elif self.novel_view_mode == 'SF':
            frame_id = 0
            outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
            outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
            outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
            valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
            outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
            # 预测gaussian球参数：方向、形状、透明度、球谐函数（颜色）
            # rot_maps, scale_maps, opacity_maps, sh_maps = \
            #     self.gs_net(inputs[('color', frame_id, 0)][:, cam, ...], outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('img_feat', frame_id, 0)])
            rot_maps, scale_maps, opacity_maps, sh_maps = \
                self.gs_net(
                    inputs[('color', frame_id, 0)][:, cam, ...],
                    outputs[('cam', cam)][('depth', frame_id, 0)],
                    outputs[('cam', cam)][('img_feat', frame_id, 0)],
                    # intrinsics=inputs[('K', 0)][:, cam, ...]
                )
            #世界坐标系下映射颜色
            c2w_rotations = rearrange(outputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3], "k i j -> k () () () i j")
            sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
            outputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
            outputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
            outputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
            outputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps

            # === GS补全网络 ===
            if getattr(self, 'use_gs_completion', False) and 'gs_completion_net' in self.models:
                #claude: 使用should_debug标志（仅第一个batch的cam 3输出）
                debug = getattr(self, '_should_debug_this_batch', False) and cam == 3

                # 1. Depth Dilation
                depth_current = outputs[('cam', cam)][('depth', frame_id, 0)]  # [B, 1, H, W]
                depth_dilated = depth_dilation(
                    depth_current,
                    kernel_size=getattr(self, 'depth_dilation_kernel', 5),
                    iterations=getattr(self, 'depth_dilation_iters', 1),
                    debug=debug
                )

                # 2. 调用GS补全网络
                img_current = inputs[('color', frame_id, 0)][:, cam, ...]  # [B, 3, H, W]
                img_feat_current = outputs[('cam', cam)][('img_feat', frame_id, 0)]  # list of features

                gs_completion_out = self.models['gs_completion_net'](
                    img_current,
                    depth_dilated,
                    img_feat_current
                )

                # 3. 暂时注释：计算补全区域的xyz（Step 3会重新实现采样逻辑）
                # TODO: Step 3 - 实现sample_completion_gaussians()
                # K = inputs[('K', 0)][:, cam, ...]  # [B, 3, 3]
                # xyz_completion = ...

                # 4. 保存补全相关的输出（暂时不保存xyz_completion）
                # outputs[('cam', cam)][('xyz_completion', frame_id, 0)] = xyz_completion
                outputs[('cam', cam)][('rot_maps_completion', frame_id, 0)] = gs_completion_out['rotation']
                outputs[('cam', cam)][('scale_maps_completion', frame_id, 0)] = gs_completion_out['scale']
                outputs[('cam', cam)][('opacity_maps_completion', frame_id, 0)] = gs_completion_out['opacity']
                outputs[('cam', cam)][('sh_maps_completion', frame_id, 0)] = gs_completion_out['sh']
                # 旧的validity已删除，不再保存
                # outputs[('cam', cam)][('validity_completion', frame_id, 0)] = gs_completion_out['validity']
                outputs[('cam', cam)][('depth_dilated', frame_id, 0)] = depth_dilated

                # 保存Student不确定性建模输出
                outputs[('cam', cam)][('completion_outputs', frame_id, 0)] = gs_completion_out

                #claude: 添加调试输出（debug已包含cam==0条件）
                if debug:
                    print(f"\n[场景补全执行] 相机{cam}:")
                    print(f"  补全网络已成功执行")

                # 可视化Student heads（每个epoch都输出）
                VISUALIZE_STUDENT = getattr(self, 'visualize_student_heads', False)
                VISUALIZE_BATCH_LIMIT = getattr(self, 'visualize_batch_limit', 20)
                if self.mode == 'train' and VISUALIZE_STUDENT:
                    if not hasattr(self, '_student_vis_batch_count'):
                        self._student_vis_batch_count = 0
                    if self._student_vis_batch_count < VISUALIZE_BATCH_LIMIT and cam == 3:
                        self._visualize_student_heads_impl(inputs, outputs, cam)
                        self._student_vis_batch_count += 1

                # Step 3: 采样生成新增GS
                self.sample_completion_gaussians(inputs, outputs, cam)

            # novel view
            for frame_id in self.frame_ids[1:]:
                outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                    torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                    torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))
                
                FovX_list = []
                FovY_list = []
                world_view_transform_list = []
                full_proj_transform_list = []
                camera_center_list = []
                for i in range(bs):
                    intr = inputs[('K', 0)][:, cam, ...][i,:]
                    extr = inputs['extrinsics_inv'][:, cam, ...][i,:]
                    T_i = outputs[('cam', cam)][('cam_T_cam', 0, frame_id)][i,:]
                    FovX = focal2fov(intr[0, 0], width)
                    FovY = focal2fov(intr[1, 1], height)
                    device = inputs[('color', 0, 0)].device
                    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).to(device)
                    world_view_transform = torch.matmul(T_i, torch.tensor(extr).to(device)).transpose(0, 1)
                    # full_proj_transform: (E^T K^T) = (K E)^T
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3] 
                    FovX_list.append(FovX)
                    FovY_list.append(FovY)
                    world_view_transform_list.append(world_view_transform.unsqueeze(0))
                    full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                    camera_center_list.append(camera_center.unsqueeze(0))
                outputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda()
                outputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda()
                outputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
                outputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
                outputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)
    
    def generate_teacher_ground_truth(self, inputs, outputs):
        """
        使用Render-and-Compare方法生成GS补全的训练真值

        核心思想：
        1. 用t时刻的GS渲染t+1视角（会有空洞，因为GS不完整）
        2. 对比渲染结果和t+1真值，找到空洞位置
        3. 用t+1真值作为监督信号

        需要inputs中包含：
        - ('color', 1, 0): t+1时刻RGB真值 [B, N_cam, 3, H, W]
        - ('depth', 1, 0): t+1时刻深度 [B, N_cam, 1, H, W]
        - outputs中的GS参数
        """
        # 可视化开关（从配置读取，训练时每个epoch前N个batch可视化）
        VISUALIZE_TEACHER_MASK = getattr(self, 'visualize_teacher_mask', False)
        VISUALIZE_BATCH_LIMIT = getattr(self, 'visualize_batch_limit', 20)  # 默认20，可配置

        # 每个epoch可视化前N个batch
        if not hasattr(self, '_teacher_vis_batch_count'):
            self._teacher_vis_batch_count = 0

        should_visualize = VISUALIZE_TEACHER_MASK and (self._teacher_vis_batch_count < VISUALIZE_BATCH_LIMIT)

        for cam in range(self.num_cams):
            # 检查是否有t+1时刻的真值
            if ('color', 1, 0) not in inputs:
                continue

            img_t1_gt = inputs[('color', 1, 0)][:, cam, ...]  # [B, 3, H, W] t+1真值

            # 获取t+1时刻深度
            if ('depth', 1, 0) in inputs:
                depth_t1 = inputs[('depth', 1, 0)][:, cam, ...]  # [B, 1, H, W]
                if depth_t1.dim() == 5:
                    depth_t1 = depth_t1.squeeze(2)
            else:
                continue

            # 1. 用t时刻的GS渲染t+1视角
            # 检查是否已经渲染了t+1视角（在pred_gaussian_imgs中）
            if ('gaussian_color', 1, 0) in outputs[('cam', cam)]:
                img_t1_rendered = outputs[('cam', cam)][('gaussian_color', 1, 0)]  # [B, 3, H, W]
            else:
                # 如果没有渲染，跳过（需要在训练流程中先调用pred_gaussian_imgs）
                print(f"[警告] 相机{cam}的t+1视角GS渲染结果不存在，跳过教师真值生成")
                continue

            # 2. 对比渲染结果和真值，生成mask
            # 计算像素差异（L1距离）
            diff = torch.abs(img_t1_rendered - img_t1_gt).mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # 2.1 使用预先生成的mask过滤无效区域（天空、车辆边缘等）
            # mask格式: [B, N_cam, 1, H, W]，值为0-1，1表示有效区域
            if 'mask' in inputs:
                scene_mask = inputs['mask'][:, cam, ...].float()  # [B, 1, H, W]
            else:
                # 如果没有mask，使用全1（不过滤）
                scene_mask = torch.ones_like(diff)

            # 2.2 可选：使用深度信息进一步过滤远处区域
            depth_filter_threshold = getattr(self, 'teacher_depth_filter_threshold', 100.0)
            if depth_filter_threshold > 0:
                # 过滤深度为0（天空）和深度过大（远处）的区域
                depth_mask = ((depth_t1 > 0.1) & (depth_t1 < depth_filter_threshold)).float()
            else:
                depth_mask = torch.ones_like(diff)

            # 组合mask：scene_mask AND depth_mask
            combined_mask = scene_mask * depth_mask

            # 2.3 使用统计方法：只标记显著高于均值的像素（空洞区域）
            # 只在有效区域内计算统计量
            valid_diff = diff * combined_mask
            num_valid_pixels = combined_mask.sum() + 1e-6
            diff_mean = valid_diff.sum() / num_valid_pixels
            diff_std = torch.sqrt(((valid_diff - diff_mean) ** 2 * combined_mask).sum() / num_valid_pixels)

            # 阈值：均值 + 2倍标准差（只有约5%的像素会被标记）
            threshold = diff_mean + 2.0 * diff_std

            # 最终mask：差异大 AND 在有效区域内
            teacher_mask_t1 = ((diff > threshold) * combined_mask).float()  # [B, 1, H, W] 在t+1视角

            # 2.4 反投影到t时刻（用于监督Head A的存在性先验）
            teacher_mask_t0 = self._reproject_mask_to_t0(
                teacher_mask_t1,
                depth_t1,
                inputs,
                cam
            )  # [B, 1, H, W] 在t视角

            # 3. 保存到outputs
            outputs[('cam', cam)][('teacher_rgb_gt', 0, 0)] = img_t1_gt
            outputs[('cam', cam)][('teacher_depth_gt', 0, 0)] = depth_t1
            outputs[('cam', cam)][('teacher_mask_t1', 0, 0)] = teacher_mask_t1  # t+1视角mask
            outputs[('cam', cam)][('teacher_mask_t0', 0, 0)] = teacher_mask_t0  # t视角mask（反投影）

            # 4. 可视化（只在前N个batch且cam==3）
            if should_visualize and cam == 3:
                self._visualize_teacher_masks(
                    inputs, outputs, cam,
                    img_t1_gt, img_t1_rendered, diff,
                    teacher_mask_t1, teacher_mask_t0,
                    scene_mask, combined_mask,
                    threshold
                )

        # 增加batch计数
        self._teacher_vis_batch_count += 1
        # 增加全局batch索引（每个batch只增加一次）
        if not hasattr(self, '_global_vis_batch_idx'):
            self._global_vis_batch_idx = 0
        self._global_vis_batch_idx += 1

    def _reproject_mask_to_t0(self, mask_t1, depth_t1, inputs, cam):
        """
        将t+1视角的mask反投影到t时刻视角

        Args:
            mask_t1: t+1视角的mask [B, 1, H, W]
            depth_t1: t+1时刻的深度 [B, 1, H, W]
            inputs: 输入数据（包含相机参数和位姿）
            cam: 相机索引

        Returns:
            mask_t0: t视角的mask [B, 1, H, W]
        """
        B, _, H, W = mask_t1.shape
        device = mask_t1.device

        # 获取相机内参和位姿
        K = inputs[('K', 0)][:, cam, :3, :3]  # [B, 3, 3]
        inv_K = inputs[('inv_K', 0)][:, cam, :3, :3]  # [B, 3, 3]

        # t到t+1的相对位姿
        T_t1_t0 = inputs[('cam_T_cam', 0, 1)]
        # 如果T_t1_t0包含多个相机，需要选择当前相机
        if T_t1_t0.dim() == 4:  # [B, N_cam, 4, 4]
            T_t1_t0 = T_t1_t0[:, cam, :, :]  # [B, 4, 4]

        # 创建像素网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=0).float()  # [3, H, W]
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]

        # Step 1: 反投影t+1像素到3D空间（t+1坐标系）
        pixel_coords_flat = pixel_coords.view(B, 3, -1)  # [B, 3, H*W]
        depth_flat = depth_t1.view(B, 1, -1)  # [B, 1, H*W]

        # 3D点在t+1坐标系
        points3d_t1 = torch.matmul(inv_K, pixel_coords_flat) * depth_flat  # [B, 3, H*W]
        points3d_t1_homo = torch.cat([points3d_t1, torch.ones(B, 1, H*W, device=device)], dim=1)  # [B, 4, H*W]

        # Step 2: 变换到t坐标系
        T_t0_t1 = torch.inverse(T_t1_t0)  # [B, 4, 4]
        points3d_t0 = torch.matmul(T_t0_t1, points3d_t1_homo)  # [B, 4, H*W]

        # Step 3: 投影到t时刻像素坐标
        points2d_t0 = torch.matmul(K, points3d_t0[:, :3, :])  # [B, 3, H*W]
        u_t0 = points2d_t0[:, 0, :] / (points2d_t0[:, 2, :] + 1e-7)  # [B, H*W]
        v_t0 = points2d_t0[:, 1, :] / (points2d_t0[:, 2, :] + 1e-7)  # [B, H*W]

        # 归一化到[-1, 1]用于grid_sample
        u_t0_norm = (u_t0 / (W - 1) - 0.5) * 2  # [B, H*W]
        v_t0_norm = (v_t0 / (H - 1) - 0.5) * 2  # [B, H*W]

        grid = torch.stack([u_t0_norm, v_t0_norm], dim=-1)  # [B, H*W, 2]
        grid = grid.view(B, H, W, 2)  # [B, H, W, 2]

        # Step 4: 使用grid_sample进行反投影采样
        mask_t0 = torch.nn.functional.grid_sample(
            mask_t1,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, 1, H, W]

        return mask_t0

    def _visualize_teacher_masks(self, inputs, outputs, cam,
                                  img_t1_gt, img_t1_rendered, diff,
                                  teacher_mask_t1, teacher_mask_t0,
                                  scene_mask, combined_mask, threshold):
        """
        可视化Teacher生成的两个mask（t+1和t）
        """
        import os
        import cv2
        import numpy as np

        # 创建输出目录
        vis_dir = os.path.join(self.log_dir, 'teacher_mask_vis')
        os.makedirs(vis_dir, exist_ok=True)

        epoch = getattr(self, 'epoch', 0)
        batch_idx = self._teacher_vis_batch_count

        # 转换为numpy
        img_t1_gt_np = img_t1_gt[0].permute(1, 2, 0).detach().cpu().numpy()
        img_t1_gt_np = (img_t1_gt_np * 255).astype(np.uint8)
        img_t1_gt_np = cv2.cvtColor(img_t1_gt_np, cv2.COLOR_RGB2BGR)

        img_t1_rendered_np = img_t1_rendered[0].permute(1, 2, 0).detach().cpu().numpy()
        img_t1_rendered_np = (img_t1_rendered_np * 255).astype(np.uint8)
        img_t1_rendered_np = cv2.cvtColor(img_t1_rendered_np, cv2.COLOR_RGB2BGR)

        img_t0_rgb_np = inputs[('color', 0, 0)][0, cam].permute(1, 2, 0).detach().cpu().numpy()
        img_t0_rgb_np = (img_t0_rgb_np * 255).astype(np.uint8)
        img_t0_rgb_np = cv2.cvtColor(img_t0_rgb_np, cv2.COLOR_RGB2BGR)

        diff_np = diff[0, 0].detach().cpu().numpy()
        mask_t1_np = teacher_mask_t1[0, 0].detach().cpu().numpy()
        mask_t0_np = teacher_mask_t0[0, 0].detach().cpu().numpy()

        # 差异图
        diff_vis = np.clip(diff_np * 255, 0, 255).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

        # t+1 mask叠加
        overlay_t1 = img_t1_gt_np.copy()
        overlay_t1[mask_t1_np > 0.5] = [0, 0, 255]  # 红色
        vis_t1 = cv2.addWeighted(img_t1_gt_np, 0.7, overlay_t1, 0.3, 0)

        # t mask叠加
        overlay_t0 = img_t0_rgb_np.copy()
        overlay_t0[mask_t0_np > 0.5] = [0, 0, 255]  # 红色
        vis_t0 = cv2.addWeighted(img_t0_rgb_np, 0.7, overlay_t0, 0.3, 0)

        # 并排对比
        comparison = np.hstack([vis_t1, vis_t0])

        # 保存图像
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_t1_gt.png'), img_t1_gt_np)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_t1_rendered.png'), img_t1_rendered_np)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_t1_diff.png'), diff_color)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_t1_teacher_mask.png'), vis_t1)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_t0_rgb.png'), img_t0_rgb_np)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_t0_teacher_mask.png'), vis_t0)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_mask_comparison.png'), comparison)

        # 打印统计信息
        if self._teacher_vis_batch_count % 10 == 0:  # 每10个batch打印一次
            print(f"\n[Teacher Mask生成] Epoch {epoch}, Batch {batch_idx}")
            print(f"  t+1 mask占比: {mask_t1_np.mean():.2%}")
            print(f"  t mask占比（反投影后）: {mask_t0_np.mean():.2%}")
            print(f"  差异阈值: {threshold.item():.4f}")
            print(f"  可视化已保存到: {vis_dir}")

    #claude: 添加重置debug计数器的方法（在每个epoch开始时调用）
    def reset_debug_counter(self):
        """重置debug计数器和可视化计数器，在每个epoch开始时调用"""
        self.debug_batch_count = 0
        self._teacher_vis_batch_count = 0
        self._student_vis_batch_count = 0
        self._completion_sampling_vis_count = 0
        self._completion_rendering_vis_count = 0
        self._global_vis_batch_idx = 0  # 重置全局batch索引

    def sample_completion_gaussians(self, inputs, outputs, cam):
        """
        Step 3: 根据Student预测采样生成新增GS

        根据存在性先验和深度偏移分布，采样生成新增的Gaussian Splats

        Args:
            inputs: 输入数据
            outputs: 输出数据
            cam: 相机索引
        """
        if ('cam', cam) not in outputs or ('completion_outputs', 0, 0) not in outputs[('cam', cam)]:
            return

        completion_out = outputs[('cam', cam)][('completion_outputs', 0, 0)]
        existence_prior = completion_out['existence_prior']  # [B, 1, H, W]
        depth_offset_mean = completion_out['depth_offset_mean']  # [B, 1, H, W]
        depth_offset_std = completion_out['depth_offset_std']  # [B, 1, H, W]

        B, _, H, W = existence_prior.shape
        device = existence_prior.device

        # 1. 根据存在性先验决定是否生成GS
        if self.mode == 'train':
            # 训练时：使用概率采样（伯努利采样）
            sample_mask = torch.bernoulli(existence_prior)  # [B, 1, H, W]
        else:
            # 推理时：使用阈值
            sample_mask = (existence_prior > 0.5).float()  # [B, 1, H, W]

        # 2. 从深度偏移分布中采样
        # === 改进：简化depth预测 + 强制正约束 ===
        # 不再使用std（过于复杂），直接预测offset并强制为正
        depth_offset_raw = depth_offset_mean  # 只使用mean

        # 使用softplus确保offset为正（强制新GS在原GS后方）
        # softplus(x) = log(1 + exp(x))，输出始终>0
        depth_offset_sampled = torch.nn.functional.softplus(depth_offset_raw) + 0.1  # [B, 1, H, W]
        # +0.1确保最小偏移量，避免新GS与原GS重叠

        # 限制最大偏移（避免过远）
        depth_offset_sampled = torch.clamp(depth_offset_sampled, max=10.0)

        # 3. 计算新增GS的3D位置
        # 获取当前深度和相机参数
        depth_base = outputs[('cam', cam)][('depth', 0, 0)]  # [B, 1, H, W]
        K = inputs[('K', 0)][:, cam, :3, :3]  # [B, 3, 3]
        inv_K = inputs[('inv_K', 0)][:, cam, :3, :3]  # [B, 3, 3]

        # 创建像素网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=0).float()  # [3, H, W]
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]
        pixel_coords_flat = pixel_coords.view(B, 3, -1)  # [B, 3, H*W]

        # 反投影到3D空间（基础位置）
        depth_flat = depth_base.view(B, 1, -1)  # [B, 1, H*W]
        points3d_base = torch.bmm(inv_K, pixel_coords_flat) * depth_flat  # [B, 3, H*W]

        # 计算射线方向（归一化）
        ray_direction = torch.bmm(inv_K, pixel_coords_flat)  # [B, 3, H*W]
        ray_direction = ray_direction / (torch.norm(ray_direction, dim=1, keepdim=True) + 1e-7)

        # 沿射线方向偏移
        depth_offset_flat = depth_offset_sampled.view(B, 1, -1)  # [B, 1, H*W]
        points3d_completion = points3d_base + ray_direction * depth_offset_flat  # [B, 3, H*W]

        # Reshape回[B, 3, H, W]
        points3d_completion = points3d_completion.view(B, 3, H, W)

        # 4. 应用采样mask（只保留被采样的GS）
        sample_mask_3d = sample_mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]
        points3d_completion = points3d_completion * sample_mask_3d

        # 5. 保存到outputs
        outputs[('cam', cam)][('completion_xyz_sampled', 0, 0)] = points3d_completion
        outputs[('cam', cam)][('completion_sample_mask', 0, 0)] = sample_mask
        outputs[('cam', cam)][('completion_depth_offset', 0, 0)] = depth_offset_sampled

        # 6. 可视化（如果启用）
        # 可视化（每个epoch都输出，不限制batch数量）
        VISUALIZE = getattr(self, 'visualize_completion_sampling', False)
        VISUALIZE_BATCH_LIMIT = getattr(self, 'visualize_batch_limit', 20)
        if VISUALIZE and self.mode == 'train':
            if not hasattr(self, '_completion_sampling_vis_count'):
                self._completion_sampling_vis_count = 0
            if self._completion_sampling_vis_count < VISUALIZE_BATCH_LIMIT and cam == 3:
                self._visualize_completion_sampling(inputs, outputs, cam)
                self._completion_sampling_vis_count += 1

    def _render_with_completion(self, inputs, outputs, cam, novel_frame_id):
        """
        Step 4: 使用原GS + 新增GS渲染t+1视角

        Args:
            inputs: 输入数据
            outputs: 输出数据
            cam: 相机索引
            novel_frame_id: 目标帧ID（通常是1，表示t+1）

        Returns:
            rendered_img: 带补全的渲染图像 [B, 3, H, W]
        """
        # 获取采样的新增GS
        if ('completion_xyz_sampled', 0, 0) not in outputs[('cam', cam)]:
            # 如果没有采样结果，返回原始渲染
            return outputs[('cam', cam)][('gaussian_color', novel_frame_id, 0)]

        completion_xyz = outputs[('cam', cam)][('completion_xyz_sampled', 0, 0)]  # [B, 3, H, W]
        sample_mask = outputs[('cam', cam)][('completion_sample_mask', 0, 0)]  # [B, 1, H, W]

        # 获取补全GS的其他属性
        completion_out = outputs[('cam', cam)][('completion_outputs', 0, 0)]
        completion_rot = completion_out['rotation']  # [B, 4, H, W]
        completion_scale = completion_out['scale']  # [B, 3, H, W]
        completion_opacity = completion_out['opacity']  # [B, 1, H, W]
        completion_sh = completion_out['sh']  # [B, H*W, 1, 3, d_sh]

        # 应用sample_mask（只保留被采样的GS）
        B, _, H, W = completion_xyz.shape
        sample_mask_expanded = sample_mask.expand(-1, 3, -1, -1)
        completion_xyz_masked = completion_xyz * sample_mask_expanded

        # 合并原GS和新增GS
        # 简化策略：直接在原GS参数上叠加补全GS（使用sample_mask过滤）

        # 保存原始GS参数
        original_xyz_backup = outputs[('cam', cam)][('xyz', 0, 0)].clone()  # 添加xyz备份
        original_rot_backup = outputs[('cam', cam)][('rot_maps', 0, 0)].clone()
        original_scale_backup = outputs[('cam', cam)][('scale_maps', 0, 0)].clone()
        original_opacity_backup = outputs[('cam', cam)][('opacity_maps', 0, 0)].clone()
        original_sh_backup = outputs[('cam', cam)][('sh_maps', 0, 0)].clone()

        # 临时修改GS参数：在sample_mask位置叠加补全GS
        # === 关键修复：更新xyz（depth），确保新GS在正确位置 ===
        outputs[('cam', cam)][('xyz', 0, 0)] = original_xyz_backup + completion_xyz_masked  # 更新xyz
        outputs[('cam', cam)][('rot_maps', 0, 0)] = original_rot_backup + completion_rot * sample_mask.expand(-1, 4, -1, -1)
        outputs[('cam', cam)][('scale_maps', 0, 0)] = original_scale_backup + completion_scale * sample_mask.expand(-1, 3, -1, -1)
        outputs[('cam', cam)][('opacity_maps', 0, 0)] = original_opacity_backup + completion_opacity * sample_mask
        # sh需要特殊处理，因为shape是[B, H*W, 1, 3, d_sh]
        # 简化：只修改前面的参数，sh保持不变

        # 渲染
        rendered_with_completion = pts2render(
            inputs=inputs,
            outputs=outputs,
            cam_num=self.num_cams,
            novel_cam=cam,
            novel_frame_id=novel_frame_id,
            bg_color=[1.0, 1.0, 1.0],
            mode=self.novel_view_mode
        )

        # 恢复原始GS参数
        outputs[('cam', cam)][('xyz', 0, 0)] = original_xyz_backup  # 恢复xyz
        outputs[('cam', cam)][('rot_maps', 0, 0)] = original_rot_backup
        outputs[('cam', cam)][('scale_maps', 0, 0)] = original_scale_backup
        outputs[('cam', cam)][('opacity_maps', 0, 0)] = original_opacity_backup
        outputs[('cam', cam)][('sh_maps', 0, 0)] = original_sh_backup

        return rendered_with_completion

    def _visualize_completion_rendering(self, inputs, outputs, cam, novel_frame_id):
        """
        可视化Step 4的带补全渲染结果
        """
        import os
        import cv2
        import numpy as np

        vis_dir = os.path.join(self.log_dir, 'completion_rendering_vis')
        os.makedirs(vis_dir, exist_ok=True)

        epoch = getattr(self, 'epoch', 0)
        batch_idx = self._completion_rendering_vis_count

        # 获取图像
        img_gt = inputs[('color', novel_frame_id, 0)][0, cam].permute(1, 2, 0).detach().cpu().numpy()
        img_gt = (img_gt * 255).astype(np.uint8)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)

        img_base = outputs[('cam', cam)][('gaussian_color', novel_frame_id, 0)][0].permute(1, 2, 0).detach().cpu().numpy()
        img_base = (img_base * 255).astype(np.uint8)
        img_base = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)

        img_complete = outputs[('cam', cam)][('gaussian_color_with_completion', novel_frame_id, 0)][0].permute(1, 2, 0).detach().cpu().numpy()
        img_complete = (img_complete * 255).astype(np.uint8)
        img_complete = cv2.cvtColor(img_complete, cv2.COLOR_RGB2BGR)

        # 差异图
        diff_base = np.abs(img_gt.astype(float) - img_base.astype(float)).mean(axis=2)
        diff_complete = np.abs(img_gt.astype(float) - img_complete.astype(float)).mean(axis=2)

        diff_base_vis = np.clip(diff_base, 0, 255).astype(np.uint8)
        diff_complete_vis = np.clip(diff_complete, 0, 255).astype(np.uint8)

        diff_base_color = cv2.applyColorMap(diff_base_vis, cv2.COLORMAP_JET)
        diff_complete_color = cv2.applyColorMap(diff_complete_vis, cv2.COLORMAP_JET)

        # 并排对比
        comparison = np.hstack([img_gt, img_base, img_complete])

        # 保存
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_gt.png'), img_gt)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_base.png'), img_base)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_complete.png'), img_complete)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_diff_base.png'), diff_base_color)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_diff_complete.png'), diff_complete_color)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_comparison.png'), comparison)

        if batch_idx == 0:
            print(f"\n[Step 4渲染可视化] Epoch {epoch}")
            print(f"  基线渲染误差: {diff_base.mean():.2f}")
            print(f"  补全渲染误差: {diff_complete.mean():.2f}")
            print(f"  改进: {(diff_base.mean() - diff_complete.mean()):.2f}")
            print(f"  可视化已保存到: {vis_dir}")

    def _visualize_completion_sampling(self, inputs, outputs, cam=0):
        """
        可视化Step 3的采样结果
        """
        import os
        import cv2
        import numpy as np

        vis_dir = os.path.join(self.log_dir, 'completion_sampling_vis')
        os.makedirs(vis_dir, exist_ok=True)

        epoch = getattr(self, 'epoch', 0)
        batch_idx = self._completion_sampling_vis_count

        # 获取采样结果
        sample_mask = outputs[('cam', cam)][('completion_sample_mask', 0, 0)][0, 0].detach().cpu().numpy()
        depth_offset = outputs[('cam', cam)][('completion_depth_offset', 0, 0)][0, 0].detach().cpu().numpy()

        # 获取输入
        rgb_input = inputs[('color', 0, 0)][0, cam].permute(1, 2, 0).detach().cpu().numpy()
        rgb_input = (rgb_input * 255).astype(np.uint8)
        rgb_input = cv2.cvtColor(rgb_input, cv2.COLOR_RGB2BGR)

        # 采样mask可视化（绿色表示被采样的位置）
        mask_vis = np.zeros_like(rgb_input)
        mask_vis[sample_mask > 0.5] = [0, 255, 0]
        mask_overlay = cv2.addWeighted(rgb_input, 0.7, mask_vis, 0.3, 0)

        # 深度偏移可视化
        offset_vis = np.clip(depth_offset / 10.0 * 255, 0, 255).astype(np.uint8)
        offset_color = cv2.applyColorMap(offset_vis, cv2.COLORMAP_JET)

        # 保存
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_rgb.png'), rgb_input)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_sample_mask.png'), mask_overlay)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_depth_offset.png'), offset_color)

        if batch_idx == 0:
            print(f"\n[Step 3采样可视化] Epoch {epoch}")
            print(f"  采样点数量: {sample_mask.sum():.0f} / {sample_mask.size} ({sample_mask.mean():.2%})")
            print(f"  深度偏移范围: {depth_offset.min():.2f}~{depth_offset.max():.2f}m")
            print(f"  可视化已保存到: {vis_dir}")

    def compute_student_loss(self, inputs, outputs, cam):
        """
        Step 5: 计算Student网络的loss

        包含3个loss:
        1. 存在性先验loss (BCE)
        2. 局部改进loss (在teacher mask区域)
        3. 全局一致性loss
        """
        if ('cam', cam) not in outputs:
            return {}

        # 检查是否有必要的数据
        if ('completion_outputs', 0, 0) not in outputs[('cam', cam)]:
            return {}
        if ('teacher_mask_t0', 0, 0) not in outputs[('cam', cam)]:
            return {}

        # 获取Student预测
        completion_out = outputs[('cam', cam)][('completion_outputs', 0, 0)]
        existence_prior = completion_out['existence_prior']  # [B, 1, H, W]

        # 获取Teacher监督信号
        teacher_mask_t0 = outputs[('cam', cam)][('teacher_mask_t0', 0, 0)]  # [B, 1, H, W] 在t视角
        teacher_mask_t1 = outputs[('cam', cam)][('teacher_mask_t1', 0, 0)]  # [B, 1, H, W] 在t+1视角

        # Loss 1: 存在性先验loss (Head A的显式监督)
        loss_existence = torch.nn.functional.binary_cross_entropy(
            existence_prior, teacher_mask_t0, reduction='mean'
        )

        # Loss 2 & 3: 渲染loss (Head B的隐式监督)
        if ('gaussian_color_with_completion', 1, 0) in outputs[('cam', cam)]:
            img_t1_complete = outputs[('cam', cam)][('gaussian_color_with_completion', 1, 0)]  # [B, 3, H, W]
            img_t1_gt = inputs[('color', 1, 0)][:, cam, ...]  # [B, 3, H, W]

            # Loss 2: 局部改进loss (只在teacher_mask_t1标记的关键区域计算)
            diff_complete = torch.abs(img_t1_complete - img_t1_gt).mean(dim=1, keepdim=True)  # [B, 1, H, W]
            loss_local = (diff_complete * teacher_mask_t1).sum() / (teacher_mask_t1.sum() + 1e-6)

            # Loss 3: 全局一致性loss (整个图像)
            loss_global = torch.nn.functional.l1_loss(img_t1_complete, img_t1_gt)
        else:
            loss_local = torch.tensor(0.0, device=existence_prior.device)
            loss_global = torch.tensor(0.0, device=existence_prior.device)

        # 加权求和
        # === 改进：调整权重，鼓励在遮挡区域生成GS ===
        # 降低BCE权重（避免过度拟合teacher mask）
        # 提高render loss权重（关注渲染质量）
        w1 = getattr(self, 'student_existence_loss_weight', 0.3)  # 从1.0降到0.3
        w2 = getattr(self, 'student_local_loss_weight', 3.0)      # 从2.0升到3.0
        w3 = getattr(self, 'student_global_loss_weight', 1.0)     # 从0.5升到1.0

        total_loss = w1 * loss_existence + w2 * loss_local + w3 * loss_global

        return {
            'student/total': total_loss,
            'student/existence': loss_existence,
            'student/local': loss_local,
            'student/global': loss_global
        }

    def _visualize_student_heads_impl(self, inputs, outputs, cam=0):
        """
        可视化Student网络的Head A和Head B输出
        用于验证存在性先验和深度偏移分布是否合理
        """
        import os
        import cv2
        import numpy as np

        # 创建输出目录
        vis_dir = os.path.join(self.log_dir, 'student_head_vis')
        os.makedirs(vis_dir, exist_ok=True)

        epoch = getattr(self, 'epoch', 0)
        batch_idx = self._student_vis_batch_count

        # 获取Student输出
        if ('cam', cam) not in outputs or ('completion_outputs', 0, 0) not in outputs[('cam', cam)]:
            return

        completion_out = outputs[('cam', cam)][('completion_outputs', 0, 0)]
        existence_prior = completion_out['existence_prior'][0, 0].detach().cpu().numpy()  # [H, W]
        depth_offset_mean = completion_out['depth_offset_mean'][0, 0].detach().cpu().numpy()  # [H, W]
        depth_offset_std = completion_out['depth_offset_std'][0, 0].detach().cpu().numpy()  # [H, W]

        # 获取输入图像
        rgb_input = inputs[('color', 0, 0)][0, cam].permute(1, 2, 0).detach().cpu().numpy()
        rgb_input = (rgb_input * 255).astype(np.uint8)
        rgb_input = cv2.cvtColor(rgb_input, cv2.COLOR_RGB2BGR)

        # 获取深度（从outputs中，因为inputs中没有）
        depth_input = outputs[('cam', cam)][('depth', 0, 0)][0, 0].detach().cpu().numpy()
        depth_vis = np.clip(depth_input / 80.0 * 255, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 存在性先验可视化（热力图）- 改进：增强对比度，让高概率区域更明显
        # 使用HOT colormap（黑->红->黄->白），高值更明显
        existence_vis = (existence_prior * 255).astype(np.uint8)
        existence_color = cv2.applyColorMap(existence_vis, cv2.COLORMAP_HOT)

        # 额外生成一个增强版本，拉伸对比度
        existence_enhanced = np.clip((existence_prior - existence_prior.min()) / (existence_prior.max() - existence_prior.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
        existence_enhanced_color = cv2.applyColorMap(existence_enhanced, cv2.COLORMAP_HOT)

        # 深度偏移均值可视化
        offset_mean_vis = np.clip(depth_offset_mean / 10.0 * 255, 0, 255).astype(np.uint8)
        offset_mean_color = cv2.applyColorMap(offset_mean_vis, cv2.COLORMAP_JET)

        # 深度偏移标准差可视化
        offset_std_vis = np.clip(depth_offset_std / 5.0 * 255, 0, 255).astype(np.uint8)
        offset_std_color = cv2.applyColorMap(offset_std_vis, cv2.COLORMAP_JET)

        # 保存图像（包括增强版本的existence_prior）
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_rgb_input.png'), rgb_input)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_depth_input.png'), depth_color)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_existence_prior.png'), existence_color)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_existence_prior_enhanced.png'), existence_enhanced_color)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_depth_offset_mean.png'), offset_mean_color)
        cv2.imwrite(os.path.join(vis_dir, f'epoch{epoch:03d}_batch{batch_idx:03d}_cam{cam}_depth_offset_std.png'), offset_std_color)

        # 打印统计信息
        if self._student_vis_batch_count % 10 == 0:  # 每10个batch打印一次
            print(f"\n[Student Head可视化] Epoch {epoch}, Batch {batch_idx}")
            print(f"  存在性先验: {existence_prior.mean():.3f}±{existence_prior.std():.3f} (min={existence_prior.min():.3f}, max={existence_prior.max():.3f})")
            print(f"  深度偏移均值: {depth_offset_mean.mean():.3f}±{depth_offset_mean.std():.3f} m")
            print(f"  深度偏移标准差: {depth_offset_std.mean():.3f}±{depth_offset_std.std():.3f} m")
            print(f"  可视化已保存到: {vis_dir}")

        # 不再增加计数（使用全局计数器）

    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """
        #claude: 检查是否应该输出debug（仅第一个batch）
        self._should_debug_this_batch = (getattr(self, 'gs_completion_debug', False) and
                                         self.debug_batch_count == 0)

        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)

        supervised_mode = getattr(self, 'use_supervised_depth_net', False)

        # compute gaussian data (两种模式都需要后续渲染/输出)
        if self.gaussian:
            self.gs_net = self.models['gs_net']
            for cam in range(self.num_cams):
                self.get_gaussian_data(inputs, outputs, cam)

        # generate image / gaussian render
        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            if self.gaussian:
                self.pred_gaussian_imgs(inputs, outputs, cam)

        # === 生成教师真值（用于GS补全训练） ===
        # 注意：必须在pred_gaussian_imgs之后调用，因为需要t+1的GS渲染结果
        use_gs = getattr(self, 'use_gs_completion', False)
        if use_gs and self.mode == 'train':
            self.generate_teacher_ground_truth(inputs, outputs)

        # 统一路径：先计算自监督 / 重建 / 高斯等原有loss，再叠加监督深度（若开启）
        for cam in range(self.num_cams):
            cam_loss, loss_dict = self.losses(inputs, outputs, cam)

            # 叠加深度 L1（弱监督），保持原有loss存在
            if supervised_mode and ('depth' in inputs):
                ext_depth = inputs['depth']
                if ext_depth.dim() == 5:
                    ext_depth = ext_depth.squeeze(2)
                if ext_depth.dim() == 4 and ('depth', 0, 0) in outputs[('cam', cam)]:
                    depth_pred = outputs[('cam', cam)][('depth', 0, 0)]
                    depth_gt = F.interpolate(ext_depth[:, cam:cam+1, ...], [self.height, self.width], mode='bilinear', align_corners=False)
                    mask = (depth_gt > 0).float()
                    l1 = (torch.abs(depth_pred - depth_gt) * mask).sum() / (mask.sum() + 1e-6)
                    cam_loss = cam_loss + l1
                    loss_dict['depth_l1'] = l1.item()

            # === GS补全损失 ===
            if getattr(self, 'use_gs_completion', False) and self.mode == 'train':
                #claude: 使用should_debug标志（仅第一个batch的cam 3输出）
                debug = getattr(self, '_should_debug_this_batch', False) and cam == 3

                # 检查是否有教师真值和补全渲染结果
                if (('teacher_rgb_gt', 0, 0) in outputs[('cam', cam)] and
                    ('teacher_valid_mask', 0, 0) in outputs[('cam', cam)] and
                    ('gs_completion_rendered', 0, 0) in outputs[('cam', cam)]):

                    rgb_gt = outputs[('cam', cam)][('teacher_rgb_gt', 0, 0)]  # [B, 3, H, W]
                    valid_mask = outputs[('cam', cam)][('teacher_valid_mask', 0, 0)]  # [B, 1, H, W]
                    rgb_rendered = outputs[('cam', cam)][('gs_completion_rendered', 0, 0)]  # [B, 3, H, W]

                    # RGB重建损失（仅在有效区域计算）
                    completion_rgb_loss = (F.mse_loss(rgb_rendered, rgb_gt, reduction='none') * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                    # 可选：深度损失
                    if (('teacher_depth_gt', 0, 0) in outputs[('cam', cam)] and
                        ('gs_completion_depth_rendered', 0, 0) in outputs[('cam', cam)]):
                        depth_gt = outputs[('cam', cam)][('teacher_depth_gt', 0, 0)]
                        depth_rendered = outputs[('cam', cam)][('gs_completion_depth_rendered', 0, 0)]
                        completion_depth_loss = (F.mse_loss(depth_rendered, depth_gt, reduction='none') * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                        # 加权组合
                        completion_loss = completion_rgb_loss + getattr(self, 'completion_depth_weight', 0.1) * completion_depth_loss
                        loss_dict['completion_depth'] = completion_depth_loss.item()
                    else:
                        completion_loss = completion_rgb_loss

                    # 添加到总损失
                    cam_loss = cam_loss + getattr(self, 'completion_loss_weight', 1.0) * completion_loss
                    loss_dict['completion_rgb'] = completion_rgb_loss.item()
                    loss_dict['completion_total'] = completion_loss.item()

                    #claude: 添加调试输出（debug已包含cam==0条件）
                    if debug:
                        print(f"\n[补全损失计算] 相机{cam}:")
                        print(f"  RGB损失: {completion_rgb_loss.item():.4f}")
                        if 'completion_depth' in loss_dict:
                            print(f"  深度损失: {loss_dict['completion_depth']:.4f}")
                        print(f"  总损失: {completion_loss.item():.4f}")
                        print(f"  有效像素数: {valid_mask.sum().item()}")
                elif debug:
                    print(f"\n[补全损失计算] 相机{cam}:")
                    print(f"  ⚠️ 缺少必要的输出数据")
                    missing = []
                    if ('teacher_rgb_gt', 0, 0) not in outputs[('cam', cam)]:
                        missing.append("教师RGB真值")
                    if ('teacher_valid_mask', 0, 0) not in outputs[('cam', cam)]:
                        missing.append("有效区域mask")
                    if ('gs_completion_rendered', 0, 0) not in outputs[('cam', cam)]:
                        missing.append("补全渲染结果")
                    print(f"  缺少: {', '.join(missing)}")

            # === Step 5 & 6: Student不确定性建模损失 ===
            if getattr(self, 'use_gs_completion', False):
                student_losses = self.compute_student_loss(inputs, outputs, cam)
                if student_losses:
                    # 添加到总损失
                    cam_loss = cam_loss + student_losses['student/total']
                    # 记录各项loss
                    for key, value in student_losses.items():
                        loss_dict[key] = value.item() if torch.is_tensor(value) else value

            losses += cam_loss  
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams
        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))
        loss_mean['total_loss'] = losses

        #claude: 递增debug计数器
        if self.mode == 'train':
            self.debug_batch_count += 1

        return loss_mean

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """                  
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict) 

    def pred_gaussian_imgs(self, inputs, outputs, cam):
        if self.novel_view_mode == 'MF':
            outputs[('cam', cam)][('gaussian_color', 0, 0)] = \
                pts2render(inputs=inputs, 
                           outputs=outputs,
                           cam_num=self.num_cams, 
                           novel_cam=cam,
                           novel_frame_id=0,
                           bg_color=[1.0, 1.0, 1.0],
                           mode=self.novel_view_mode)
        elif self.novel_view_mode == 'SF':
            for novel_frame_id in self.frame_ids[1:]:
                # 原始渲染（不带补全）
                outputs[('cam', cam)][('gaussian_color', novel_frame_id, 0)] = \
                    pts2render(inputs=inputs,
                               outputs=outputs,
                               cam_num=self.num_cams,
                               novel_cam=cam,
                               novel_frame_id=novel_frame_id,
                               bg_color=[1.0, 1.0, 1.0],
                               mode=self.novel_view_mode)

                # Step 4: 带补全的渲染（如果启用了GS补全）
                if getattr(self, 'use_gs_completion', False) and novel_frame_id == 1:
                    # 只对t+1时刻进行补全渲染
                    if ('completion_sample_mask', 0, 0) in outputs[('cam', cam)]:
                        outputs[('cam', cam)][('gaussian_color_with_completion', novel_frame_id, 0)] = \
                            self._render_with_completion(inputs, outputs, cam, novel_frame_id)

                        # 可视化（每个epoch都输出，不限制batch数量）
                        VISUALIZE = getattr(self, 'visualize_completion_rendering', False)
                        VISUALIZE_BATCH_LIMIT = getattr(self, 'visualize_batch_limit', 20)
                        if VISUALIZE and self.mode == 'train':
                            if not hasattr(self, '_completion_rendering_vis_count'):
                                self._completion_rendering_vis_count = 0
                            if self._completion_rendering_vis_count < VISUALIZE_BATCH_LIMIT and cam == 3:
                                self._visualize_completion_rendering(inputs, outputs, cam, novel_frame_id)
                                self._completion_rendering_vis_count += 1

