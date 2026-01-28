import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

from utils import Logger

from lpips import LPIPS
from jaxtyping import Float, UInt8
from skimage.metrics import structural_similarity
from einops import reduce

from PIL import Image
from pathlib import Path
from einops import rearrange, repeat
from typing import Union
import numpy as np

from tqdm import tqdm

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

class DrivingForwardTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.lpips = LPIPS(net="vgg").to(device)

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        for self.epoch in range(self.num_epochs):
                
            self.train(model, train_dataloader, start_time)
            
            # save model after each epoch using rank 0 gpu 
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-'*110) 
                
            if self.ddp_enable:
                dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        pbar = tqdm(total=len(data_loader), desc='training on epoch {}'.format(self.epoch), mininterval=100)
        for batch_idx, inputs in enumerate(data_loader):         
            before_op_time = time.time()
            model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model.process_batch(inputs, self.rank)
            losses['total_loss'].backward()
            model.optimizer.step()

            if self.rank == 0: 
                self.logger.update(
                    'train', 
                    self.epoch, 
                    self.world_size,
                    batch_idx, 
                    self.step,
                    start_time,
                    before_op_time, 
                    inputs,
                    outputs,
                    losses
                )

                if self.logger.is_checkpoint(self.step):
                    self.validate(model)

            self.step += 1
            pbar.update(1)

        pbar.close()
        model.lr_scheduler.step()
        
    @torch.no_grad()
    def validate(self, model, vis_results=False):
        """
        This function validates models on the validation dataset to monitor training process.
        """
        val_dataloader = model.val_dataloader()
        val_iter = iter(val_dataloader)
        
        # Ensure the model is in validation mode
        model.set_val()

        avg_reconstruction_metric = defaultdict(float)

        inputs = next(val_iter)
        outputs, _ = model.process_batch(inputs, self.rank)
            
        psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)
        depth_l1 = self.compute_depth_l1(inputs, outputs)

        avg_reconstruction_metric['psnr'] += psnr   
        avg_reconstruction_metric['ssim'] += ssim
        avg_reconstruction_metric['lpips'] += lpips
        if depth_l1 is not None:
            avg_reconstruction_metric['depth_l1'] = depth_l1

        print('Validation reconstruction result...\n')
        print(f"\n{inputs['token'][0]}")
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

        # Set the model back to training mode
        model.set_train()

    @torch.no_grad()
    def evaluate(self, model):
        """
        This function evaluates models on validation dataset of samples with context.
        """
        eval_dataloader = model.eval_dataloader()

        # load model
        model.load_weights()
        model.set_eval()

        avg_reconstruction_metric = defaultdict(float)
        depth_l1_sum = 0.0
        depth_l1_count = 0

        count = 0

        process = tqdm(eval_dataloader)
        for batch_idx, inputs in enumerate(process):
            outputs, _ = model.process_batch(inputs, self.rank)

            # --- 【新增】: 导出 PLY 用于可视化调试 ---
            # 建议只导出一个 batch 或者特定的 token，防止硬盘写满
            # 这里默认全部导出，你可以加 if batch_idx == 0: 来限制
            # save_dir = Path(self.save_path) / "debug_ply"
            # self.export_ply(inputs, outputs, save_dir)
            # ----------------------------------------
            
            psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)
            depth_l1 = self.compute_depth_l1(inputs, outputs)

            avg_reconstruction_metric['psnr'] += psnr   
            avg_reconstruction_metric['ssim'] += ssim
            avg_reconstruction_metric['lpips'] += lpips
            count += 1
            if depth_l1 is not None:
                depth_l1_sum += depth_l1
                depth_l1_count += 1

            process.set_description(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")

            print(f"\n{inputs['token'][0]}")
            print(f"avg PSNR: {avg_reconstruction_metric['psnr']/count:.4f}, avg SSIM: {avg_reconstruction_metric['ssim']/count:.4f}, avg LPIPS: {avg_reconstruction_metric['lpips']/count:.4f}")
            
        avg_reconstruction_metric['psnr'] /= len(eval_dataloader)
        avg_reconstruction_metric['ssim'] /= len(eval_dataloader)
        avg_reconstruction_metric['lpips'] /= len(eval_dataloader)
        if depth_l1_count > 0:
            avg_reconstruction_metric['depth_l1'] = depth_l1_sum / depth_l1_count

        print('Evaluation reconstruction result...\n')
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

# =========================================================================
    # 【修改点 2】: 新增 PLY 导出函数
    # =========================================================================
    def export_ply(self, inputs, outputs, save_dir):
        """
        导出调试用的 PLY 文件。
        【修正点】:
        1. 颜色: 执行 SH -> RGB 的转换，解决"没颜色"问题。
        2. 坐标: 强制重新计算 Camera-Space 坐标，解决"乱七八糟/找不到点"问题。
        3. 格式: 严格对齐数据长度。
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        token = inputs['token'][0]
        frame_id = 0 
        
        # 获取所有相机 ID
        cam_keys = [k for k in outputs.keys() if isinstance(k, tuple) and k[0] == 'cam']
        cam_ids = sorted(list(set([k[1] for k in cam_keys])))

        for cam in cam_ids:
            # 必须要有 depth 才能重算坐标
            required_keys = [('depth', frame_id, 0), ('scale_maps', frame_id, 0), ('rot_maps', frame_id, 0), ('opacity_maps', frame_id, 0), ('sh_maps', frame_id, 0)]
            cam_out = outputs[('cam', cam)]
            if any(k not in cam_out for k in required_keys):
                continue

            # =========================================================
            # 1. 重算坐标 (Camera Space Re-projection)
            # =========================================================
            # 这样可以保证点云一定在原点附近，形状清晰，不受外参影响
            depth_map = cam_out[('depth', frame_id, 0)] # [B, 1, H, W]
            B, _, H, W = depth_map.shape
            depth = depth_map[0, 0].detach().cpu().numpy() # [H, W]
            
            # 获取内参
            K = inputs[('K', 0)][0, cam].cpu().numpy() # [3, 3]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # 生成网格
            grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            
            # 反投影公式: Z=depth, X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy
            # 注意: 这里的 Y 轴方向取决于数据集定义，通常 Image Y down -> Camera Y down
            x_flat = (grid_x.flatten() - cx) * depth.flatten() / fx
            y_flat = (grid_y.flatten() - cy) * depth.flatten() / fy
            z_flat = depth.flatten()
            
            xyz = np.stack([x_flat, y_flat, z_flat], axis=1) # [N, 3]

            # =========================================================
            # 2. 处理 Scale, Rotation, Opacity
            # =========================================================
            # Scale
            scale_map = cam_out[('scale_maps', frame_id, 0)] # [B, 3, H, W]
            scale = scale_map[0].permute(1, 2, 0).reshape(-1, 3).detach().cpu().numpy()
            # 存 Log Scale (3DGS viewer 标准)
            scale = np.log(np.clip(scale, 1e-8, 1e8))

            # Rotation
            rot_map = cam_out[('rot_maps', frame_id, 0)] # [B, 4, H, W]
            rot = rot_map[0].permute(1, 2, 0).reshape(-1, 4).detach().cpu().numpy()
            # 归一化
            rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9)

            # Opacity
            op_map = cam_out[('opacity_maps', frame_id, 0)] # [B, 1, H, W]
            opacity = op_map[0].permute(1, 2, 0).reshape(-1, 1).detach().cpu().numpy()

            # =========================================================
            # 3. 处理颜色 (SH -> RGB) 【核心修复】
            # =========================================================
            sh_tensor = cam_out[('sh_maps', frame_id, 0)] 
            
            # 提取 DC 分量 (第0个系数)
            if sh_tensor.dim() == 6: # [B, N, 1, 1, 3, D_sh]
                f_dc = sh_tensor[..., 0].reshape(-1, 3).detach().cpu().numpy()
            elif sh_tensor.dim() == 5:
                f_dc = sh_tensor[..., 0].reshape(-1, 3).detach().cpu().numpy()
            elif sh_tensor.dim() == 4: # [B, N, 3, D_sh] or [B, 3, H, W]
                if sh_tensor.shape[1] == 3 and sh_tensor.shape[-1] != 3: # Image map [B, 3, H, W]
                     f_dc = sh_tensor[0].permute(1, 2, 0).reshape(-1, 3).detach().cpu().numpy()
                else:
                     f_dc = sh_tensor[..., 0].reshape(-1, 3).detach().cpu().numpy()
            else:
                # Fallback
                f_dc = sh_tensor.flatten()[:xyz.shape[0]*3].reshape(-1, 3).detach().cpu().numpy()

            # SH 0阶系数转 RGB 公式: RGB = 0.282 * SH + 0.5
            SH_C0 = 0.28209479177387814
            rgb = f_dc * SH_C0 + 0.5
            # Clip 到 0-1 范围，防止颜色溢出变成怪色
            rgb = np.clip(rgb, 0.0, 1.0)
            
            # SuperSplat/CloudCompare 读取 f_dc 时通常直接读取数值
            # 我们把算好的 RGB 存回去，这样打开就是彩色了
            f_dc_final = rgb

            # =========================================================
            # 4. 对齐与保存
            # =========================================================
            # 强行截断到相同长度 (以防 reshape 过程中有微小差异)
            min_len = min(xyz.shape[0], scale.shape[0], rot.shape[0], opacity.shape[0], f_dc_final.shape[0])
            
            xyz = xyz[:min_len]
            scale = scale[:min_len]
            rot = rot[:min_len]
            opacity = opacity[:min_len]
            f_dc_final = f_dc_final[:min_len]

            # 剔除透明点 (Opacity < 0.05)
            mask = (opacity > 0.05).squeeze()
            if mask.sum() == 0:
                print(f"[Export] Skip cam {cam} (empty)")
                continue
                
            xyz = xyz[mask]
            scale = scale[mask]
            rot = rot[mask]
            opacity = opacity[mask]
            f_dc_final = f_dc_final[mask]

            # 构建 PLY
            dtype_full = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            elements['x'] = xyz[:, 0]
            elements['y'] = xyz[:, 1]
            elements['z'] = xyz[:, 2]
            elements['nx'] = 0
            elements['ny'] = 0
            elements['nz'] = 0
            # 存 RGB 颜色
            elements['f_dc_0'] = f_dc_final[:, 0]
            elements['f_dc_1'] = f_dc_final[:, 1]
            elements['f_dc_2'] = f_dc_final[:, 2]
            elements['opacity'] = opacity[:, 0]
            elements['scale_0'] = scale[:, 0]
            elements['scale_1'] = scale[:, 1]
            elements['scale_2'] = scale[:, 2]
            elements['rot_0'] = rot[:, 0]
            elements['rot_1'] = rot[:, 1]
            elements['rot_2'] = rot[:, 2]
            elements['rot_3'] = rot[:, 3]

            out_name = f"{token}_cam{cam}_camera_space.ply"
            out_path = save_dir / out_name
            
            header = f"""ply
format binary_little_endian 1.0
element vertex {xyz.shape[0]}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
            with open(out_path, 'wb') as f:
                f.write(header.encode('utf-8'))
                elements.tofile(f)
            
            if self.rank == 0 and cam == cam_ids[0]:
                print(f"[Export] Saved PLY to {out_path}")

    def save_image(
        self,
        image: FloatImage,
        path: Union[Path, str],
    ) -> None:
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save the image.
        Image.fromarray(self.prep_image(image)).save(path)


    def prep_image(self, image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
        # Handle batched images.
        if image.ndim == 4:
            image = rearrange(image, "b c h w -> c h (b w)")

        # Handle single-channel images.
        if image.ndim == 2:
            image = rearrange(image, "h w -> () h w")

        # Ensure that there are 3 or 4 channels.
        channel, _, _ = image.shape
        if channel == 1:
            image = repeat(image, "() h w -> c h w", c=3)
        assert image.shape[0] in (3, 4)

        image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
        return rearrange(image, "c h w -> h w c").cpu().numpy()

    @torch.no_grad()
    def compute_reconstruction_metrics(self, inputs, outputs):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0
        if self.novel_view_mode == 'SF':
            frame_id = 1
        elif self.novel_view_mode == 'MF':
            frame_id = 0
        else:
            raise ValueError(f"Invalid novel view mode: {self.novel_view_mode}")
        for cam in range(self.num_cams):
            rgb_gt = inputs[('color', frame_id, 0)][:, cam, ...]
            image = outputs[('cam', cam)][('gaussian_color', frame_id, 0)]
            psnr += self.compute_psnr(rgb_gt, image).mean()
            ssim += self.compute_ssim(rgb_gt, image).mean()
            lpips += self.compute_lpips(rgb_gt, image).mean()
            if self.save_images:
                assert self.eval_batch_size == 1
                if self.novel_view_mode == 'SF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_0_gt.png")
                elif self.novel_view_mode == 'MF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', -1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_prev_gt.png")
                    self.save_image(inputs[('color', 1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_next_gt.png")
        psnr /= self.num_cams
        ssim /= self.num_cams
        lpips /= self.num_cams
        return psnr, ssim, lpips

    def compute_depth_l1(self, inputs, outputs):
        """Compute mean L1 depth error if GT depth available; else return None."""
        if 'depth' not in inputs:
            return None
        ext_depth = inputs['depth']
        if ext_depth.dim() == 5:
            ext_depth = ext_depth.squeeze(2)
        num_cams = ext_depth.shape[1]
        total = 0.0
        count = 0
        for cam in range(num_cams):
            key = ('cam', cam)
            if key not in outputs:
                continue
            if ('depth', 0, 0) not in outputs[key]:
                continue
            depth_pred = outputs[key][('depth', 0, 0)]
            depth_gt = F.interpolate(ext_depth[:, cam:cam+1, ...], depth_pred.shape[-2:], mode='bilinear', align_corners=False)
            mask = (depth_gt > 0).float()
            l1 = (torch.abs(depth_pred - depth_gt) * mask).sum() / (mask.sum() + 1e-6)
            total += l1
            count += 1
        if count == 0:
            return None
        return total / count
    
    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
