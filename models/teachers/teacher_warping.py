"""
Teacher Warping Module: 前向 3D Warping + Z-Test

将t+1时刻的图像投影到t时刻，通过深度比较（Z-Test）识别解遮挡区域，生成训练真值。

核心思想（正确版本）：
- 将t+1的每个像素投影到t坐标系
- Z-Test: z_proj > depth_t + threshold
  → 如果t+1的点在t视角下更远（在t的前景后面），说明t+1看到了被t遮挡的背景（解遮挡）✅
  → 否则，t+1看到的是前景或相同物体（非解遮挡）❌

物理意义：
- t时刻：前景（depth_t=5m）遮挡了背景
- t+1时刻：视角变化，背景可见（depth_t1=10m）
- 投影：t+1的背景点投影到t坐标系，深度z_proj=10m
- 判断：z_proj (10m) > depth_t (5m) → 解遮挡 ✅
"""

import torch
import torch.nn.functional as F
import numpy as np


def depth_to_pointcloud(depth, K, pose=None):
    """
    将深度图转换为3D点云

    Args:
        depth: [H, W] 深度图
        K: [3, 3] 相机内参矩阵
        pose: [4, 4] 相机位姿（可选，用于变换到世界坐标系）

    Returns:
        xyz: [H, W, 3] 3D点云坐标
    """
    H, W = depth.shape

    # 生成像素坐标网格
    u, v = torch.meshgrid(
        torch.arange(W, device=depth.device, dtype=torch.float32),
        torch.arange(H, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )

    # 反投影到3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    xyz = torch.stack([x, y, z], dim=-1)  # [H, W, 3]

    # 如果提供了位姿，变换到世界坐标系
    if pose is not None:
        # 转换为齐次坐标
        ones = torch.ones_like(x).unsqueeze(-1)
        xyz_homo = torch.cat([xyz, ones], dim=-1)  # [H, W, 4]
        # 应用变换
        xyz_homo = xyz_homo @ pose.T  # [H, W, 4]
        xyz = xyz_homo[..., :3]  # [H, W, 3]

    return xyz


def project_pointcloud(xyz, K, pose=None):
    """
    将3D点云投影到图像平面

    Args:
        xyz: [H, W, 3] 或 [N, 3] 3D点云坐标
        K: [3, 3] 相机内参矩阵
        pose: [4, 4] 相机位姿（可选，用于从世界坐标系变换）

    Returns:
        uv: [..., 2] 像素坐标 (u, v)
        depth: [...] 深度值
    """
    original_shape = xyz.shape[:-1]

    # 如果提供了位姿，从世界坐标系变换到相机坐标系
    if pose is not None:
        # 转换为齐次坐标
        ones = torch.ones(*original_shape, 1, device=xyz.device, dtype=xyz.dtype)
        xyz_homo = torch.cat([xyz, ones], dim=-1)  # [..., 4]
        # 应用逆变换
        pose_inv = torch.inverse(pose)
        xyz_homo = xyz_homo @ pose_inv.T
        xyz = xyz_homo[..., :3]

    # 投影到图像平面
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    # 避免除零
    z = torch.clamp(z, min=1e-6)

    u = fx * x / z + cx
    v = fy * y / z + cy

    uv = torch.stack([u, v], dim=-1)
    depth = z

    return uv, depth


def compute_depth_edges(depth, threshold=1.0):
    """
    计算深度图的边缘（前景边界）

    Args:
        depth: [H, W] 深度图
        threshold: float, 深度梯度阈值（米）

    Returns:
        edge_mask: [H, W] 边缘mask（1表示边缘，0表示非边缘）
    """
    # 计算深度梯度（Sobel算子）
    # 水平梯度
    grad_x = torch.zeros_like(depth)
    grad_x[:, :-1] = torch.abs(depth[:, 1:] - depth[:, :-1])

    # 垂直梯度
    grad_y = torch.zeros_like(depth)
    grad_y[:-1, :] = torch.abs(depth[1:, :] - depth[:-1, :])

    # 梯度幅值
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # 边缘mask：梯度大于阈值的区域
    edge_mask = (grad_mag > threshold).float()

    return edge_mask


#claude: 添加debug参数
def warp_future_to_current(img_t, depth_t, img_t1, depth_t1, pose_t_to_t1, K, threshold=2.0, debug=False):
    """
    前向 Warping + Z-Test: 将t+1的每个像素投影到t，识别解遮挡区域

    核心逻辑（正确版本）：
    - 将t+1的每个像素投影到t坐标系
    - Z-Test: z_proj > depth_t + threshold
    - 如果t+1的点在t视角下更远，说明t+1看到了被t遮挡的背景（解遮挡）

    Args:
        img_t: [H, W, 3] 或 [B, 3, H, W] RGB图像 t时刻
        depth_t: [H, W] 或 [B, 1, H, W] 深度图 t时刻
        img_t1: [H, W, 3] 或 [B, 3, H, W] RGB图像 t+1时刻
        depth_t1: [H, W] 或 [B, 1, H, W] 深度图 t+1时刻
        pose_t_to_t1: [4, 4] 或 [B, 4, 4] 从t到t+1的变换矩阵
        K: [3, 3] 相机内参矩阵
        threshold: float, Z-Test的深度阈值（默认0.5m，增加以减少噪声）
        debug: bool, 是否输出调试信息

    Returns:
        rgb_gt: [H, W, 3] 或 [B, 3, H, W] 补全区域的RGB真值
        depth_gt: [H, W] 或 [B, 1, H, W] 补全区域的深度真值
        valid_mask: [H, W] 或 [B, 1, H, W] 有效区域mask（解遮挡区域）
    """
    # 处理batch维度
    if img_t.dim() == 4:  # [B, 3, H, W]
        batch_mode = True
        B, _, H, W = img_t.shape
        # 转换为 [B, H, W, 3]
        img_t = img_t.permute(0, 2, 3, 1)
        img_t1 = img_t1.permute(0, 2, 3, 1)
        depth_t = depth_t.squeeze(1)  # [B, H, W]
        depth_t1 = depth_t1.squeeze(1)  # [B, H, W]
    else:  # [H, W, 3]
        batch_mode = False
        H, W, _ = img_t.shape
        # 添加batch维度
        img_t = img_t.unsqueeze(0)
        img_t1 = img_t1.unsqueeze(0)
        depth_t = depth_t.unsqueeze(0)
        depth_t1 = depth_t1.unsqueeze(0)
        pose_t_to_t1 = pose_t_to_t1.unsqueeze(0)
        B = 1

    # 初始化输出
    rgb_gt_list = []
    depth_gt_list = []
    valid_mask_list = []

    for b in range(B):
        # 1. 将t+1的深度图转换为3D点云（在t+1坐标系）
        xyz_t1 = depth_to_pointcloud(depth_t1[b], K)  # [H, W, 3]

        # 2. 变换到t坐标系
        pose_inv = torch.inverse(pose_t_to_t1[b])  # t+1 -> t
        ones = torch.ones(H, W, 1, device=xyz_t1.device, dtype=xyz_t1.dtype)
        xyz_t1_homo = torch.cat([xyz_t1, ones], dim=-1)  # [H, W, 4]
        xyz_t_homo = xyz_t1_homo @ pose_inv.T  # [H, W, 4]
        xyz_t = xyz_t_homo[..., :3]  # [H, W, 3]

        # 3. 投影到t的图像平面
        uv_proj, z_proj = project_pointcloud(xyz_t, K)  # [H, W, 2], [H, W]

        # 4. Z-Test: 比较投影深度和t时刻的深度
        # z_proj > depth_t + threshold: 解遮挡区域（t+1看到了t看不到的背景）
        # 首先需要从depth_t采样对应位置的深度
        u_norm = 2.0 * uv_proj[..., 0] / (W - 1) - 1.0
        v_norm = 2.0 * uv_proj[..., 1] / (H - 1) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        # 从depth_t采样投影位置的深度
        depth_t_sampled = F.grid_sample(
            depth_t[b].unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0).squeeze(0)  # [H, W]

        # Z-Test判断
        valid_mask = (z_proj > depth_t_sampled + threshold).float()  # [H, W]

        # 过滤掉投影到图像外的点
        valid_mask = valid_mask * (uv_proj[..., 0] >= 0) * (uv_proj[..., 0] < W)
        valid_mask = valid_mask * (uv_proj[..., 1] >= 0) * (uv_proj[..., 1] < H)

        # 过滤掉深度无效的点
        valid_mask = valid_mask * (depth_t1[b] > 0) * (z_proj > 0) * (depth_t_sampled > 0)

        # 额外过滤：只保留t时刻的边缘区域（前景边界）
        # 计算depth_t的边缘
        edge_mask_t = compute_depth_edges(depth_t[b], threshold=1.0)  # 1m深度变化
        # 从edge_mask_t采样到投影位置
        edge_mask_sampled = F.grid_sample(
            edge_mask_t.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0).squeeze(0)  # [H, W]
        # 只保留边缘区域的解遮挡
        valid_mask = valid_mask * (edge_mask_sampled > 0.5)

        # 5. 使用scatter操作将t+1的RGB和深度投影到t的位置
        # 初始化输出
        rgb_gt = torch.zeros_like(img_t1[b])  # [H, W, 3]
        depth_gt = torch.zeros_like(depth_t1[b])  # [H, W]
        mask_gt = torch.zeros_like(depth_t1[b])  # [H, W]

        # 将有效的t+1像素投影到t
        # 这里使用简单的scatter操作（后续可以优化为保留最近深度）
        valid_indices = valid_mask > 0
        if valid_indices.sum() > 0:
            # 获取投影坐标（四舍五入到整数）
            u_int = torch.clamp(uv_proj[..., 0].round().long(), 0, W - 1)
            v_int = torch.clamp(uv_proj[..., 1].round().long(), 0, H - 1)

            # 只处理有效的像素
            valid_u = u_int[valid_indices]
            valid_v = v_int[valid_indices]
            valid_rgb = img_t1[b][valid_indices]  # [N, 3]
            valid_depth = z_proj[valid_indices]  # [N]

            # Scatter到输出（简单版本：直接覆盖，不处理冲突）
            rgb_gt[valid_v, valid_u] = valid_rgb
            depth_gt[valid_v, valid_u] = valid_depth
            mask_gt[valid_v, valid_u] = 1.0

        rgb_gt_list.append(rgb_gt)
        depth_gt_list.append(depth_gt)
        valid_mask_list.append(mask_gt)

    # 合并batch
    rgb_gt = torch.stack(rgb_gt_list, dim=0)  # [B, H, W, 3]
    depth_gt = torch.stack(depth_gt_list, dim=0)  # [B, H, W]
    valid_mask = torch.stack(valid_mask_list, dim=0)  # [B, H, W]

    # 转换回原始格式
    if batch_mode:
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)  # [B, 3, H, W]
        depth_gt = depth_gt.unsqueeze(1)  # [B, 1, H, W]
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
    else:
        rgb_gt = rgb_gt.squeeze(0)  # [H, W, 3]
        depth_gt = depth_gt.squeeze(0)  # [H, W]
        valid_mask = valid_mask.squeeze(0)  # [H, W]

    #claude: 添加调试输出（仅第一个样本）
    if debug:
        # 只输出第一个样本的关键指标
        if batch_mode:
            valid_ratio = valid_mask[0].float().mean().item()
            valid_pixels = valid_mask[0].sum().item()
            depth_gt_sample = depth_gt[0][depth_gt[0] > 0]
        else:
            valid_ratio = valid_mask.float().mean().item()
            valid_pixels = valid_mask.sum().item()
            depth_gt_sample = depth_gt[depth_gt > 0]

        print("\n[教师模型-3D投影] 样本0:")
        print(f"  解遮挡区域占比: {valid_ratio:.2%} ({valid_pixels}像素)")
        if len(depth_gt_sample) > 0:
            print(f"  补全深度范围: {depth_gt_sample.min().item():.2f}~{depth_gt_sample.max().item():.2f}m")

    return rgb_gt, depth_gt, valid_mask
