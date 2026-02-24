"""
教师端 3D Warping + Z-Test 实现

功能：将未来帧 t+1 warp 回当前帧 t，通过深度比较识别解遮挡区域
"""

import torch
import torch.nn.functional as F


def warp_future_to_current(I_t, D_t, I_t1, D_t1, P_t_to_t1, K, threshold=0.1):
    """
    将 t+1 帧 warp 回 t 帧，生成训练真值

    Args:
        I_t: 当前帧图像 [B, 3, H, W]
        D_t: 当前帧深度 [B, 1, H, W]
        I_t1: 未来帧图像 [B, 3, H, W]
        D_t1: 未来帧深度 [B, 1, H, W]
        P_t_to_t1: 从 t 到 t+1 的位姿变换矩阵 [B, 4, 4]
        K: 相机内参 [B, 3, 3]
        threshold: Z-Test 阈值（米）

    Returns:
        RGB_GT: RGB 真值 [B, 3, H, W]
        Depth_GT: 深度真值 [B, 1, H, W]
        M_valid: 有效性 mask [B, 1, H, W]
    """
    B, _, H, W = I_t.shape
    device = I_t.device

    # 1. 生成像素网格
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    # [H, W, 2] -> [B, H, W, 2]
    grid = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # 2. 反投影 t+1 的深度图到 3D 点云
    # [B, H, W, 2] -> [B, H*W, 2]
    pixels = grid.reshape(B, -1, 2)
    depths = D_t1.squeeze(1).reshape(B, -1, 1)  # [B, H*W, 1]

    # 反投影公式：xyz_cam = K^{-1} @ [u*d, v*d, d]
    # 构造齐次坐标 [u, v, 1]
    pixels_homo = torch.cat([pixels, torch.ones(B, H*W, 1, device=device)], dim=-1)  # [B, H*W, 3]

    # K^{-1}
    K_inv = torch.inverse(K)  # [B, 3, 3]

    # xyz_cam = K^{-1} @ (pixels_homo * depth)^T
    xyz_t1_cam = torch.bmm(K_inv, (pixels_homo * depths).transpose(1, 2))  # [B, 3, H*W]
    xyz_t1_cam = xyz_t1_cam.transpose(1, 2)  # [B, H*W, 3]

    # 3. 变换到 t 坐标系
    # xyz_t = P_t_to_t1 @ [xyz_t1; 1]
    xyz_t1_homo = torch.cat([xyz_t1_cam, torch.ones(B, H*W, 1, device=device)], dim=-1)  # [B, H*W, 4]
    xyz_t_homo = torch.bmm(P_t_to_t1, xyz_t1_homo.transpose(1, 2))  # [B, 4, H*W]
    xyz_t_cam = xyz_t_homo[:, :3, :].transpose(1, 2)  # [B, H*W, 3]

    # 4. 投影到 t 图像平面
    # [u, v, z] = K @ xyz_t
    uv_homo = torch.bmm(K, xyz_t_cam.transpose(1, 2))  # [B, 3, H*W]
    Z_proj = uv_homo[:, 2, :]  # [B, H*W] 投影深度
    uv = uv_homo[:, :2, :] / (Z_proj.unsqueeze(1) + 1e-8)  # [B, 2, H*W]

    # 5. 归一化 uv 到 [-1, 1]（用于 grid_sample）
    uv_norm = torch.zeros_like(uv)
    uv_norm[:, 0, :] = 2.0 * uv[:, 0, :] / (W - 1) - 1.0  # x
    uv_norm[:, 1, :] = 2.0 * uv[:, 1, :] / (H - 1) - 1.0  # y

    # Reshape 为 grid 格式 [B, H, W, 2]
    grid_warp = uv_norm.reshape(B, 2, H, W).permute(0, 2, 3, 1)  # [B, H, W, 2]

    # 6. Warp I_t1 到 t 视角
    I_t1_warped = F.grid_sample(
        I_t1,
        grid_warp,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # 7. Z-Test：比较投影深度和当前深度
    Z_proj_map = Z_proj.reshape(B, 1, H, W)  # [B, 1, H, W]

    # 解遮挡区域：Z_proj > D_t + threshold
    # 说明 t+1 看到了 t 看不到的背景
    M_valid = (Z_proj_map > D_t + threshold) & (Z_proj_map > 0) & (D_t > 0)

    # 8. 生成真值
    RGB_GT = torch.zeros_like(I_t)
    Depth_GT = torch.zeros_like(D_t)

    # 只在有效区域填充真值
    RGB_GT = I_t1_warped * M_valid.float()
    Depth_GT = Z_proj_map * M_valid.float()

    return RGB_GT, Depth_GT, M_valid


def depth2pc(depth, extrinsic, intrinsic):
    """
    将深度图转换为点云（世界坐标系）

    Args:
        depth: 深度图 [B, 1, H, W]
        extrinsic: 外参矩阵 [B, 4, 4]（相机到世界）
        intrinsic: 内参矩阵 [B, 3, 3]

    Returns:
        xyz: 点云 [B, H*W, 3]
    """
    B, C, H, W = depth.shape
    device = depth.device
    depth = depth[:, 0, :, :]  # [B, H, W]

    # 生成像素网格
    y, x = torch.meshgrid(
        torch.linspace(0.5, H-0.5, H, device=device),
        torch.linspace(0.5, W-0.5, W, device=device),
        indexing='ij'
    )
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 3]

    # 反投影
    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)  # [B, 3, H*W]

    # 变换到世界坐标系
    rot = extrinsic[:, :3, :3]  # [B, 3, 3]
    trans = extrinsic[:, :3, 3:]  # [B, 3, 1]

    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)  # [B, H*W, 3]
