"""
Depth Utilities for GS Completion

包含Depth Dilation等辅助函数，用于处理遮挡区域的深度预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#claude: 添加debug参数
def depth_dilation(depth, kernel_size=5, iterations=1, debug=False):
    """
    深度膨胀操作：使用MaxPool获取邻域最大深度

    目的：让被遮挡区域（depth=0）继承周围背景的深度值

    Args:
        depth: [B, 1, H, W] 深度图（包含0值的遮挡区域）
        kernel_size: int, 膨胀核大小（默认5）
        iterations: int, 膨胀迭代次数（默认1）
        debug: bool, 是否输出调试信息

    Returns:
        dilated_depth: [B, 1, H, W] 膨胀后的深度图

    Example:
        原始深度图：[5, 5, 0, 0, 10]  # 0是遮挡区域
             ↓ MaxPool(kernel=3)
        膨胀深度图：[5, 5, 5, 10, 10]  # 遮挡区域继承了邻域深度
    """
    dilated = depth.clone()

    # 创建MaxPool层
    padding = kernel_size // 2
    maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)

    # 迭代膨胀
    for _ in range(iterations):
        dilated = maxpool(dilated)

    #claude: 添加调试输出（仅第一个样本）
    if debug:
        # 只输出第一个样本的关键指标
        zero_ratio_before = (depth[0] == 0).float().mean().item()
        zero_ratio_after = (dilated[0] == 0).float().mean().item()
        depth_sample = depth[0][depth[0] > 0]
        dilated_sample = dilated[0][dilated[0] > 0]

        print("\n[深度膨胀] 样本0:")
        print(f"  零值占比: {zero_ratio_before:.2%} → {zero_ratio_after:.2%}")
        if len(dilated_sample) > 0:
            print(f"  深度范围: {dilated_sample.min().item():.2f}~{dilated_sample.max().item():.2f}m")

    return dilated


def merge_gaussian_params(gs_visible, gs_completion, validity_threshold=0.5):
    """
    合并可见区域和补全区域的GS球参数

    Args:
        gs_visible: dict, 可见区域的GS球参数
            - xyz: [B, N_visible, 3]
            - rotation: [B, N_visible, 4]
            - scale: [B, N_visible, 3]
            - opacity: [B, N_visible, 1]
            - sh: [B, N_visible, 1, 3, d_sh]

        gs_completion: dict, 补全区域的GS球参数
            - xyz: [B, H, W, 3]
            - rotation: [B, 4, H, W]
            - scale: [B, 3, H, W]
            - opacity: [B, 1, H, W]
            - sh: [B, H*W, 1, 3, d_sh]
            - validity: [B, 1, H, W]

        validity_threshold: float, 有效性阈值（默认0.5）

    Returns:
        gs_merged: dict, 合并后的GS球参数
            - xyz: [B, N_total, 3]
            - rotation: [B, N_total, 4]
            - scale: [B, N_total, 3]
            - opacity: [B, N_total, 1]
            - sh: [B, N_total, 1, 3, d_sh]
    """
    B = gs_visible['xyz'].shape[0]
    device = gs_visible['xyz'].device

    # 1. 处理补全区域的GS球
    # 根据validity筛选有效的GS球
    validity = gs_completion['validity']  # [B, 1, H, W]
    valid_mask = (validity > validity_threshold).squeeze(1)  # [B, H, W]

    # 展平空间维度
    H, W = valid_mask.shape[1], valid_mask.shape[2]

    # 提取有效的GS球参数
    xyz_completion = gs_completion['xyz']  # [B, H, W, 3]
    rotation_completion = gs_completion['rotation'].permute(0, 2, 3, 1)  # [B, H, W, 4]
    scale_completion = gs_completion['scale'].permute(0, 2, 3, 1)  # [B, H, W, 3]
    opacity_completion = gs_completion['opacity'].permute(0, 2, 3, 1)  # [B, H, W, 1]
    sh_completion = gs_completion['sh']  # [B, H*W, 1, 3, d_sh]

    # 重塑sh
    d_sh = sh_completion.shape[-1]
    sh_completion = sh_completion.view(B, H, W, 1, 3, d_sh)  # [B, H, W, 1, 3, d_sh]

    # 2. 对每个batch分别处理
    xyz_list = []
    rotation_list = []
    scale_list = []
    opacity_list = []
    sh_list = []

    for b in range(B):
        # 可见区域的GS球
        xyz_vis = gs_visible['xyz'][b]  # [N_visible, 3]
        rot_vis = gs_visible['rotation'][b]  # [N_visible, 4]
        scale_vis = gs_visible['scale'][b]  # [N_visible, 3]
        opacity_vis = gs_visible['opacity'][b]  # [N_visible, 1]
        sh_vis = gs_visible['sh'][b]  # [N_visible, 1, 3, d_sh]

        # 补全区域的有效GS球
        mask_b = valid_mask[b]  # [H, W]
        xyz_comp = xyz_completion[b][mask_b]  # [N_completion, 3]
        rot_comp = rotation_completion[b][mask_b]  # [N_completion, 4]
        scale_comp = scale_completion[b][mask_b]  # [N_completion, 3]
        opacity_comp = opacity_completion[b][mask_b]  # [N_completion, 1]
        sh_comp = sh_completion[b][mask_b]  # [N_completion, 1, 3, d_sh]

        # 合并
        xyz_merged = torch.cat([xyz_vis, xyz_comp], dim=0)  # [N_total, 3]
        rot_merged = torch.cat([rot_vis, rot_comp], dim=0)  # [N_total, 4]
        scale_merged = torch.cat([scale_vis, scale_comp], dim=0)  # [N_total, 3]
        opacity_merged = torch.cat([opacity_vis, opacity_comp], dim=0)  # [N_total, 1]
        sh_merged = torch.cat([sh_vis, sh_comp], dim=0)  # [N_total, 1, 3, d_sh]

        xyz_list.append(xyz_merged)
        rotation_list.append(rot_merged)
        scale_list.append(scale_merged)
        opacity_list.append(opacity_merged)
        sh_list.append(sh_merged)

    # 3. 堆叠成batch（需要padding到相同长度）
    # 找到最大长度
    max_len = max([x.shape[0] for x in xyz_list])

    # Padding
    xyz_padded = []
    rotation_padded = []
    scale_padded = []
    opacity_padded = []
    sh_padded = []

    for b in range(B):
        n = xyz_list[b].shape[0]
        if n < max_len:
            # Padding with zeros
            pad_len = max_len - n
            xyz_padded.append(torch.cat([xyz_list[b], torch.zeros(pad_len, 3, device=device)], dim=0))
            rotation_padded.append(torch.cat([rotation_list[b], torch.zeros(pad_len, 4, device=device)], dim=0))
            scale_padded.append(torch.cat([scale_list[b], torch.zeros(pad_len, 3, device=device)], dim=0))
            opacity_padded.append(torch.cat([opacity_list[b], torch.zeros(pad_len, 1, device=device)], dim=0))
            sh_padded.append(torch.cat([sh_list[b], torch.zeros(pad_len, 1, 3, d_sh, device=device)], dim=0))
        else:
            xyz_padded.append(xyz_list[b])
            rotation_padded.append(rotation_list[b])
            scale_padded.append(scale_list[b])
            opacity_padded.append(opacity_list[b])
            sh_padded.append(sh_list[b])

    # Stack
    gs_merged = {
        'xyz': torch.stack(xyz_padded, dim=0),  # [B, N_total, 3]
        'rotation': torch.stack(rotation_padded, dim=0),  # [B, N_total, 4]
        'scale': torch.stack(scale_padded, dim=0),  # [B, N_total, 3]
        'opacity': torch.stack(opacity_padded, dim=0),  # [B, N_total, 1]
        'sh': torch.stack(sh_padded, dim=0),  # [B, N_total, 1, 3, d_sh]
    }

    return gs_merged


#claude: 添加debug参数
def compute_completion_xyz(depth_dilated, xyz_offset, K, scale_factor=10.0, debug=False):
    """
    计算补全区域GS球的3D坐标

    Args:
        depth_dilated: [B, 1, H, W] 膨胀后的深度图
        xyz_offset: [B, 1, H, W] 深度偏移（范围[-1, 1]）
        K: [3, 3] 相机内参矩阵
        scale_factor: float, 偏移缩放因子（默认10.0）
        debug: bool, 是否输出调试信息

    Returns:
        xyz: [B, H, W, 3] 3D坐标
    """
    B, _, H, W = depth_dilated.shape
    device = depth_dilated.device

    # 计算实际深度
    depth_actual = depth_dilated + xyz_offset * scale_factor  # [B, 1, H, W]
    depth_actual = torch.clamp(depth_actual, min=0.1)  # 避免负深度

    # 生成像素坐标网格
    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy'
    )
    u = u.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # [B, 1, H, W]
    v = v.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # [B, 1, H, W]

    # 反投影到3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * depth_actual / fx
    y = (v - cy) * depth_actual / fy
    z = depth_actual

    xyz = torch.cat([x, y, z], dim=1)  # [B, 3, H, W]
    xyz = xyz.permute(0, 2, 3, 1)  # [B, H, W, 3]

    #claude: 添加调试输出（仅第一个样本）
    if debug:
        # 只输出第一个样本的关键指标
        offset_sample = xyz_offset[0]
        depth_sample = depth_actual[0]
        xyz_sample = xyz[0]

        print("\n[3D坐标计算] 样本0:")
        print(f"  深度偏移范围: {offset_sample.min().item():.3f}~{offset_sample.max().item():.3f}")
        print(f"  实际深度范围: {depth_sample.min().item():.2f}~{depth_sample.max().item():.2f}m")
        print(f"  XYZ范围: X=[{xyz_sample[..., 0].min().item():.2f},{xyz_sample[..., 0].max().item():.2f}] "
              f"Z=[{xyz_sample[..., 2].min().item():.2f},{xyz_sample[..., 2].max().item():.2f}]")

    return xyz
