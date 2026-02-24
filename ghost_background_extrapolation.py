#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据深度图 + RGB 生成 3D Gaussian Splatting 格式的 PLY，
并对遮挡区域做受限的几何外推。
"""

import argparse
import os
import sys
from typing import Optional, Tuple
import cv2
import numpy as np

# ================= 核心工具函数 =================

def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """
    将 [0, 255] 的 RGB 转换为 0阶 Spherical Harmonics (f_dc)
    公式: (RGB/255 - 0.5) / 0.28209479177387814
    """
    C0 = 0.28209479177387814
    return (rgb.astype(np.float32) / 255.0 - 0.5) / C0

def save_ply_as_gaussian(filename: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    保存为标准的 3DGS PLY 格式，以此骗过可视化软件
    points: (N, 3) XYZ
    colors: (N, 3) RGB uint8
    """
    N = len(points)
    print(f"[info] Converting {N} points to 3D Gaussians...")

    # 1. 准备基础属性
    xyz = points.astype(np.float32)
    
    # 2. 准备颜色 (转换到 SH DC 分量)
    # GS 查看器通常读取 f_dc_0, f_dc_1, f_dc_2
    f_dc = rgb_to_sh(colors)

    # 3. 准备 Opacity (不透明度)
    # 通常存储为 logit 形式，或者有些查看器直接读 float。
    # 这里我们模拟高斯的不透明度。为了保险，给一个较高的值。
    # Logit(0.99) ≈ 4.6
    opacities = np.ones((N, 1), dtype=np.float32) * 4.6

    # 4. 准备 Scale (缩放)
    # 存储为 log 形式。我们希望球很小，类似点云。
    # log(0.01) ≈ -4.6
    scales = np.ones((N, 3), dtype=np.float32) * -5.0

    # 5. 准备 Rotation (旋转)
    # 四元数 [w, x, y, z]，单位四元数 [1, 0, 0, 0]
    rots = np.zeros((N, 4), dtype=np.float32)
    rots[:, 0] = 1.0

    # 6. 构建结构化数组用于写入 PLY
    # 定义 PLY 的 Header 结构
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    elements = np.empty(N, dtype=dtype_list)
    
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opacities[:, 0]
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rots[:, 0]
    elements['rot_1'] = rots[:, 1]
    elements['rot_2'] = rots[:, 2]
    elements['rot_3'] = rots[:, 3]

    # 7. 写入文件
    # 确保目录存在
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    from plyfile import PlyData, PlyElement  # 如果没有 plyfile 库，请 pip install plyfile
    # 为了避免依赖外部库，我手写二进制写入，这样你不安装库也能跑
    
    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
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
        f.write(header.encode('utf-8'))
        f.write(elements.tobytes())
    
    print(f"[info] Saved 3DGS PLY to: {filename}")


# ================= 数据加载与处理 =================

def load_depth_from_npz(npz_path: str) -> Optional[np.ndarray]:
    try:
        data = np.load(npz_path)
        possible_keys = ["pred", "depth", "arr_0", "prediction", "inv_depth"]
        depth = None
        for key in possible_keys:
            if key in data:
                depth = data[key]
                break
        if depth is None:
            for key in data.keys():
                if data[key].ndim >= 2:
                    depth = data[key]
                    break
        if depth is not None and depth.ndim == 3:
            depth = depth.squeeze()
        return depth
    except Exception as e:
        print(f"[error] {e}")
        return None

def generate_extrapolation_constrained(
    depth: np.ndarray,
    K: np.ndarray,
    edge_thresh_metric: float = 2.0,
    steps: int = 15,            # 减少默认步数
    border_margin: float = 0.1, # 忽略图像左右 10% 的区域
    max_bg_depth: float = 80.0  # 忽略超过 80米 的背景（天空）
) -> np.ndarray:
    """
    受限的几何外推：增加边界过滤和深度过滤
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 计算水平梯度
    diff = depth[:, 1:] - depth[:, :-1]
    diff = np.pad(diff, ((0, 0), (0, 1)), constant_values=0)

    mask_right_bg = diff > edge_thresh_metric
    mask_left_bg = diff < -edge_thresh_metric

    new_pts = []
    
    # 定义有效区域 (Crop Window)
    col_start = int(W * border_margin)
    col_end = int(W * (1 - border_margin))

    print(f"[info] Extrapolating within col range [{col_start}, {col_end}], max_depth={max_bg_depth}")

    # 隔行扫描 (Stride=2) 减少点数
    for v in range(0, H, 2): 
        # Case A: 背景在右，向左补全
        cols = np.where(mask_right_bg[v, :])[0]
        for u in cols:
            # 约束 1: 忽略图像边缘
            if u < col_start or u > col_end: continue
            
            bg_u = u + 1
            if bg_u >= W: continue
            
            z_seed = depth[v, bg_u]
            # 约束 2: 忽略无穷远背景 (天空)
            if z_seed > max_bg_depth: continue

            for i in range(1, steps + 1):
                new_u = bg_u - i * 1.0
                # 约束 3: 不要补全出画框
                if new_u < 0: break 
                
                new_z = z_seed
                new_x = (new_u - cx) * new_z / fx
                new_y = (v - cy) * new_z / fy
                new_pts.append([new_x, new_y, new_z])

        # Case B: 背景在左，向右补全
        cols = np.where(mask_left_bg[v, :])[0]
        for u in cols:
            # 约束 1
            if u < col_start or u > col_end: continue
            
            bg_u = u
            z_seed = depth[v, bg_u]
            # 约束 2
            if z_seed > max_bg_depth: continue

            for i in range(1, steps + 1):
                new_u = bg_u + i * 1.0
                # 约束 3
                if new_u >= W: break

                new_z = z_seed
                new_x = (new_u - cx) * new_z / fx
                new_y = (v - cy) * new_z / fy
                new_pts.append([new_x, new_y, new_z])

    return np.array(new_pts)

def backproject_points(depth, image, K, step=4, z_min=0.1, z_max=200.0):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    v, u = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing='ij')
    v = v.flatten()
    u = u.flatten()
    z = depth[::step, ::step].flatten()
    c = image[::step, ::step].reshape(-1, 3)
    
    valid = (z > z_min) & (z < z_max)
    
    x = (u[valid] - cx) * z[valid] / fx
    y = (v[valid] - cy) * z[valid] / fy
    z = z[valid]
    
    return np.stack([x, y, z], axis=1), c[valid]

def main():
    parser = argparse.ArgumentParser()
    # 默认路径
    parser.add_argument("--image", default="/data/lfliang/project/DrivingForward/input_data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-24-10-42-41+0800__CAM_BACK_LEFT__1532400339697441.jpg", help="输入 RGB 图像路径 (jpg/png)")
    parser.add_argument("--depth", default="/data/lfliang/project/DrivingForward/input_data/nuscenes/samples/DEPTH_DVGT/CAM_BACK_LEFT/n015-2018-07-24-10-42-41+0800__CAM_BACK_LEFT__1532400339697441.jpg.npz", help="输入深度 npz 路径")
    parser.add_argument("--output", default="vis_gs.ply", help="输出路径")
    
    # 调整这些参数来控制生成的点数
    parser.add_argument("--edge-thresh", type=float, default=2.0, help="深度跳变阈值")
    parser.add_argument("--steps", type=int, default=15, help="外推步数 (不要太大)")
    parser.add_argument("--border-margin", type=float, default=0.05, help="忽略图像左右 5% 区域")
    parser.add_argument("--max-bg-depth", type=float, default=80.0, help="超过80米不补全 (去天空)")
    
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(args.image) or not os.path.exists(args.depth):
        print("[error] Files not found.")
        return

    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    
    depth = load_depth_from_npz(args.depth)
    if depth.shape[:2] != (H, W):
        print(f"[warn] Resizing depth {depth.shape} -> {(H, W)}")
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    # 2. Intrinsics (Approx)
    K = np.array([[W*1.2, 0, W/2], [0, H*1.2, H/2], [0, 0, 1]])

    # 3. Generate Original Points
    print("[1/3] Generating Original Cloud...")
    orig_pts, orig_colors = backproject_points(depth, img, K, step=4)

    # 4. Generate Extrapolated Points (Constrained)
    print("[2/3] Generating Extrapolated Latent Gaussians...")
    ghost_pts = generate_extrapolation_constrained(
        depth, K, 
        edge_thresh_metric=args.edge_thresh, 
        steps=args.steps,
        border_margin=args.border_margin,
        max_bg_depth=args.max_bg_depth
    )

    if len(ghost_pts) > 0:
        # 补全点设为绿色
        ghost_colors = np.zeros_like(ghost_pts)
        ghost_colors[:, 1] = 255 
        
        final_pts = np.vstack([orig_pts, ghost_pts])
        final_colors = np.vstack([orig_colors, ghost_colors])
    else:
        print("[warn] No points extrapolated.")
        final_pts = orig_pts
        final_colors = orig_colors

    # 5. Save as GS PLY
    print("[3/3] Saving as 3D Gaussian PLY...")
    save_ply_as_gaussian(args.output, final_pts, final_colors)
    print("[done] File saved. Open with SuperSplat/SIBR.")

if __name__ == "__main__":
    main()
