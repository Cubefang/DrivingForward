import argparse
import os
from pathlib import Path

import torch
import numpy as np

import utils
from models import DrivingForwardModel


def write_ply(path, points, colors=None):
    """
    Save point cloud to ASCII PLY. points: (N,3), colors: (N,3) in [0,1] or [0,255]
    """
    assert points.shape[1] == 3
    if colors is None:
        colors = np.ones_like(points) * 255
    if colors.max() <= 1.0:
        colors = (colors * 255).clip(0, 255)
    colors = colors.astype(np.uint8)
    pts = np.concatenate([points.astype(np.float32), colors], axis=1)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, pts, fmt="%.6f %.6f %.6f %d %d %d")


@torch.no_grad()
def export_sample(cfg, model, inputs, outputs, out_dir, token, cam=0, frame_id=0):
    """
    Export a single sample to PLY using xyz from outputs and RGB from inputs.
    """
    # xyz: [B, N, 3]; valid mask: [B, N]
    xyz = outputs[("cam", cam)][("xyz", frame_id, 0)]
    valid = outputs[("cam", cam)][("pts_valid", frame_id, 0)]
    b, n, _ = xyz.shape
    assert b == 1, "eval batch_size should be 1 for export"
    xyz = xyz[0].cpu().numpy()
    valid = valid[0].cpu().numpy().astype(bool)

    # colors: [B, V, C, H, W] -> pick cam, frame_id
    img = inputs[("color", frame_id, 0)][0, cam]  # [3,H,W], in [0,1]
    c, h, w = img.shape
    img_flat = img.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()

    xyz = xyz[valid]
    colors = img_flat[valid]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ply_path = Path(out_dir) / f"{token}_cam{cam}_f{frame_id}.ply"
    write_ply(ply_path, xyz, colors)
    print(f"Saved {ply_path} with {xyz.shape[0]} points")


def main():
    parser = argparse.ArgumentParser(description="Export PLY point cloud from DrivingForward eval sample")
    parser.add_argument("--config_file", type=str, default="./configs/nuscenes/main.yaml")
    parser.add_argument("--novel_view_mode", type=str, default="MF")
    parser.add_argument("--weight_path", type=str, default="./weights_MF")
    parser.add_argument("--token", type=str, default=None, help="sample token to export; if None export all eval samples")
    parser.add_argument("--cam", type=int, default=0, help="camera index to export")
    parser.add_argument("--frame_id", type=int, default=0, help="frame id (0 for reference, -1/1 for MF)")
    parser.add_argument("--out_dir", type=str, default="./results/ply_export")
    args = parser.parse_args()

    cfg = utils.get_config(args.config_file, mode="eval", novel_view_mode=args.novel_view_mode)
    cfg["load"]["pretrain"] = True
    cfg["load"]["load_dir"] = args.weight_path

    model = DrivingForwardModel(cfg, rank=0)
    model.set_eval()
    eval_loader = model.eval_dataloader()

    for inputs in eval_loader:
        if args.token is not None and inputs["token"][0] != args.token:
            continue
        outputs, _ = model.process_batch(inputs, rank=0)
        export_sample(cfg, model, inputs, outputs, args.out_dir, inputs["token"][0], cam=args.cam, frame_id=args.frame_id)
        if args.token is not None:
            break


if __name__ == "__main__":
    main()
