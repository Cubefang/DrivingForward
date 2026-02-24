# 实现步骤总结

基于更新后的方案（3D Warping + Z-Test，无需LaMa），以下是具体实现步骤：

## 步骤1：实现教师端 Warping 函数

**目标**：将 t+1 帧 warp 回 t 帧，通过 Z-Test 生成训练真值

**新建文件**：`dataset/teacher_warping.py`

**核心功能**：
- 3D Warping：将 t+1 点云投影回 t 视角
- Z-Test：比较 Z_proj 和 D_t，识别解遮挡区域
- 生成 RGB_GT, Depth_GT, M_valid

## 步骤2：修改数据集加载

**修改文件**：`dataset/nuscenes_dataset.py`（或对应数据集）

**核心修改**：
- 加载 t+1 帧数据（RGB, Depth, Pose）
- 调用 teacher_warping 生成真值
- 返回训练所需的所有数据

## 步骤3：集成 GS 补全网络

**修改文件**：`models/drivingforward_model.py`

**核心修改**：
- 在 `prepare_model()` 中添加 gs_completion_net
- 实现 `set_gs_completion_net()` 方法

## 步骤4：修改 GS 生成逻辑（添加 Depth Dilation）

**修改文件**：`models/drivingforward_model.py` - `get_gaussian_data()`

**核心修改**：
- 使用 MaxPool 实现 Depth Dilation
- 调用 gs_completion_net 预测额外 GS 球
- 合并可见和额外的 GS 球

## 步骤5：添加场景补全损失

**修改文件**：`models/drivingforward_model.py` - `compute_losses()`

**核心修改**：
- 添加 completion_rgb 损失
- 使用 M_valid mask 只在解遮挡区域计算损失

## 步骤6：更新配置文件

**修改文件**：`configs/nuscenes/main.yaml`

**核心修改**：
- 添加 gs_completion 配置
- 设置超参数（depth_offset_scale, validity_threshold 等）

---

接下来我会为每个步骤提供详细的代码实现。
