# GS场景补全 - 代码生成总结

## 生成的文件清单

### 1. 教师模型模块 (/models/teachers/)

#### models/teachers/teacher_warping.py
**功能**: 3D Warping + Z-Test，生成训练真值

**核心函数**:
- `depth_to_pointcloud(depth, K, pose)`: 深度图转3D点云
- `project_pointcloud(xyz, K, pose)`: 3D点云投影到图像平面
- `warp_future_to_current(img_t, depth_t, img_t1, depth_t1, pose_t_to_t1, K, threshold)`: 主函数

**输入**:
- img_t: [B, 3, H, W] RGB图像 t时刻
- depth_t: [B, 1, H, W] 深度图 t时刻
- img_t1: [B, 3, H, W] RGB图像 t+1时刻
- depth_t1: [B, 1, H, W] 深度图 t+1时刻
- pose_t_to_t1: [B, 4, 4] 从t到t+1的变换矩阵
- K: [3, 3] 相机内参

**输出**:
- rgb_gt: [B, 3, H, W] 补全区域的RGB真值
- depth_gt: [B, 1, H, W] 补全区域的深度真值
- valid_mask: [B, 1, H, W] 有效区域mask（解遮挡区域）

#### models/teachers/__init__.py
**功能**: 模块初始化，导出教师模型相关函数

---

### 2. GS球预测模块 (/models/gaussian/)

#### models/gaussian/gs_completion_network.py (已存在，已修改)
**功能**: 预测额外的GS球用于补全被遮挡的背景

**输入**:
- img: [B, 3, H, W] RGB图像
- depth: [B, 1, H, W] 深度图（膨胀后）
- img_feat: (feat1, feat2, feat3) RGB特征

**输出**:
- xyz_offset: [B, 1, H, W] 深度偏移
- rotation: [B, 4, H, W] 旋转四元数
- scale: [B, 3, H, W] 缩放
- opacity: [B, 1, H, W] 不透明度
- sh: [B, H*W, 1, 3, d_sh] 球谐函数
- validity: [B, 1, H, W] 有效性分数

#### models/gaussian/depth_utils.py
**功能**: 深度处理和GS球合并的辅助函数

**核心函数**:
1. `depth_dilation(depth, kernel_size, iterations)`: 深度膨胀操作
   - 输入: depth [B, 1, H, W]
   - 输出: dilated_depth [B, 1, H, W]

2. `merge_gaussian_params(gs_visible, gs_completion, validity_threshold)`: 合并GS球参数
   - 输入: 可见区域和补全区域的GS球参数
   - 输出: 合并后的GS球参数

3. `compute_completion_xyz(depth_dilated, xyz_offset, K, scale_factor)`: 计算补全区域GS球的3D坐标
   - 输入: depth_dilated [B, 1, H, W], xyz_offset [B, 1, H, W], K [3, 3]
   - 输出: xyz [B, H, W, 3]

---

### 3. 主模型修改 (models/drivingforward_model.py)

#### 修改1: 导入新模块
```python
from .gaussian.gs_completion_network import GSCompletionNetwork
from .gaussian.depth_utils import depth_dilation, merge_gaussian_params, compute_completion_xyz
from .teachers import warp_future_to_current
```

#### 修改2: prepare_model() - 添加gs_completion_net
```python
if getattr(self, 'use_gs_completion', False):
    models['gs_completion_net'] = self.set_gs_completion_net(cfg)
```

新增方法:
```python
def set_gs_completion_net(self, cfg):
    device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
    return GSCompletionNetwork(rgb_dim=3, depth_dim=1).to(device)
```

#### 修改3: get_gaussian_data() - 集成补全网络
在SF模式下，调用gs_net之后添加:
1. Depth Dilation
2. 调用gs_completion_net
3. 计算补全区域的xyz
4. 保存补全相关的输出到outputs

#### 修改4: 新增generate_teacher_ground_truth()方法
**功能**: 使用Teacher Warping生成GS补全的训练真值

**流程**:
1. 获取t和t+1时刻的数据
2. 调用warp_future_to_current
3. 保存rgb_gt, depth_gt, valid_mask到outputs

#### 修改5: compute_losses() - 添加补全损失
在训练开始时调用generate_teacher_ground_truth()，然后在损失计算中添加:
- completion_rgb_loss: RGB重建损失（仅在有效区域）
- completion_depth_loss: 深度损失（可选）
- 加权组合到总损失

---

### 4. 配置文件 (configs/nuscenes/)

#### configs/nuscenes/main_with_completion.yaml
**新增配置项**:
```yaml
training:
  # GS补全配置
  use_gs_completion: True              # 启用GS补全网络
  depth_dilation_kernel: 5             # 深度膨胀核大小
  depth_dilation_iters: 1              # 深度膨胀迭代次数
  teacher_depth_threshold: 0.1         # Z-Test深度阈值（米）
  xyz_offset_scale: 10.0               # xyz偏移缩放因子
  completion_loss_weight: 1.0          # 补全总损失权重
  completion_depth_weight: 0.1         # 补全深度损失权重

load:
  models_to_load: ['depth_net', 'gs_net', 'gs_completion_net']
```

---

## 数据流图

### 训练阶段
```
输入数据 (inputs)
    ├─ ('color', 0, 0): t时刻RGB
    ├─ ('color', 1, 0): t+1时刻RGB
    ├─ ('depth', 0, 0): t时刻深度
    ├─ ('depth', 1, 0): t+1时刻深度
    ├─ ('cam_T_cam', 0, 1): t到t+1变换
    └─ ('K', 0): 相机内参

↓ generate_teacher_ground_truth()

教师真值 (outputs)
    ├─ ('teacher_rgb_gt', 0, 0)
    ├─ ('teacher_depth_gt', 0, 0)
    └─ ('teacher_valid_mask', 0, 0)

↓ get_gaussian_data()

GS球参数 (outputs)
    ├─ 可见区域: ('rot_maps', 0, 0), ('scale_maps', 0, 0), ...
    └─ 补全区域: ('rot_maps_completion', 0, 0), ('xyz_completion', 0, 0), ...

↓ pred_gaussian_imgs()

渲染结果 (outputs)
    └─ ('gs_completion_rendered', 0, 0)

↓ compute_losses()

损失
    ├─ completion_rgb_loss
    ├─ completion_depth_loss
    └─ completion_total_loss
```

### 推理阶段
```
输入: 单帧 I_t + D_t

↓ GaussianNetwork

可见区域GS球

↓ Depth Dilation + GSCompletionNetwork

补全区域GS球

↓ 合并 + 渲染

完整场景图像
```

---

## 使用说明

### 1. 启用GS补全功能
修改配置文件，设置:
```yaml
training:
  use_gs_completion: True
  novel_view_mode: 'SF'  # 推荐使用SF模式
```

### 2. 训练
```bash
python train.py --config configs/nuscenes/main_with_completion.yaml
```

### 3. 推理
```bash
python eval.py --config configs/nuscenes/main_with_completion.yaml
```

---

## 注意事项

1. **数据要求**: 训练时需要提供t+1时刻的数据（RGB和深度），用于生成教师真值

2. **渲染实现**: 当前代码中补全区域的渲染逻辑（pred_gaussian_imgs）需要进一步实现，以生成('gs_completion_rendered', 0, 0)

3. **内存占用**: 补全网络为每个像素预测GS球，会增加内存占用。可以通过validity_threshold过滤来减少

4. **超参数调优**:
   - depth_dilation_kernel: 控制膨胀范围
   - teacher_depth_threshold: 控制解遮挡区域的识别
   - completion_loss_weight: 控制补全损失的权重

5. **模型加载**: 如果使用预训练模型，需要在models_to_load中添加'gs_completion_net'

---

## 后续工作

1. **实现补全区域的渲染**: 在pred_gaussian_imgs中添加补全区域的渲染逻辑

2. **优化合并策略**: 实现merge_gaussian_params的调用，合并可见和补全区域的GS球

3. **数据集适配**: 确保数据加载器提供t+1时刻的数据

4. **可视化**: 添加补全区域的可视化，方便调试

5. **性能优化**: 优化depth_dilation和merge_gaussian_params的性能
