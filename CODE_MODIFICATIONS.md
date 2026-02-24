# 代码修改指南

## 修改1：集成 GS 补全网络到主模型

**文件**：`models/drivingforward_model.py`

**位置**：`prepare_model()` 方法

**添加代码**：
```python
def prepare_model(self, cfg, rank):
    models = {}
    if self.use_pose_net:
        models['pose_net'] = self.set_posenet(cfg)
    depth_net_cls = SupervisedDepthNetwork if getattr(self, 'use_supervised_depth_net', False) else DepthNetwork
    models['depth_net'] = self.set_depthnet(cfg, depth_net_cls)
    if self.gaussian:
        models['gs_net'] = self.set_gaussiannet(cfg)
        # 新增：GS 补全网络
        models['gs_completion_net'] = self.set_gs_completion_net(cfg)

    return models

def set_gs_completion_net(self, cfg):
    """初始化 GS 补全网络"""
    from .gaussian.gs_completion_network import GSCompletionNetwork
    gs_completion_net = GSCompletionNetwork(
        rgb_dim=3,
        depth_dim=1,
        norm_fn='group'
    )
    return gs_completion_net.to(self.device)
```

---

## 修改2：在 get_gaussian_data() 中添加 Depth Dilation 和 GS 补全

**文件**：`models/drivingforward_model.py`

**位置**：`get_gaussian_data()` 方法，第456-458行附近

**原代码**：
```python
outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(...)
valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
```

**修改为**：
```python
# 1. 生成可见区域的 GS 球
depth = outputs[('cam', cam)][('depth', frame_id, 0)]
xyz_visible = depth2pc(depth, ...)
valid_visible = depth != 0.0

# 2. 预测可见区域 GS 参数
rot_visible, scale_visible, opacity_visible, sh_visible = \
    self.gs_net(
        inputs[('color', frame_id, 0)][:, cam, ...],
        depth,
        outputs[('cam', cam)][('img_feat', frame_id, 0)]
    )

# 3. 预测额外 GS 球（遮挡区域）
gs_additional = self.models['gs_completion_net'](
    inputs[('color', frame_id, 0)][:, cam, ...],
    depth,
    outputs[('cam', cam)][('img_feat', frame_id, 0)]
)

# 4. Depth Dilation（关键改进）
depth_dilated = F.max_pool2d(
    depth,
    kernel_size=5,
    stride=1,
    padding=2
)

# 5. 计算额外 GS 球的深度
depth_offset_scale = getattr(self, 'depth_offset_scale', 5.0)
depth_additional = depth_dilated + gs_additional['xyz_offset'] * depth_offset_scale

# 6. 计算额外 GS 球的 xyz
xyz_additional = depth2pc(depth_additional, ...)

# 7. 根据有效性分数过滤
validity_threshold = getattr(self, 'validity_threshold', 0.5)
validity_mask = (gs_additional['validity'] > validity_threshold).squeeze(1)  # [B, H, W]

# 8. 合并可见和额外的 GS 球
# 将 map 形式转换为 list 形式
valid_visible_flat = valid_visible.view(bs, -1)  # [B, H*W]
validity_mask_flat = validity_mask.view(bs, -1)  # [B, H*W]

xyz_visible_list = xyz_visible[valid_visible_flat]  # 需要处理 batch
xyz_additional_list = xyz_additional[validity_mask_flat]

# 简化版：直接拼接（需要后续处理 batch）
xyz_all = torch.cat([xyz_visible, xyz_additional], dim=1)
valid_all = torch.cat([valid_visible_flat, validity_mask_flat], dim=1)

outputs[('cam', cam)][('xyz', frame_id, 0)] = xyz_all
outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid_all

# 同样处理其他参数（rotation, scale, opacity, sh）
# ...
```

---

## 修改3：添加场景补全损失

**文件**：`models/drivingforward_model.py`

**位置**：`compute_losses()` 方法

**添加代码**：
```python
def compute_losses(self, inputs, outputs):
    """计算损失"""
    losses = {}

    # ... 现有损失计算 ...

    # 新增：场景补全损失
    if 'RGB_GT' in inputs and 'M_valid' in inputs:
        # 渲染当前视角
        rendered_rgb = outputs.get(('render', 0, 0))

        if rendered_rgb is not None:
            # 获取真值和 mask
            rgb_gt = inputs['RGB_GT']  # [B, 3, H, W]
            m_valid = inputs['M_valid']  # [B, 1, H, W]

            # 计算损失（只在有效区域）
            loss_weight = getattr(self, 'completion_rgb_weight', 1.0)
            losses['completion_rgb'] = loss_weight * F.mse_loss(
                rendered_rgb * m_valid,
                rgb_gt * m_valid
            ) / (m_valid.sum() + 1e-8)

            # 可选：深度损失
            if 'Depth_GT' in inputs and ('depth', 0, 0) in outputs:
                rendered_depth = outputs[('depth', 0, 0)]
                depth_gt = inputs['Depth_GT']
                depth_weight = getattr(self, 'completion_depth_weight', 0.5)
                losses['completion_depth'] = depth_weight * F.mse_loss(
                    rendered_depth * m_valid,
                    depth_gt * m_valid
                ) / (m_valid.sum() + 1e-8)

    return losses
```

---

## 修改4：更新配置文件

**文件**：`configs/nuscenes/main.yaml`

**添加配置**：
```yaml
# GS 场景补全配置
gs_completion:
  enable: true

  # Depth Dilation 配置
  depth_offset_scale: 5.0  # 深度偏移缩放因子
  validity_threshold: 0.5  # 有效性分数阈值

  # Z-Test 配置
  z_test_threshold: 0.1  # 深度比较阈值（米）

  # 损失权重
  completion_rgb_weight: 1.0
  completion_depth_weight: 0.5
```

---

## 修改5：数据集加载（需要根据具体数据集调整）

**文件**：`dataset/nuscenes_dataset.py`（或对应数据集）

**修改 `__getitem__()` 方法**：
```python
from .teacher_warping import warp_future_to_current

def __getitem__(self, idx):
    # ... 现有代码加载 I_t, D_t ...

    # 新增：加载 t+1 帧
    if idx + 1 < len(self):
        I_t1 = self.load_image(idx + 1)
        D_t1 = self.load_depth(idx + 1)
        P_t_to_t1 = self.get_pose_transform(idx, idx + 1)
        K = self.get_intrinsic(idx)

        # 教师端：生成训练真值
        RGB_GT, Depth_GT, M_valid = warp_future_to_current(
            I_t, D_t, I_t1, D_t1, P_t_to_t1, K,
            threshold=self.config.get('z_test_threshold', 0.1)
        )
    else:
        # 最后一帧，没有 t+1
        RGB_GT = torch.zeros_like(I_t)
        Depth_GT = torch.zeros_like(D_t)
        M_valid = torch.zeros_like(D_t, dtype=torch.bool)

    return {
        'I_t': I_t,
        'D_t': D_t,
        'RGB_GT': RGB_GT,
        'Depth_GT': Depth_GT,
        'M_valid': M_valid,
        # ... 其他数据 ...
    }
```

---

## 总结

以上是核心的代码修改。主要改动点：

1. ✅ 已创建 `GSCompletionNetwork` ([gs_completion_network.py](models/gaussian/gs_completion_network.py))
2. ✅ 已创建 `teacher_warping.py` ([teacher_warping.py](dataset/teacher_warping.py))
3. ⏳ 需要修改 `drivingforward_model.py`（3处修改）
4. ⏳ 需要修改数据集加载代码
5. ⏳ 需要更新配置文件

接下来可以逐步测试每个模块。
