# GS场景补全 - 时空蒸馏实现方案（更新版）

## 核心创新：Self-Supervised Temporal-to-Spatial Distillation

**将时间维度的信息（未来帧）蒸馏到空间维度（单帧遮挡补全）**

## 1. 架构总览

### 训练阶段流程

```
输入：视频序列 {I_t, D_t, I_{t+1}, D_{t+1}, P_{t→t+1}}
                ↓
        ┌─────────────────────────┐
        │   教师端（离线数据处理）  │
        └─────────────────────────┘
                ↓
    3D Warping：将 t+1 投影回 t 视角
                ↓
    Z-Test：比较投影深度 Z_proj 和 D_t
                ↓
    生成 Validity Mask（Z_proj > D_t 的区域）
                ↓
    生成真值：RGB_GT, Depth_GT, M_valid
                ↓
        ┌─────────────────────────┐
        │   学生端（在线训练）      │
        └─────────────────────────┘
                ↓
    输入：单帧 I_t + D_t（无未来信息！）
                ↓
    GaussianNetwork → 可见区域 GS 球
                ↓
    GSCompletionNetwork → 额外 GS 球（遮挡区域）
                ↓
    Depth Dilation：MaxPool(D_t) + offset
                ↓
    合并两组 GS 球
                ↓
    渲染 t 视角 → I_rendered
                ↓
    损失：MSE(I_rendered, RGB_GT) * M_valid
                ↓
    反向传播 → 更新学生网络
```

### 推理阶段流程

```
输入：单帧 I_t + D_t
                ↓
    GaussianNetwork → 可见区域 GS 球
                ↓
    GSCompletionNetwork → 额外 GS 球
                ↓
    Depth Dilation + 合并
                ↓
    渲染任意视角 → 输出完整图像
```

## 2. 关键设计决策

### 2.1 教师端：3D Warping + Z-Test

**目标**：自动识别"解遮挡"区域，生成训练真值

**步骤**：

```python
# 1. 3D Warping：将 t+1 的点云投影回 t 视角
for each pixel (u', v') in I_{t+1}:
    # 反投影到 3D
    xyz_{t+1} = depth2pc(D_{t+1}[u', v'], P_{t+1}, K)

    # 变换到 t 坐标系
    xyz_t = P_{t→t+1} @ xyz_{t+1}

    # 投影到 t 图像平面
    (u, v, Z_proj) = project(xyz_t, P_t, K)

# 2. Z-Test：深度比较
M_valid = (Z_proj > D_t + threshold)
# 解释：
# - Z_proj > D_t：t+1 看到了 t 看不到的背景（解遮挡）✅
# - Z_proj ≤ D_t：新遮挡或相同区域（忽略）❌

# 3. 生成真值
RGB_GT = zeros_like(I_t)
Depth_GT = zeros_like(D_t)
RGB_GT[M_valid] = I_{t+1}_warped[M_valid]
Depth_GT[M_valid] = Z_proj[M_valid]

# 4. 输出
return RGB_GT, Depth_GT, M_valid
```

**优势**：
- 自动识别解遮挡区域，无需手动标注
- 通过深度比较过滤新遮挡和动态物体
- 简单高效，不需要复杂的 Consistency Check

### 2.2 学生端：Depth Dilation（膨胀）

**问题**：被遮挡区域 depth=0，如何预测合理的深度？

**原方案（不好）**：
```python
depth_additional = depth + offset  # depth=0 时，offset 没有参考
```

**新方案（更好）**：
```python
# 使用邻域背景深度作为起点
depth_dilated = MaxPool2d(depth, kernel_size=5)  # 膨胀操作
depth_additional = depth_dilated + offset

# 解释：
# - MaxPool 获取邻域最大深度（背景深度）
# - 被遮挡区域会继承周围背景的深度
# - offset 在此基础上微调
```

**示例**：
```
原始深度图：
[5, 5, 0, 0, 10]  # 0 是遮挡区域
     ↓ MaxPool(kernel=3)
[5, 5, 5, 10, 10]  # 遮挡区域继承了邻域深度
     ↓ + offset
[5, 5, 7, 12, 10]  # 网络预测 offset=[0,0,2,2,0]
```

### 2.3 GS补全网络设计（方法B）

**动机**：每个像素都可能对应一个被遮挡的背景 GS 球

**设计**：
- 输入：RGB + Depth（包含 0 值）
- 输出：
  - xyz_offset：深度偏移 [B, 1, H, W]
  - rotation：旋转四元数 [B, 4, H, W]
  - scale：缩放 [B, 3, H, W]
  - opacity：不透明度 [B, 1, H, W]
  - sh：球谐函数 [B, H*W, 1, 3, d_sh]
  - validity：有效性分数 [B, 1, H, W] ⭐关键！

**有效性分数的作用**：
- 判断哪些像素背后有被遮挡的背景
- 天空背后：validity ≈ 0（没有背景）
- 车辆背后：validity ≈ 1（有路面）
- 建筑背后：validity ≈ 0（静止物体，不需要补全）

### 2.4 训练策略

**信息不对称**：
- 教师：拥有"上帝视角"（时间机器），能看到未来
- 学生：只有"凡人视角"，只能看到当前帧

**训练目标**：
- 让学生学会"预知未来"
- 从单帧图像推断被遮挡区域的外观

**损失函数**：
```python
L_rgb = MSE(I_rendered, RGB_GT) * M_valid
L_depth = MSE(D_rendered, Depth_GT) * M_valid  # 可选
L_total = L_rgb + λ * L_depth
```

