# DrivingForward 项目结构梳理

## 项目概述
DrivingForward是一个基于3D Gaussian Splatting的驾驶场景重建项目，从稀疏的环视输入进行前馈式场景重建。

## 核心目录结构

### 1. 训练入口
- **`train.py`** - 训练脚本主入口
  - 解析配置参数
  - 初始化模型和训练器
  - 启动训练流程

### 2. 模型定义 (`models/`)
- **`drivingforward_model.py`** ⭐ **核心模型文件**
  - `DrivingForwardModel`类：整个系统的核心模型
  - 整合深度网络、姿态网络、Gaussian网络
  - 处理前向传播、损失计算、Gaussian数据生成

- **`base_model.py`** - 基础模型类

#### 2.1 Gaussian相关 (`models/gaussian/`)
- **`gaussian_network.py`** ⭐ **Gaussian参数预测网络**
  - `GaussianNetwork`类：预测Gaussian参数（旋转、缩放、不透明度、球谐系数）
  - 使用CNN架构（UnetExtractor + ResidualBlock）
  - 输入：RGB图像、深度图、图像特征
  - 输出：旋转、缩放、不透明度、SH系数

- **`extractor.py`** - 特征提取器
  - `UnetExtractor`：深度编码器（CNN架构）
  - `ResidualBlock`：残差块

- **`utils.py`** - Gaussian工具函数
  - `depth2pc`：深度图转点云
  - `rotate_sh`：球谐系数旋转
  - 相机参数转换函数

- **`GaussianRender.py`** - Gaussian渲染器
- **`gaussian_renderer/`** - 渲染器实现
- **`gaussian-splatting/`** - Gaussian Splatting子模块

#### 2.2 几何处理 (`models/geometry/`)
- **`view_rendering.py`** ⭐ **视图渲染模块**
  - `ViewRendering`类：处理图像拼接和视图变换
  - `get_virtual_image`：图像warping
  - `get_norm_image_single`：图像归一化对齐
  - 支持时空视图融合

- **`pose.py`** - 姿态计算
- **`geometry_util.py`** - 几何工具函数

#### 2.3 损失函数 (`models/losses/`)
- **`multi_cam_loss.py`** - 多相机损失
- **`single_cam_loss.py`** - 单相机损失
- **`base_loss.py`** - 基础损失类
- **`loss_util.py`** - 损失工具函数

### 3. 网络架构 (`network/`)
- **`depth_network.py`** ⭐ **深度预测网络**
  - `DepthNetwork`类：多视图深度融合网络
  - 使用ResNet编码器提取特征
  - `VFNet`进行体素融合
  - `DepthDecoder`解码深度图

- **`volumetric_fusionnet.py`** ⭐ **体素融合网络**
  - `VFNet`类：环视融合核心模块
  - `backproject_into_voxel`：将图像特征反投影到3D体素空间
  - `project_voxel_into_image`：将体素特征投影回图像空间
  - 处理重叠和非重叠区域

- **`pose_network.py`** - 姿态估计网络
  - `PoseNetwork`类：估计相机间相对姿态
  - 使用BEV特征进行姿态预测

- **`blocks.py`** - 基础网络块
  - `conv2d`、`conv1d`：卷积层构建函数
  - `pack_cam_feat`、`unpack_cam_feat`：多相机特征打包/解包

### 4. 数据集 (`dataset/`)
- **`nuscenes_dataset.py`** ⭐ **nuScenes数据集加载器**
  - 加载多相机图像、深度、姿态等数据
  - 支持SF（单帧）和MF（多帧）模式

- **`base_dataset.py`** - 基础数据集类
- **`data_util.py`** - 数据工具函数
- **`nuscenes/`** - nuScenes数据集相关文件

### 5. 训练器 (`trainer/`)
- **`trainer.py`** ⭐ **训练器主文件**
  - `DrivingForwardTrainer`类：训练和评估流程
  - `train`：训练循环
  - `validate`：验证
  - `evaluate`：评估
  - 计算PSNR、SSIM、LPIPS等指标

### 6. 工具函数 (`utils/`)
- **`misc.py`** - 杂项工具函数
- **`logger.py`** - 日志记录
- **`visualize.py`** - 可视化工具

### 7. 配置文件 (`configs/`)
- **`nuscenes/main.yaml`** ⭐ **主配置文件**
  - 模型参数、训练参数、损失权重等

### 8. 外部依赖 (`external/`)
- **`packnet_sfm/`** - PackNet深度估计（外部库）
- **`dgp/`** - DGP数据集工具
- **`layers/`** - 外部网络层（ResNet编码器等）

## 数据流程

### 训练流程
1. **数据加载** (`dataset/nuscenes_dataset.py`)
   - 加载多相机图像、深度、姿态

2. **深度预测** (`network/depth_network.py`)
   - ResNet编码器提取特征
   - VFNet进行多视图融合
   - 解码器输出深度图

3. **姿态估计** (`network/pose_network.py`)
   - 估计相机间相对姿态

4. **Gaussian参数预测** (`models/gaussian/gaussian_network.py`)
   - 从RGB、深度、特征预测Gaussian参数

5. **视图渲染** (`models/geometry/view_rendering.py`)
   - 图像拼接和视图变换

6. **损失计算** (`models/losses/`)
   - 计算重建损失、深度损失等

### 推理流程（简化）
- 仅使用深度网络和Gaussian网络
- 单帧图像即可预测Gaussian参数

## 关键技术点

1. **多视图融合**：VFNet将多相机特征融合到3D体素空间
2. **Gaussian Splatting**：使用3D Gaussian进行场景表示
3. **前馈式预测**：无需迭代优化，实时重建
4. **灵活输入**：支持SF（单帧）和MF（多帧）模式

