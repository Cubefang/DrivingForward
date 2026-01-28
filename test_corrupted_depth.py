#!/usr/bin/env python3
"""
测试损坏的深度图文件
用于验证文件损坏检测和错误处理机制
"""

import os
import sys
import numpy as np
import zlib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 损坏的文件路径
# CORRUPTED_FILE = "/home/lfliang/project/DrivingForward/input_data/nuscenes/samples/DEPTH_MAP/CAM_BACK/samples/CAM_BACK/n008-2018-08-01-15-52-19-0400__CAM_BACK__1533153733537558.jpg.npz"
CORRUPTED_FILE = "/home/lfliang/data/nuscenes/raw/Trainval/samples/DEPTH_DVGT/CAM_BACK_LEFT/n015-2018-07-24-10-42-41+0800__CAM_BACK_LEFT__1532400339697441.jpg.npz"
# 尝试找到一个正常文件进行对比
DEPTH_MAP_DIR = "/home/lfliang/project/DrivingForward/input_data/nuscenes/samples/DEPTH_DVGT/CAM_FRONT"
NORMAL_FILE = "/home/lfliang/project/DrivingForward/input_data/nuscenes/samples/DEPTH_MAP/CAM_FRONT/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915366912465.jpg.npz"

def find_normal_file(corrupted_file):
    """查找一个正常的文件用于对比"""
    dir_path = os.path.dirname(corrupted_file)
    files = [f for f in os.listdir(dir_path) if f.endswith('.npz') and f != os.path.basename(corrupted_file)]
    if not files:
        return None
    
    # 尝试加载第一个文件，如果成功就返回
    for filename in files[:5]:  # 最多尝试5个文件
        filepath = os.path.join(dir_path, filename)
        try:
            data = np.load(filepath, allow_pickle=True)
            if 'depth' in data:
                return filepath
        except:
            continue
    return None


def get_file_info(filepath):
    """获取文件基本信息"""
    if not os.path.exists(filepath):
        return None
    
    info = {
        'path': filepath,
        'exists': True,
        'size': os.path.getsize(filepath),
        'size_mb': os.path.getsize(filepath) / (1024 * 1024),
        'readable': False,
        'error': None,
        'data': None,
        'depth_shape': None,
        'depth_stats': None
    }
    
    # 尝试读取文件
    try:
        data = np.load(filepath, allow_pickle=True)
        info['readable'] = True
        info['data'] = data
        
        if 'depth' in data:
            depth = data['depth']
            info['depth_shape'] = depth.shape
            info['depth_stats'] = {
                'min': float(np.min(depth)),
                'max': float(np.max(depth)),
                'mean': float(np.mean(depth)),
                'std': float(np.std(depth)),
                'non_zero': int(np.count_nonzero(depth)),
                'total': int(depth.size)
            }
    except zlib.error as e:
        info['error'] = f"zlib.error: {str(e)}"
        info['error_type'] = 'zlib.error'
    except OSError as e:
        info['error'] = f"OSError: {str(e)}"
        info['error_type'] = 'OSError'
    except ValueError as e:
        info['error'] = f"ValueError: {str(e)}"
        info['error_type'] = 'ValueError'
    except KeyError as e:
        info['error'] = f"KeyError: {str(e)}"
        info['error_type'] = 'KeyError'
    except Exception as e:
        info['error'] = f"{type(e).__name__}: {str(e)}"
        info['error_type'] = type(e).__name__
    
    return info


def print_file_comparison(corrupted_info, normal_info=None):
    """打印文件对比信息"""
    print("=" * 80)
    print("深度图文件测试报告")
    print("=" * 80)
    
    print("\n【损坏文件信息】")
    print(f"  路径: {corrupted_info['path']}")
    print(f"  文件大小: {corrupted_info['size']:,} 字节 ({corrupted_info['size_mb']:.4f} MB)")
    print(f"  可读性: {'✓ 可读' if corrupted_info['readable'] else '✗ 不可读'}")
    
    if corrupted_info['error']:
        print(f"  错误类型: {corrupted_info.get('error_type', 'Unknown')}")
        print(f"  错误信息: {corrupted_info['error']}")
    else:
        print(f"  深度图形状: {corrupted_info['depth_shape']}")
        if corrupted_info['depth_stats']:
            stats = corrupted_info['depth_stats']
            print(f"  深度统计:")
            print(f"    - 最小值: {stats['min']:.4f}")
            print(f"    - 最大值: {stats['max']:.4f}")
            print(f"    - 平均值: {stats['mean']:.4f}")
            print(f"    - 标准差: {stats['std']:.4f}")
            print(f"    - 非零像素: {stats['non_zero']:,} / {stats['total']:,}")
    
    if normal_info:
        print("\n【正常文件对比】")
        print(f"  路径: {normal_info['path']}")
        print(f"  文件大小: {normal_info['size']:,} 字节 ({normal_info['size_mb']:.4f} MB)")
        print(f"  可读性: {'✓ 可读' if normal_info['readable'] else '✗ 不可读'}")
        if normal_info['depth_shape']:
            print(f"  深度图形状: {normal_info['depth_shape']}")
            if normal_info['depth_stats']:
                stats = normal_info['depth_stats']
                print(f"  深度统计:")
                print(f"    - 最小值: {stats['min']:.4f}")
                print(f"    - 最大值: {stats['max']:.4f}")
                print(f"    - 平均值: {stats['mean']:.4f}")
                print(f"    - 标准差: {stats['std']:.4f}")
                print(f"    - 非零像素: {stats['non_zero']:,} / {stats['total']:,}")
        
        print("\n【对比分析】")
        size_diff = abs(corrupted_info['size'] - normal_info['size'])
        size_diff_pct = (size_diff / normal_info['size']) * 100
        print(f"  文件大小差异: {size_diff:,} 字节 ({size_diff_pct:.2f}%)")
        if corrupted_info['readable'] != normal_info['readable']:
            print(f"  ⚠️  可读性不同: 损坏文件{'可读' if corrupted_info['readable'] else '不可读'}, 正常文件{'可读' if normal_info['readable'] else '不可读'}")
    
    print("\n" + "=" * 80)


def visualize_comparison(corrupted_info, normal_info=None):
    """可视化对比"""
    if not corrupted_info['readable'] and (not normal_info or not normal_info['readable']):
        print("\n⚠️  无法可视化：文件损坏无法读取")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 损坏文件可视化
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('损坏文件 - 深度图', fontsize=12, fontweight='bold', color='red')
    if corrupted_info['readable'] and corrupted_info['depth_shape']:
        depth = corrupted_info['data']['depth']
        im1 = ax1.imshow(depth, cmap='jet', aspect='auto')
        plt.colorbar(im1, ax=ax1, label='深度值')
        ax1.text(0.02, 0.98, f"形状: {depth.shape}\n可读: ✓", 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, f"文件损坏\n无法读取\n\n错误: {corrupted_info.get('error_type', 'Unknown')}", 
                ha='center', va='center', fontsize=14, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        ax1.set_xticks([])
        ax1.set_yticks([])
    
    # 损坏文件直方图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('损坏文件 - 深度值分布', fontsize=12, fontweight='bold', color='red')
    if corrupted_info['readable'] and corrupted_info['depth_stats']:
        depth = corrupted_info['data']['depth']
        depth_flat = depth.flatten()
        depth_flat = depth_flat[depth_flat > 0]  # 只显示非零值
        if len(depth_flat) > 0:
            ax2.hist(depth_flat, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax2.set_xlabel('深度值')
            ax2.set_ylabel('频数')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无有效深度数据', ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, '无法读取数据', ha='center', va='center', color='red')
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # 损坏文件统计信息
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    stats_text = "损坏文件统计\n" + "=" * 30 + "\n"
    stats_text += f"文件大小: {corrupted_info['size_mb']:.4f} MB\n"
    stats_text += f"可读性: {'✓ 可读' if corrupted_info['readable'] else '✗ 不可读'}\n"
    if corrupted_info['error']:
        stats_text += f"\n错误类型:\n{corrupted_info.get('error_type', 'Unknown')}\n"
        stats_text += f"\n错误信息:\n{corrupted_info['error'][:100]}...\n" if len(corrupted_info['error']) > 100 else f"\n错误信息:\n{corrupted_info['error']}\n"
    if corrupted_info['depth_stats']:
        stats = corrupted_info['depth_stats']
        stats_text += f"\n深度统计:\n"
        stats_text += f"  形状: {corrupted_info['depth_shape']}\n"
        stats_text += f"  最小值: {stats['min']:.4f}\n"
        stats_text += f"  最大值: {stats['max']:.4f}\n"
        stats_text += f"  平均值: {stats['mean']:.4f}\n"
        stats_text += f"  标准差: {stats['std']:.4f}\n"
        stats_text += f"  非零像素: {stats['non_zero']:,}/{stats['total']:,}\n"
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 正常文件可视化（如果存在）
    if normal_info and normal_info['readable']:
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title('正常文件 - 深度图', fontsize=12, fontweight='bold', color='green')
        depth = normal_info['data']['depth']
        im4 = ax4.imshow(depth, cmap='jet', aspect='auto')
        plt.colorbar(im4, ax=ax4, label='深度值')
        ax4.text(0.02, 0.98, f"形状: {depth.shape}\n可读: ✓", 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 正常文件直方图
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title('正常文件 - 深度值分布', fontsize=12, fontweight='bold', color='green')
        depth_flat = depth.flatten()
        depth_flat = depth_flat[depth_flat > 0]
        if len(depth_flat) > 0:
            ax5.hist(depth_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax5.set_xlabel('深度值')
            ax5.set_ylabel('频数')
            ax5.grid(True, alpha=0.3)
        
        # 正常文件统计信息
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        stats_text = "正常文件统计\n" + "=" * 30 + "\n"
        stats_text += f"文件大小: {normal_info['size_mb']:.4f} MB\n"
        stats_text += f"可读性: ✓ 可读\n"
        if normal_info['depth_stats']:
            stats = normal_info['depth_stats']
            stats_text += f"\n深度统计:\n"
            stats_text += f"  形状: {normal_info['depth_shape']}\n"
            stats_text += f"  最小值: {stats['min']:.4f}\n"
            stats_text += f"  最大值: {stats['max']:.4f}\n"
            stats_text += f"  平均值: {stats['mean']:.4f}\n"
            stats_text += f"  标准差: {stats['std']:.4f}\n"
            stats_text += f"  非零像素: {stats['non_zero']:,}/{stats['total']:,}\n"
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        # 如果没有正常文件，显示提示
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        ax4.text(0.5, 0.5, '未找到正常文件进行对比', 
                ha='center', va='center', fontsize=16, color='gray')
    
    plt.suptitle('深度图文件损坏检测对比', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图片
    output_path = '/home/lfliang/project/DrivingForward/test_depth_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存到: {output_path}")
    
    plt.show()


def main():
    """主函数"""
    print("开始测试损坏的深度图文件...\n")
    
    # 检查损坏文件是否存在
    if not os.path.exists(CORRUPTED_FILE):
        print(f"❌ 错误: 文件不存在: {CORRUPTED_FILE}")
        return
    
    # 获取损坏文件信息
    print("正在分析损坏文件...")
    corrupted_info = get_file_info(CORRUPTED_FILE)
    
    # 查找正常文件
    print("正在查找正常文件进行对比...")
    # normal_file = find_normal_file(CORRUPTED_FILE)
    normal_file = NORMAL_FILE
    if normal_file:
        normal_info = get_file_info(normal_file)
        print(f"✓ 找到正常文件: {os.path.basename(normal_file)}")
    else:
        print("⚠️  未找到正常文件进行对比")
    
    # 打印对比信息
    print_file_comparison(corrupted_info, normal_info)
    
    # 可视化
    print("\n正在生成可视化图表...")
    try:
        visualize_comparison(corrupted_info, normal_info)
    except Exception as e:
        print(f"⚠️  可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    if corrupted_info['error']:
        print(f"✓ 成功检测到文件损坏")
        print(f"  错误类型: {corrupted_info.get('error_type', 'Unknown')}")
        print(f"  错误信息: {corrupted_info['error']}")
        print(f"\n建议: 该文件将在训练过程中被自动跳过")
    else:
        print("⚠️  文件可以正常读取，可能不是损坏文件")
    print("=" * 80)


if __name__ == '__main__':
    main()

