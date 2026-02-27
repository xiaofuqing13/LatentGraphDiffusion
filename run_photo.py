"""
run_photo.py
LGD 论文 Table 4 - Photo 数据集完整复现脚本 (一键运行)

用法:
  python run_photo.py

流程:
  1. Encoder 预训练 (--repeat 10)
  2. 自动查找最佳 Encoder checkpoint
  3. Diffusion 训练 (--repeat 10)
  4. 输出最终结果
"""
import subprocess
import sys
import os
import json


def run_cmd(cmd, desc):
    """运行命令并检查返回码"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  命令: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[错误] {desc} 失败! 返回码: {result.returncode}")
        sys.exit(1)
    print(f"\n[完成] {desc}")


def print_results(json_path, label):
    """打印 JSON 结果文件"""
    if os.path.isfile(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        acc = data.get('accuracy', 'N/A')
        std = data.get('accuracy_std', 'N/A')
        print(f"  {label}: accuracy = {acc} ± {std}")
    else:
        print(f"  {label}: 结果文件未找到 ({json_path})")


if __name__ == '__main__':
    print("=" * 60)
    print("  LGD 论文 Table 4 - Photo 数据集完整复现")
    print("  论文目标: accuracy = 96.94 ± 0.14")
    print("=" * 60)

    python = sys.executable

    # Step 1: Encoder 预训练
    run_cmd(
        f'{python} pretrain.py --cfg cfg/photo-encoder.yaml --repeat 10 wandb.use False',
        'Step 1/4: Encoder 预训练 (--repeat 10, 约 1.7h)'
    )

    # Step 2: 查找最佳 ckpt 并更新 diffusion 配置
    run_cmd(
        f'{python} find_best_ckpt.py --encoder_dir results/photo-encoder --diffusion_cfg cfg/photo-diffusion.yaml',
        'Step 2/4: 查找最佳 Encoder Checkpoint'
    )

    # Step 3: Diffusion 训练
    run_cmd(
        f'{python} train_diffusion.py --cfg cfg/photo-diffusion.yaml --repeat 10 wandb.use False',
        'Step 3/4: Diffusion 训练 (--repeat 10, 约 25h)'
    )

    # Step 4: 输出结果
    print(f"\n{'='*60}")
    print("  最终结果汇总")
    print(f"{'='*60}")
    print_results('results/photo-encoder/agg/test/best.json', 'Encoder')
    print_results('results/photo-diffusion/agg/test/best.json', 'Diffusion (最终)')
    print(f"\n  论文目标: Photo accuracy = 96.94 ± 0.14")
    print(f"{'='*60}")
