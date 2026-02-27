"""
find_best_ckpt.py
从 encoder 的多次 repeat 训练结果中找到验证集最佳 ckpt,
并自动更新 diffusion 配置文件的 first_stage_config 字段。

用法:
  python find_best_ckpt.py --encoder_dir results/photo-encoder --diffusion_cfg cfg/photo-diffusion.yaml
"""
import argparse
import json
import os
import re


def parse_best_from_log(log_path):
    """从 logging.log 最后一行 'Best so far' 解析最佳 epoch 和 val_accuracy"""
    best_epoch = None
    best_val_acc = None
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'Best so far: epoch (\d+).*?val_accuracy: ([\d.]+)', line)
            if m:
                best_epoch = int(m.group(1))
                best_val_acc = float(m.group(2))
    return best_epoch, best_val_acc


def parse_best_from_stats(stats_path):
    """从 val/stats.json 找到 accuracy 最高的 epoch"""
    with open(stats_path, 'r') as f:
        content = f.read().strip()
    
    best_epoch = None
    best_acc = -1
    for line in content.split('\n'):
        line = line.strip().rstrip(',')
        if not line or line in ['[', ']']:
            continue
        try:
            data = json.loads(line)
            acc = data.get('accuracy', 0)
            epoch = data.get('epoch', 0)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
        except json.JSONDecodeError:
            continue
    return best_epoch, best_acc


def find_best_encoder_ckpt(encoder_dir):
    """遍历所有 run 目录, 找到 val accuracy 最高的 run 和对应的 best epoch ckpt"""
    best_acc = -1
    best_run = None
    best_epoch = None

    for run_name in sorted(os.listdir(encoder_dir)):
        run_path = os.path.join(encoder_dir, run_name)
        if not os.path.isdir(run_path) or run_name == 'agg':
            continue

        # 方法1: 尝试从 logging.log 解析
        log_path = os.path.join(run_path, 'logging.log')
        if os.path.isfile(log_path):
            epoch, acc = parse_best_from_log(log_path)
            if epoch is not None:
                print(f"  Run {run_name}: val_accuracy={acc:.5f}, best_epoch={epoch} (from log)")
                if acc > best_acc:
                    best_acc = acc
                    best_run = run_name
                    best_epoch = epoch
                continue

        # 方法2: 从 val/stats.json 解析
        stats_path = os.path.join(run_path, 'val', 'stats.json')
        if os.path.isfile(stats_path):
            epoch, acc = parse_best_from_stats(stats_path)
            if epoch is not None:
                print(f"  Run {run_name}: val_accuracy={acc:.5f}, best_epoch={epoch} (from stats)")
                if acc > best_acc:
                    best_acc = acc
                    best_run = run_name
                    best_epoch = epoch
                continue

        print(f"  Run {run_name}: 跳过 (无有效结果文件)")

    if best_run is None:
        raise RuntimeError(f"在 {encoder_dir} 中未找到有效的训练结果")

    ckpt_path = f"{encoder_dir}/{best_run}/ckpt/{best_epoch}.ckpt"
    ckpt_path = ckpt_path.replace('\\', '/')

    # 验证 ckpt 文件存在
    if not os.path.isfile(ckpt_path):
        # 尝试搜索最接近的 ckpt
        ckpt_dir = os.path.join(encoder_dir, best_run, 'ckpt')
        available = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        available_epochs = sorted([int(f.replace('.ckpt', '')) for f in available])
        # 选最接近的
        closest = min(available_epochs, key=lambda x: abs(x - best_epoch))
        ckpt_path = f"{encoder_dir}/{best_run}/ckpt/{closest}.ckpt".replace('\\', '/')
        print(f"\n  注意: epoch {best_epoch} 的 ckpt 不存在, 使用最接近的 epoch {closest}")

    print(f"\n最佳 Encoder: run={best_run}, epoch={best_epoch}, val_accuracy={best_acc:.5f}")
    print(f"Checkpoint: {ckpt_path}")
    return ckpt_path, best_acc


def update_diffusion_cfg(diffusion_cfg_path, ckpt_path):
    """更新 diffusion yaml 中的 first_stage_config 字段"""
    with open(diffusion_cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_line = f"  first_stage_config: {ckpt_path}  # 由 find_best_ckpt.py 自动更新"
    content = re.sub(
        r'  first_stage_config:.*',
        new_line,
        content
    )

    with open(diffusion_cfg_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"已更新 {diffusion_cfg_path} -> {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='查找最佳 encoder checkpoint 并更新 diffusion 配置')
    parser.add_argument('--encoder_dir', type=str, default='results/photo-encoder',
                        help='Encoder 训练结果目录')
    parser.add_argument('--diffusion_cfg', type=str, default='cfg/photo-diffusion.yaml',
                        help='Diffusion 配置文件路径')
    args = parser.parse_args()

    print(f"扫描 encoder 训练结果: {args.encoder_dir}")
    ckpt_path, best_acc = find_best_encoder_ckpt(args.encoder_dir)
    update_diffusion_cfg(args.diffusion_cfg, ckpt_path)
