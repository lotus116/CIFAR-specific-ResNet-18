"""最终模型的可复现训练与鲁棒性评估脚本。

示例：
    python train_experiments.py --variant basic --epochs 50
    python train_experiments.py --variant robust --epochs 60

``basic`` 与 ``robust`` 使用完全相同的网络，方便把性能差异归因到数据增强和
正则化。公共测试集仅在训练结束后报告，不参与 checkpoint 选择。
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from model import build_model


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def seed_everything(seed):
    """固定 Python、NumPy、CPU 与全部 CUDA 随机数生成器。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(data_root, batch_size, workers, seed, robust=False):
    """建立可复现的 45k/5k 固定划分及测试 loader。

    验证集和测试集严格使用 ToTensor + Normalize。训练集可根据实验
    选择基础增强或强增强，但二者使用同一组样本索引，保证横向比较公平。
    """
    eval_transform = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    if robust:
        # 强增强只作用于训练图像：RandAugment 模拟颜色和几何变化，RandomErasing
        # 模拟局部遮挡。它们不会改变评测预处理，也不会引入额外标注数据。
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
        ])
    else:
        # 基础增强是 CIFAR 常用配置：反射填充后随机裁剪，再随机水平翻转。
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    train_data = torchvision.datasets.CIFAR10(data_root, train=True, download=True,
                                               transform=train_transform)
    eval_data = torchvision.datasets.CIFAR10(data_root, train=True, download=False,
                                              transform=eval_transform)
    test_data = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
                                              transform=eval_transform)
    # 必须与 Notebook 完全一致：seed=42，前 5,000 个索引作为验证集。
    split = torch.randperm(50_000, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices, val_indices = split[5_000:], split[:5_000]
    generator = torch.Generator().manual_seed(seed)
    train_options = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                         persistent_workers=workers > 0)
    # Windows 下多份 CUDA worker 容易占用大量页面文件，评测采用单进程更稳定。
    eval_options = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data, train_indices), shuffle=True,
        generator=generator, drop_last=True, **train_options)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(eval_data, val_indices), shuffle=False, **eval_options)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **eval_options)
    return train_loader, val_loader, test_loader


def mixup(images, labels, alpha):
    """按 Beta(alpha, alpha) 混合一个 batch 内的图像及其两组监督目标。"""
    lam = np.random.beta(alpha, alpha)
    order = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[order], labels, labels[order], lam


@torch.inference_mode()
def evaluate(model, loader, device, corruption=None):
    """计算平均交叉熵和准确率，可选固定噪声或中心遮挡压力测试。

    corruption 只用于最终分析，不参与训练或最佳 checkpoint 的选择，因此不会
    向训练过程泄漏代理测试结果。
    """
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    corruption_generator = torch.Generator(device=device).manual_seed(2026)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if corruption == "noise":
            # 在已归一化的 [-1,1] 空间加入固定高斯噪声，seed 使结果可复现。
            noise = torch.randn(images.shape, device=device, generator=corruption_generator) * 0.15
            images = (images + noise).clamp(-1, 1)
        elif corruption == "occlusion":
            # 将图像中心 8x8 区域置为归一化后的中性值 0，模拟局部遮挡。
            images = images.clone()
            images[:, :, 12:20, 12:20] = 0
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return {"loss": loss_sum / total, "accuracy": 100 * correct / total}


def main():
    """解析配置、训练、按干净验证准确率保存最佳模型并完成一次最终评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("basic", "robust"), default="robust")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=80,
                        help="触发早停前至少完成的 epoch 数")
    parser.add_argument("--patience", type=int, default=20,
                        help="达到 min-epochs 后允许验证指标无实质提升的 epoch 数")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="./runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for these experiments")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    train_loader, val_loader, test_loader = make_loaders(
        args.data_root, args.batch_size, args.workers, args.seed, args.variant == "robust")
    model = build_model().to(device)
    # SGD + Nesterov 与 weight decay 是唯一优化器配置；学习率按 epoch 余弦衰减。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # torch.cuda.amp is retained for compatibility with the required PyTorch 2.0.1.
    scaler = torch.cuda.amp.GradScaler()
    output = Path(args.output) / args.variant
    output.mkdir(parents=True, exist_ok=True)
    history, best_accuracy, best_epoch = [], -1.0, 0
    # 早停与 checkpoint 使用两个阈值：任何提升都会保存，但只有超过 0.02 个百分点
    # 的实质提升才重置 patience，避免验证准确率的微小抖动无限延长训练。
    early_stop_best, stale_epochs, min_delta = -1.0, 0, 0.02
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.variant == "robust" and random.random() < 0.5:
                # 仅鲁棒实验以 50% 概率使用 Mixup；普通 batch 保留真实训练准确率。
                images, targets_a, targets_b, lam = mixup(images, labels, 0.2)
            else:
                targets_a = targets_b = labels
                lam = 1.0
            with torch.cuda.amp.autocast():
                logits = model(images)
                smoothing = 0.1 if args.variant == "robust" else 0.0
                loss = (lam * F.cross_entropy(logits, targets_a, label_smoothing=smoothing)
                        + (1 - lam) * F.cross_entropy(logits, targets_b, label_smoothing=smoothing))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
        scheduler.step()
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum / total, "train_accuracy": 100 * correct / total,
               "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
        history.append(row)
        if val["accuracy"] > best_accuracy:
            # 模型选择只看干净验证集，不看公共测试集或合成腐蚀集。
            best_accuracy, best_epoch = val["accuracy"], epoch
            torch.save(model.state_dict(), output / "best.pt")
        if val["accuracy"] > early_stop_best + min_delta:
            early_stop_best, stale_epochs = val["accuracy"], 0
        else:
            stale_epochs += 1
        with (output / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(f"{args.variant} {epoch:03d}/{args.epochs} "
              f"train {row['train_accuracy']:.2f}% val {val['accuracy']:.2f}% "
              f"best {best_accuracy:.2f}%", flush=True)
        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}: validation accuracy had no "
                  f"improvement > {min_delta:.2f} points for {stale_epochs} epochs.",
                  flush=True)
            break

    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    results = {
        "variant": args.variant, "seed": args.seed,
        "epochs_requested": args.epochs, "epochs_ran": len(history),
        "early_stopping": {"min_epochs": args.min_epochs,
                           "patience": args.patience, "min_delta": min_delta},
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "best_validation": evaluate(model, val_loader, device),
        "validation_noise": evaluate(model, val_loader, device, "noise"),
        "validation_occlusion": evaluate(model, val_loader, device, "occlusion"),
        "public_test": evaluate(model, test_loader, device),
        "elapsed_minutes": (time.time() - started) / 60,
    }
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
