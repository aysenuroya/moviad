"""
DiAD "unified case" reproduction entrypoint for MoViAD.
"""

import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.diad.diad import DiAD, DiADConfig, DiADTrainArgs
from moviad.models.diad.weights import load_stable_diffusion
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.metrics import RocAuc, AvgPrec, F1, ProAuc, MetricLvl


def _min_max_norm(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


@dataclass
class DiADUnifiedArgs:
    dataset_path: str
    categories: Optional[List[str]] = None
    img_input_size: Tuple[int, int] = (256, 256)
    sd_checkpoint_path: Optional[str] = None
    autoencoder_path: Optional[str] = None
    config: DiADConfig = field(default_factory=DiADConfig)
    batch_size: int = 12
    test_batch_size: int = 4
    epochs: int = 1000
    lr: float = 1e-5
    weight_decay: float = 1e-2
    accumulate_grad_batches: int = 4
    eval_every: int = 25
    num_workers: int = 0
    device: torch.device = None
    save_path: Optional[str] = None


def _evaluate(model: DiAD, dataloader: DataLoader, metrics, device: torch.device) -> dict:
    model.eval()
    gt_mask, gt_label, pred_map, pred_score = [], [], [], []
    for images, label, mask, _path in dataloader:
        with torch.no_grad():
            anomaly_maps, anomaly_scores = model(images.to(device))
        gt_mask.append(mask.cpu().numpy().astype(int))
        gt_label.append(label.cpu().numpy())
        pred_map.append(anomaly_maps.cpu().numpy())
        pred_score.append(anomaly_scores.cpu().numpy())

    gt_mask = np.concatenate(gt_mask)
    gt_label = np.concatenate(gt_label)
    pred_map = _min_max_norm(np.concatenate(pred_map))
    pred_score = np.concatenate(pred_score)

    report = {}
    for metric in metrics:
        gt, pred = (gt_label, pred_score) if metric.level == MetricLvl.IMAGE else (gt_mask, pred_map)
        report[metric.name] = metric.compute(gt, pred)
    return report


def train_diad_unified(args: DiADUnifiedArgs, logger=None) -> Tuple[Dict[str, dict], dict]:
    categories = args.categories or MVTecDataset.get_categories()
    dataset_args = DatasetArguments(
        dataset_path=args.dataset_path,
        img_size=args.img_input_size,
        gt_mask_size=args.img_input_size,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(args.img_input_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    train_datasets = [MVTecDataset(dataset_args, category=c, split=Split.TRAIN) for c in categories]
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loaders = {
        c: DataLoader(
            MVTecDataset(dataset_args, category=c, split=Split.TEST),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for c in categories
    }

    model = DiAD(args.config)
    if args.sd_checkpoint_path is not None:
        load_stable_diffusion(model, args.sd_checkpoint_path, args.autoencoder_path)
    else:
        print("WARNING: no Stable Diffusion checkpoint given, the model starts from random weights")
    model.to(args.device)

    train_args = DiADTrainArgs(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    train_args.init_train(model)

    if logger is not None:
        logger.config.update(train_args.__to_dict__())

    metrics = [
        RocAuc(MetricLvl.IMAGE), AvgPrec(MetricLvl.IMAGE), F1(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL), AvgPrec(MetricLvl.PIXEL), F1(MetricLvl.PIXEL), ProAuc(MetricLvl.PIXEL),
    ]

    def evaluate_all_categories(epoch: int):
        per_category = {c: _evaluate(model, loader, metrics, args.device) for c, loader in test_loaders.items()}
        for report in per_category.values():
            report["mAD"] = float(np.mean(list(report.values())))
        mean_metrics = {m.name: float(np.mean([per_category[c][m.name] for c in categories])) for m in metrics}
        mean_metrics["mAD"] = float(np.mean(list(mean_metrics.values())))
        if logger is not None:
            log_dict = {f"{c}/{k}": v for c, r in per_category.items() for k, v in r.items()}
            log_dict.update({f"mean/{k}": v for k, v in mean_metrics.items()})
            log_dict["epoch"] = epoch
            logger.log(log_dict)
        return per_category, mean_metrics

    for epoch in range(args.epochs):
        model.train()
        avg_loss = model.train_epoch(epoch, train_loader, train_args)

        if logger is not None:
            logger.log({"epoch": epoch, "train_loss": avg_loss})
        print(f"[epoch {epoch}] train_loss={avg_loss:.5f}")

        if (epoch + 1) % args.eval_every == 0 and epoch != args.epochs - 1:
            _, mean_metrics = evaluate_all_categories(epoch)
            print(f"[epoch {epoch}] loss={avg_loss:.5f} mean={mean_metrics}")

    per_category, mean_metrics = evaluate_all_categories(args.epochs - 1)

    print("\nFinal per-category results:")
    for c in categories:
        print(f"  {c}: {per_category[c]}")
    print(f"Mean: {mean_metrics}")

    if args.save_path:
        model.save(args.save_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return per_category, mean_metrics
