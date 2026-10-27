"""Fully fine-tune one RF-DETR model for SAR classification and detection."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.rfdetr import (
    RFDETR_IMAGE_MEAN,
    RFDETR_IMAGE_STD,
    RFDETRMultiTask,
    prepare_detection_targets,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data"))
    parser.add_argument("--variant", choices=("small", "medium"), default="small")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4, help="physical batch size per GPU")
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4, help="detector and new classification-head LR")
    parser.add_argument("--encoder-lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument(
        "--box-loss-scale",
        type=float,
        default=1.0,
        help="extra multiplier for all weighted bbox and GIoU losses",
    )
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--resume", type=Path, help="resume a complete RF-DETR training-state checkpoint")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument(
        "--run-name",
        help="optional output directory name; defaults to the variant and epoch count",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_every < 1 or args.grad_accum_steps < 1:
        parser.error("--epochs, --eval-every and --grad-accum-steps must be positive")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if args.keep_last_checkpoints < 0:
        parser.error("--keep-last-checkpoints cannot be negative")
    if args.classification_weight < 0 or args.box_loss_scale <= 0:
        parser.error("--classification-weight must be non-negative and --box-loss-scale positive")
    return args


def write_log(message, file):
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split):
    train = split == "train"
    dataset = SARMultiTask(
        args.data,
        split,
        train=train,
        limit=args.limit,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        rotation90=False,
        mean=RFDETR_IMAGE_MEAN,
        std=RFDETR_IMAGE_STD,
    )
    return DataLoader(
        dataset,
        args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        collate_fn=collate,
        persistent_workers=train and args.workers > 0,
        pin_memory=True,
    )


def lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def train_one_epoch(model, loader, optimizer, scheduler, epoch, args, log_file):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start, seen = time.time(), 0
    remainder = len(loader) % args.grad_accum_steps
    for step, (images, labels, raw_targets) in enumerate(loader, 1):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        targets = prepare_detection_targets(raw_targets)
        targets = [
            {name: value.cuda(non_blocking=True) for name, value in target.items()}
            for target in targets
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, outputs = model(images)
            classification_loss = F.cross_entropy(logits, labels)
            detection_loss, detection_parts = model.detection_losses(
                outputs,
                targets,
                box_loss_scale=args.box_loss_scale,
            )
            loss = args.classification_weight * classification_loss + detection_loss

        in_final_partial_window = remainder and step > len(loader) - remainder
        divisor = remainder if in_final_partial_window else args.grad_accum_steps
        (loss / divisor).backward()
        update = step % args.grad_accum_steps == 0 or step == len(loader)
        if update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            lrs = [group["lr"] for group in optimizer.param_groups]
            core = " ".join(
                f"{name} {value.item():.3f}"
                for name, value in detection_parts.items()
                if name in {"loss_ce", "loss_bbox", "loss_giou"}
            )
            write_log(
                f"epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} "
                f"(classification {classification_loss.item():.3f} detection {detection_loss.item():.3f} {core}) "
                f"lr min:{min(lrs):.2e}/max:{max(lrs):.2e} "
                f"{seen / max(time.time() - start, 1e-6):.1f} images/s",
                log_file,
            )
            start, seen = time.time(), 0


@torch.inference_mode()
def evaluate(model, loader, gt_json):
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, expected, targets in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, outputs = model(images.cuda(non_blocking=True))
        batch_detections = model.detections(outputs)
        labels += expected.tolist()
        predicted += logits.argmax(1).cpu().tolist()
        for target, output in zip(targets, batch_detections):
            image_ids.append(target["image_id"])
            boxes = to_original_xywh(output["boxes"].float().cpu(), target["orig_size"]).tolist()
            detections += [
                {
                    "image_id": target["image_id"],
                    "category_id": int(category) + 1,
                    "bbox": box,
                    "score": float(score),
                }
                for box, category, score in zip(
                    boxes,
                    output["labels"].cpu().tolist(),
                    output["scores"].float().cpu().tolist(),
                )
            ]
    metrics = {
        **classification_metrics(labels, predicted),
        **detection_metrics(gt_json, detections, image_ids),
    }
    return metrics, detections


def headline(metrics):
    keys = ("accuracy", "macro_f1", "balanced_accuracy", "mAP", "AP50", "AP75")
    return " ".join(f"{key} {metrics[key]:.4f}" for key in keys)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_name = args.run_name or f"rfdetr_{args.variant}_coco_full_e{args.epochs}"
    if args.seed and not args.run_name:
        run_name += f"_seed{args.seed}"
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "log.txt", "a" if args.resume else "w")
    write_log(json.dumps(vars(args), default=str), log_file)

    train_loader, val_loader = (make_loader(args, split) for split in ("train", "val"))
    (run_dir / "split.json").write_text(
        json.dumps(
            {
                "split_seed": args.split_seed,
                "val_fraction": args.val_fraction,
                "train_image_ids": train_loader.dataset.image_ids,
                "val_image_ids": val_loader.dataset.image_ids,
            },
            indent=2,
        )
    )

    model = RFDETRMultiTask(pretrained=True, variant=args.variant, resolution=512)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    write_log(f"parameters: {parameters:,} total, {trainable:,} trainable", log_file)
    if parameters > 130_000_000 or trainable > 40_000_000:
        raise ValueError(f"course budget exceeded: {parameters:,} total, {trainable:,} trainable")
    if parameters != trainable:
        raise ValueError("RF-DETR full fine-tuning expected every parameter to be trainable")
    model.cuda()

    parameter_groups = model.optimizer_parameter_groups(args.lr, args.encoder_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    warmup = optimizer_steps_per_epoch
    total_steps = args.epochs * optimizer_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup, total_steps)
    )

    start_epoch, history = 1, []
    best_map, best_map_epoch, best_val = float("-inf"), None, None
    elapsed_seconds = 0.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        history = state["history"]
        best_map = state["best_mAP"]
        best_map_epoch = state["best_mAP_epoch"]
        best_val = state["best_val"]
        elapsed_seconds = state.get("elapsed_seconds", 0.0)
        write_log(f"resumed after epoch {start_epoch - 1} from {args.resume}", log_file)

    evaluation_epochs = [
        epoch
        for epoch in range(1, args.epochs + 1)
        if epoch % args.eval_every == 0 or epoch == args.epochs
    ]
    retained_epochs = (
        set(evaluation_epochs[-args.keep_last_checkpoints :]) if args.keep_last_checkpoints else set()
    )
    segment_start = time.time()
    val_detections = None
    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scheduler, epoch, args, log_file)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(model, val_loader, val_loader.dataset.gt_json)
            history.append({"epoch": epoch, **val})
            write_log(f"epoch {epoch} val: {headline(val)}", log_file)
            if val["mAP"] > best_map:
                best_map, best_map_epoch, best_val = val["mAP"], epoch, {"epoch": epoch, **val}
                torch.save(model.state_dict(), run_dir / "model_best_map.pt")
                write_log(f"epoch {epoch}: saved model_best_map.pt", log_file)
            if epoch in retained_epochs:
                torch.save(model.state_dict(), run_dir / f"model_epoch_{epoch:03d}.pt")
                write_log(f"epoch {epoch}: saved retained checkpoint", log_file)
            current_elapsed = elapsed_seconds + time.time() - segment_start
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_mAP": best_map,
                    "best_mAP_epoch": best_map_epoch,
                    "best_val": best_val,
                    "elapsed_seconds": current_elapsed,
                },
                run_dir / "training_state_latest.pt",
            )
            write_log(f"epoch {epoch}: saved resumable training state", log_file)

    if not history:
        raise RuntimeError("no validation result was produced")
    train_hours = (elapsed_seconds + time.time() - segment_start) / 3600
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    write_log(f"training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB", log_file)
    summary = {
        "args": vars(args),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "train_hours": train_hours,
        "peak_gpu_memory_gb": peak_gb,
        "best_mAP": best_map,
        "best_mAP_epoch": best_map_epoch,
        "best_val": best_val,
        "val_history": history,
        "val": history[-1],
        "test": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "predictions_val.json").write_text(json.dumps(val_detections))
    torch.save(model.state_dict(), run_dir / "model.pt")


if __name__ == "__main__":
    main()
