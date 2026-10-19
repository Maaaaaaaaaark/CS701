"""Fully fine-tune one CSPNeXt-m/RTMDet model for SAR classification and detection."""
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
from sarbench.rtmdet import RTMDetMultiTask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('cs701-sar-course-data'))
    parser.add_argument('--epochs', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4, help='neck and task-head learning rate')
    parser.add_argument('--backbone-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--eval-every', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--scratch', action='store_true', help='do not load the ImageNet CSPNeXt-m weights')
    parser.add_argument('--keep-last-checkpoints', type=int, default=3)
    parser.add_argument('--resume', type=Path, help='resume a complete RTMDet training-state checkpoint')
    parser.add_argument('--out', type=Path, default=Path('runs'))
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_every < 1:
        parser.error('--epochs and --eval-every must be positive')
    if not 0 < args.val_fraction < 1:
        parser.error('--val-fraction must be between 0 and 1')
    if args.keep_last_checkpoints < 0:
        parser.error('--keep-last-checkpoints cannot be negative')
    return args


def write_log(message, file):
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split):
    train = split == 'train'
    dataset = SARMultiTask(
        args.data, split, train=train, limit=args.limit,
        val_fraction=args.val_fraction, split_seed=args.split_seed, rotation90=False)
    return DataLoader(
        dataset, args.batch_size, shuffle=train, num_workers=args.workers,
        collate_fn=collate, persistent_workers=train and args.workers > 0,
        pin_memory=True)


def lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def train_one_epoch(model, loader, optimizer, scheduler, epoch, log_file):
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        targets = [{'boxes': target['boxes'].cuda(non_blocking=True),
                    'labels': target['labels'].cuda(non_blocking=True)} for target in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, detection_losses = model(images, targets)
            losses = {'classification': F.cross_entropy(logits, labels), **detection_losses}
            loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            rates = '/'.join(f'{group["name"]}:{group["lr"]:.2e}' for group in optimizer.param_groups)
            write_log(
                f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {rates} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.inference_mode()
def evaluate(model, loader, gt_json):
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, expected, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda(non_blocking=True))
        labels += expected.tolist()
        predicted += logits.argmax(1).cpu().tolist()
        for target, output in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(output['boxes'].cpu(), target['orig_size']).tolist()
            detections += [
                {'image_id': target['image_id'], 'category_id': int(category),
                 'bbox': box, 'score': float(score)}
                for box, category, score in zip(
                    boxes, output['labels'].cpu().tolist(), output['scores'].cpu().tolist())]
    metrics = {**classification_metrics(labels, predicted),
               **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections


def headline(metrics):
    keys = ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50', 'AP75')
    return ' '.join(f'{key} {metrics[key]:.4f}' for key in keys)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_name = f'rtmdet_m_imagenet_full_e{args.epochs}'
    if args.seed:
        run_name += f'_seed{args.seed}'
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'a' if args.resume else 'w')
    write_log(json.dumps(vars(args), default=str), log_file)

    train_loader, val_loader = (make_loader(args, split) for split in ('train', 'val'))
    (run_dir / 'split.json').write_text(json.dumps({
        'split_seed': args.split_seed, 'val_fraction': args.val_fraction,
        'train_image_ids': train_loader.dataset.image_ids,
        'val_image_ids': val_loader.dataset.image_ids,
    }, indent=2))

    model = RTMDetMultiTask(pretrained=not args.scratch)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    write_log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)
    if parameters > 130_000_000 or trainable > 40_000_000:
        raise ValueError(f'course budget exceeded: {parameters:,} total, {trainable:,} trainable')
    if parameters != trainable:
        raise ValueError('RTMDet full fine-tuning expected every parameter to be trainable')
    model.cuda()

    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone_parameters, 'lr': args.backbone_lr, 'name': 'backbone'},
        {'params': head_parameters, 'lr': args.lr, 'name': 'heads'},
    ], weight_decay=args.weight_decay)
    warmup = len(train_loader)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup, total_steps))

    start_epoch, history = 1, []
    best_map, best_map_epoch, best_val = float('-inf'), None, None
    elapsed_seconds = 0.0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        history = state['history']
        best_map = state['best_mAP']
        best_map_epoch = state['best_mAP_epoch']
        best_val = state['best_val']
        elapsed_seconds = state.get('elapsed_seconds', 0.0)
        write_log(f'resumed after epoch {start_epoch - 1} from {args.resume}', log_file)

    evaluation_epochs = [epoch for epoch in range(1, args.epochs + 1)
                         if epoch % args.eval_every == 0 or epoch == args.epochs]
    retained_epochs = set(evaluation_epochs[-args.keep_last_checkpoints:]) if args.keep_last_checkpoints else set()
    segment_start = time.time()
    val_detections = None
    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scheduler, epoch, log_file)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(model, val_loader, val_loader.dataset.gt_json)
            history.append({'epoch': epoch, **val})
            write_log(f'epoch {epoch} val: {headline(val)}', log_file)
            if val['mAP'] > best_map:
                best_map, best_map_epoch, best_val = val['mAP'], epoch, {'epoch': epoch, **val}
                torch.save(model.state_dict(), run_dir / 'model_best_map.pt')
                write_log(f'epoch {epoch}: saved model_best_map.pt', log_file)
            if epoch in retained_epochs:
                torch.save(model.state_dict(), run_dir / f'model_epoch_{epoch:03d}.pt')
                write_log(f'epoch {epoch}: saved retained checkpoint', log_file)
            current_elapsed = elapsed_seconds + time.time() - segment_start
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'history': history, 'best_mAP': best_map, 'best_mAP_epoch': best_map_epoch,
                'best_val': best_val, 'elapsed_seconds': current_elapsed,
            }, run_dir / 'training_state_latest.pt')
            write_log(f'epoch {epoch}: saved resumable training state', log_file)

    train_hours = (elapsed_seconds + time.time() - segment_start) / 3600
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    write_log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)
    summary = {
        'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
        'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
        'best_mAP': best_map, 'best_mAP_epoch': best_map_epoch, 'best_val': best_val,
        'val_history': history, 'val': history[-1], 'test': None,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    torch.save(model.state_dict(), run_dir / 'model.pt')


if __name__ == '__main__':
    main()
