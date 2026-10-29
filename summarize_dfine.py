#!/usr/bin/env python3
"""Print the validation curve recorded by the official D-FINE trainer."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    log_path = args.run / "log.txt"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"no validation records in {log_path}")
    print(f"{'epoch':>5s} {'mAP':>8s} {'AP50':>8s} {'AP75':>8s} {'APs':>8s}")
    best = None
    for record in records:
        values = record["test_coco_eval_bbox"]
        print(
            f"{record['epoch'] + 1:5d} {values[0] * 100:8.2f} "
            f"{values[1] * 100:8.2f} {values[2] * 100:8.2f} {values[3] * 100:8.2f}"
        )
        if best is None or values[0] > best[1][0]:
            best = (record, values)
    record, values = best
    print("\nBest recorded checkpoint:")
    print(f"epoch: {record['epoch'] + 1}")
    print(f"mAP:   {values[0] * 100:.2f}%")
    print(f"AP50:  {values[1] * 100:.2f}%")
    print(f"AP75:  {values[2] * 100:.2f}%")
    print(f"APs:   {values[3] * 100:.2f}%")
    checkpoints = [p for p in (args.run / "best_stg2.pth", args.run / "best_stg1.pth") if p.exists()]
    if checkpoints:
        print("checkpoint:", checkpoints[0])


if __name__ == "__main__":
    main()
