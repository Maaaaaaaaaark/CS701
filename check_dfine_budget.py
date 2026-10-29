#!/usr/bin/env python3
"""Build the configured D-FINE model and enforce the CS701 parameter budgets."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(".dfine/source"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-total", type=int, default=130_000_000)
    parser.add_argument("--max-trainable", type=int, default=40_000_000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config.resolve()))
    model = cfg.model
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total parameters:     {total / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable / 1e6:.3f} M")
    if total > args.max_total:
        raise RuntimeError(f"total parameter budget exceeded: {total} > {args.max_total}")
    if trainable > args.max_trainable:
        raise RuntimeError(
            f"trainable parameter budget exceeded: {trainable} > {args.max_trainable}"
        )
    print("CS701 parameter budget check passed")


if __name__ == "__main__":
    main()
