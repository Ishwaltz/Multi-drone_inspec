"""Thin BridgeInspection training wrapper for the GCBF+ training entry point."""

# Extends GCBF+ (MIT-REALM/gcbfplus), MIT License.
# Original authors: Zhang, So, Garg, Fan (IEEE T-RO 2025).
# This file is an extension; original algorithm is unmodified.

import argparse
import datetime
import os
from pathlib import Path

import orbax.checkpoint as ocp

import train as gcbf_train


def main():
    """Parse BridgeInspection fine-tuning arguments and call train.py.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=4)
    parser.add_argument("--algo", type=str, default="gcbf+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, default=200.0)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--loss-action-coef", type=float, default=0.0001)
    parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--loss-safe-coef", type=float, default=1.0)
    parser.add_argument("--loss-h-dot-coef", type=float, default=0.01)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--n-env-train", type=int, default=16)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--finetune-lr-actor", type=float, default=1e-4)
    parser.add_argument("--finetune-lr-cbf", type=float, default=1e-4)
    parser.add_argument("--domain-randomize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.env = "BridgeInspection"
    now = datetime.datetime.now()
    if args.pretrained_path:
        checkpointer = ocp.PyTreeCheckpointer()
        print(f"Verified checkpoint reader for {args.pretrained_path}: {checkpointer.__class__.__name__}")
        args.lr_actor = args.finetune_lr_actor
        args.lr_cbf = args.finetune_lr_cbf
        print(f"Loaded pretrained weights from {args.pretrained_path}. Fine-tuning with barrier network UNFROZEN at lr={args.lr_cbf}.")
        print(f"Fine-tuning started: {now}")
    else:
        print("No pretrained checkpoint. Training from scratch.")
        print(f"Training started: {now}")
    start_dir = Path(args.log_dir) / args.env / args.algo
    start_dir.mkdir(parents=True, exist_ok=True)
    (start_dir / f"seed{args.seed}_train_start_time.txt").write_text(now.isoformat())
    gcbf_train.train(args)


if __name__ == "__main__":
    main()
