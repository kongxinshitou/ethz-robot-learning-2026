"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 600
BATCH_SIZE = 320
LR = 1e-3
VAL_SPLIT = 0.1


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )

    parser.add_argument(
        "--d-model",
        "--hiden-dim",
        dest="d_model",
        type=int,
        default=256,
        help="MLP hidden width (alias: --hiden-dim), default 256.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Number of MLP hidden layers, default 3.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate (default: {LR}).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=VAL_SPLIT,
        help=f"Validation fraction (default: {VAL_SPLIT}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override the checkpoint output directory "
        "(default: ./checkpoints/single_cube or ./checkpoints/multi_cube).",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Optional .pt checkpoint to warm-start the model weights from, "
        "e.g. the exercise-1 checkpoint when retraining for exercise 2.",
    )

    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--extra-zarr", type=Path, nargs="*", default=[])

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # eval_utils.load_checkpoint 需要 state_keys 才能把 obs 拼成同样的输入；
    # 不传的话 checkpoint 里 state_keys=None，评估时会直接报错。
    if args.state_keys is None or args.action_keys is None:
        raise SystemExit(
            "请显式指定 --state-keys 和 --action-keys（不要依赖 zarr metadata 的默认值），例如：\n"
            '  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle '
            "--action-keys action_ee_xyz action_gripper"
        )

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optional warm start (useful for ex2: continue from the ex1 checkpoint).
    if args.init_from is not None:
        try:
            init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
            model.load_state_dict(init_ckpt["model_state_dict"])
            print(f"Warm-started model weights from {args.init_from}")
        except Exception as exc:  # shape mismatch / missing file -> train from scratch
            print(f"WARNING: could not warm-start from {args.init_from}: {exc}")
            print("         Training from scratch instead.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epochs
    )
    print(
        f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, val_split={args.val_split}"
    )
    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if args.checkpoint_dir is not None:
        ckpt_dir = args.checkpoint_dir
    elif "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "epochs": args.epochs,
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
