#!/usr/bin/env python3
"""Train a minimal GCN baseline on MPI-INF-3DHP using existing data utilities."""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common.fusion import Fusion
from common.loss import mpjpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple GCN trainer for MPI-INF-3DHP")
    parser.add_argument("--data-root", type=str, default="./", help="Directory containing data_train_<mode>.npz")
    parser.add_argument("--mode", type=str, default="3dhp", help="Dataset variant suffix, e.g. 3dhp or univ_3dhp")
    parser.add_argument("--dataset", type=str, default="3dhp", help="Logical dataset name (keep default)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between logging training loss")
    parser.add_argument("--eval-every", type=int, default=1, help="Epoch interval for validation evaluation")
    parser.add_argument("--save-path", type=str, default="", help="Optional path to store the best checkpoint")
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA if available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of the training dataset to use (0, 1].")
    # Dataset loader knobs expected by Fusion
    parser.add_argument("--actions", type=str, default="*", help="Action filter, * for all")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--data-augmentation", action="store_true")
    parser.add_argument("--reverse-augmentation", action="store_true")
    parser.add_argument("--out-all", type=int, default=0)
    parser.add_argument("--test-augmentation", action="store_true")
    parser.add_argument("--num-cameras", type=int, nargs="*", default=None,
                        help="Optional subset of camera ids to use")
    return parser.parse_args()


def build_mpi3dhp_adjacency(num_joints: int = 16) -> torch.Tensor:
    edges = torch.tensor([
        [0, 1], [1, 2], [2, 3],
        [3, 4], [1, 5], [5, 6],
        [6, 7], [1, 15], [15, 14],
        [14, 11], [11, 12], [12, 13],
        [14, 8], [8, 9], [9, 10],
    ], dtype=torch.long)
    adj = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj += torch.eye(num_joints, dtype=torch.float32)
    degree = torch.sum(adj, dim=1, keepdim=True)
    degree = torch.where(degree > 0, degree, torch.ones_like(degree))
    adj_norm = adj / degree
    return adj_norm


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.matmul(adj, self.linear(x))


class SimplePoseGCN(nn.Module):
    def __init__(self, adj: torch.Tensor, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("adjacency", adj)
        self.layer1 = nn.Linear(2, hidden_dim)
        self.layer2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.layer3 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.layer4 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, 3)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = self.adjacency
        x = self.layer1(x)
        x = self.layer2(x, adj) + x
        x = self.act(x)
        x = self.drop(x)
        x = self.layer3(x, adj) + x
        x = self.act(x)
        x = self.drop(x)
        x = self.layer4(x, adj) + x
        x = self.act(x)
        x = self.drop(x)
        x = self.layer5(x)
        return x


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subset_dataset(dataset, fraction: float, seed: int):
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError("subset fraction must be in (0, 1].")
    total = len(dataset)
    keep = max(1, int(total * fraction))
    rng = np.random.RandomState(seed)
    indices = rng.choice(total, size=keep, replace=False)
    indices.sort()
    subset = Subset(dataset, indices.tolist())
    percent = fraction * 100.0
    print(f"INFO: Training on {len(subset)} frames after applying subset ({percent:.1f}% of data)")
    return subset


def prepare_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = Fusion(opt=args, root_path=args.data_root, train=True)
    train_dataset = maybe_subset_dataset(train_dataset, args.subset, args.seed)
    val_dataset = Fusion(opt=args, root_path=args.data_root, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def preprocess_inputs(inputs_2d: torch.Tensor) -> torch.Tensor:
    if inputs_2d.dim() == 2:
        inputs_2d = inputs_2d.unsqueeze(0)
    inputs = inputs_2d[..., :2]
    return inputs


def preprocess_targets(targets_3d: torch.Tensor) -> torch.Tensor:
    if targets_3d.dim() == 2:
        targets_3d = targets_3d.unsqueeze(0)
    targets_3d[:, 14, :] = 0
    return targets_3d


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, log_interval: int) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        _, target_3d, input_2d, *_ = batch
        input_2d = preprocess_inputs(input_2d.float()).to(device)
        target_3d = preprocess_targets(target_3d.float()).to(device)

        optimizer.zero_grad()
        pred_3d = model(input_2d)
        loss = mpjpe(pred_3d, target_3d)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        # if log_interval and step % log_interval == 0:
        #     elapsed = time.time() - start_time
        #     print(f"  step {step:05d} | mpjpe {loss.item() * 1000:.2f} mm | {elapsed:.1f}s")
        #     start_time = time.time()

    mean_loss = total_loss / max(total_batches, 1)
    return mean_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        _, target_3d, input_2d, *_ = batch
        input_2d = preprocess_inputs(input_2d.float()).to(device)
        target_3d = preprocess_targets(target_3d.float()).to(device)
        pred_3d = model(input_2d)
        loss = mpjpe(pred_3d, target_3d)
        losses.append(loss.item())
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def maybe_save_checkpoint(model: nn.Module, path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(model.state_dict(), path)


def main() -> None:
    args = parse_args()
    if not 0 < args.subset <= 1:
        raise ValueError("--subset must be in the interval (0, 1].")
    set_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    adjacency = build_mpi3dhp_adjacency().to(device)
    model = SimplePoseGCN(adjacency, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, val_loader = prepare_dataloaders(args)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.log_interval)
        print(f"  train mpjpe: {train_loss * 1000:.2f} mm")

        if epoch % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  val mpjpe: {val_loss * 1000:.2f} mm")
            if val_loss < best_val:
                best_val = val_loss
                maybe_save_checkpoint(model, os.path.join(args.save_path,"best_checkpoint.pth"))
                if args.save_path:
                    print(f"  saved new best model to {args.save_path}")

    print(f"Best validation MPJPE: {best_val * 1000:.2f} mm")


if __name__ == "__main__":
    main()
