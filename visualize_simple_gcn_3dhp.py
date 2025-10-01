#!/usr/bin/env python3
"""Visualize SimplePoseGCN predictions on the MPI-INF-3DHP dataset."""

import argparse
import os
import subprocess
from typing import Iterable, Tuple

import numpy as np
import torch

from common.fusion import Fusion
from common.skeleton import Skeleton
from common.visualization import render_animation_3dhp
from main_simple_gcn_3dhp import (
    SimplePoseGCN,
    build_mpi3dhp_adjacency,
    preprocess_inputs,
    set_seed,
)
from matplotlib.animation import writers


# MPI3DHP_PARENTS = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14]
MPI3DHP_PARENTS = [-1, 0, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 15, 1]
# MPI3DHP_PARENTS = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14]
MPI3DHP_JOINTS_LEFT = [5, 6, 7, 8, 9, 10]
MPI3DHP_JOINTS_RIGHT = [2, 3, 4, 11, 12, 13]
MPI3DHP_SKELETON = Skeleton(
    parents=MPI3DHP_PARENTS,
    joints_left=MPI3DHP_JOINTS_LEFT,
    joints_right=MPI3DHP_JOINTS_RIGHT,
)

ROOT_INDEX = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Simple GCN predictions on MPI-INF-3DHP")
    parser.add_argument("--data-root", type=str, default="./", help="Directory containing data_(train|test)_<mode>.npz")
    parser.add_argument("--mode", type=str, default="3dhp", help="Dataset variant suffix, e.g. 3dhp or univ_3dhp")
    parser.add_argument("--dataset", type=str, default="3dhp", help="Logical dataset name (keep default)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained SimplePoseGCN checkpoint")
    parser.add_argument("--output", type=str, default="simple_gcn_vis.mp4", help="Output animation file (.mp4 or .gif)")
    parser.add_argument("--sequence", type=str, default="", help="Optional identifier to pick a specific sequence")
    parser.add_argument("--num-frames", type=int, default=240, help="Number of frames to render")
    parser.add_argument("--skip", type=int, default=0, help="Number of initial frames to skip")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--bitrate", type=int, default=3000, help="Output video bitrate")
    parser.add_argument("--azim", type=float, default=0.0, help="Azimuth angle for the 3D view")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Choose between train/test split")
    parser.add_argument("--use-cuda", action="store_true", help="Run inference on CUDA if available")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension used by the trained model")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used by the trained model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size passed to the dataset generator")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--data-augmentation", action="store_true")
    parser.add_argument("--reverse-augmentation", action="store_true")
    parser.add_argument("--out-all", type=int, default=0)
    parser.add_argument("--test-augmentation", action="store_true")
    parser.add_argument("--num-cameras", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return (X + [1, h / w]) * w / 2

def ensure_time_dim(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None]
    if array.ndim == 3:
        return array
    raise ValueError(f"Unexpected array shape {array.shape}")


def infer_viewport(sequence: str, split: str) -> Tuple[int, int]:
    if split == "train":
        return 2048, 2048
    if sequence in {"TS5", "TS6"}:
        return 1920, 1080
    return 2048, 2048


def iter_dataset(dataset: Fusion, split: str) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if split == "train":
            _, gt_3d, input_2d, seq, subject, _, _, cam_ind = sample
            seq_name = f"{subject}_{seq}_cam{cam_ind}"
        else:
            _, gt_3d, input_2d, seq, *_ = sample
            seq_name = seq
        yield seq_name, np.asarray(gt_3d), np.asarray(input_2d)


def collect_sequence(dataset: Fusion, model: SimplePoseGCN, device: torch.device, args: argparse.Namespace):
    keypoints_norm = []
    preds_3d = []
    gts_3d = []
    sequence_name = None
    seen_frames = 0

    with torch.no_grad():
        for seq_name, gt_3d, input_2d in iter_dataset(dataset, args.split):
            if args.sequence and seq_name != args.sequence:
                continue
            if sequence_name is None:
                sequence_name = seq_name
            elif seq_name != sequence_name:
                if len(keypoints_norm) >= args.num_frames:
                    break
                else:
                    continue

            inputs = ensure_time_dim(np.asarray(input_2d, dtype=np.float32))
            targets_all = ensure_time_dim(np.asarray(gt_3d, dtype=np.float32))

            for frame_idx in range(inputs.shape[0]):
                if seen_frames < args.skip:
                    seen_frames += 1
                    continue

                frame_2d = inputs[frame_idx, :, :2]
                target_frame = targets_all[frame_idx].copy()
                trajactory = target_frame[ROOT_INDEX:ROOT_INDEX + 1, :].copy()
                tensor_2d = torch.from_numpy(frame_2d).unsqueeze(0).to(device)
                model_in = preprocess_inputs(tensor_2d)
                pred = model(model_in).cpu().numpy()[0]
                pred[ROOT_INDEX, :] = 0
                pred += trajactory
                target_frame[ROOT_INDEX, :] = 0
                target_frame += trajactory

                keypoints_norm.append(frame_2d)
                preds_3d.append(pred)
                gts_3d.append(target_frame)
                seen_frames += 1

                if len(keypoints_norm) >= args.num_frames:
                    break

            if len(keypoints_norm) >= args.num_frames:
                break

    if not keypoints_norm:
        raise RuntimeError("No frames collected; adjust --sequence/--skip/--num-frames.")

    return (
        sequence_name,
        np.stack(keypoints_norm),
        np.stack(preds_3d),
        np.stack(gts_3d),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    adjacency = build_mpi3dhp_adjacency().to(device)
    model = SimplePoseGCN(adjacency, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    dataset = Fusion(opt=args, root_path=args.data_root, train=args.split == "train")

    sequence_name, keypoints_norm, preds_3d, gts_3d = collect_sequence(dataset, model, device, args)
    viewport_w, viewport_h = infer_viewport(sequence_name, args.split)

    keypoints_px = image_coordinates(
        keypoints_norm.reshape(-1, 2), viewport_w, viewport_h
    ).reshape(keypoints_norm.shape)

    pred_min, pred_max = preds_3d.min(), preds_3d.max()
    gt_min, gt_max = gts_3d.min(), gts_3d.max()
    print(f"Prediction range: [{pred_min:.4f}, {pred_max:.4f}] | GT range: [{gt_min:.4f}, {gt_max:.4f}]")

    poses = {
        "Prediction": preds_3d,
        "Ground Truth": gts_3d,
    }

    mpjpe_val = np.mean(np.linalg.norm(preds_3d - gts_3d, axis=-1)) * 1000.0
    print(f"Sequence: {sequence_name} | Frames: {preds_3d.shape[0]} | MPJPE: {mpjpe_val:.2f} mm")

    desired_output = args.output
    fallback_output = None
    if desired_output.lower().endswith(".mp4"):
        fallback_output = os.path.splitext(desired_output)[0] + ".gif"
        if not writers.is_available("ffmpeg"):
            print(f"FFmpeg not available in this environment, falling back to GIF: {fallback_output}")
            desired_output = fallback_output

    tried_outputs = []
    final_output = None
    for candidate in [desired_output] + ([fallback_output] if fallback_output and desired_output != fallback_output else []):
        if candidate is None:
            continue
        tried_outputs.append(candidate)
        try:
            render_animation_3dhp(
                keypoints_px,
                poses,
                MPI3DHP_SKELETON,
                fps=args.fps,
                bitrate=args.bitrate,
                azim=args.azim,
                output=candidate,
                viewport=(viewport_w, viewport_h),
            )
            final_output = candidate
            break
        except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError, BrokenPipeError) as err:
            print(f"Rendering to {candidate} failed ({err}); trying next option...")
            continue

    if final_output is None:
        raise RuntimeError(f"Unable to render animation. Tried: {tried_outputs}")

    print(f"Saved visualization to {final_output}")


if __name__ == "__main__":
    main()
