import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import DSGNN
import utils

DSGNN_CONFIG = {"hidden_dim": 64, "heads": 4, "init_layers": 1, "recurrent_layers": 1, "dropout": 0.1}


def parse_args():
    p = argparse.ArgumentParser(description="Train one LOSO fold of DSGNN.")
    p.add_argument("--data", type=str, default="data.txt")
    p.add_argument("--t0", type=float, default=12.0)

    p.add_argument("--holdout-strain", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--huber-beta", type=float, default=0.02)
    p.add_argument("--slope-weight", type=float, default=0.15)
    p.add_argument("--amplitude-reg", type=float, default=2e-5)
    p.add_argument("--rate-reg", type=float, default=2e-5)
    p.add_argument("--rate-smoothness", type=float, default=2e-4)
    p.add_argument("--initial-rate", type=float, default=0.025)
    p.add_argument("--initial-amplitude", type=float, default=0.10)
    p.add_argument("--min-rate", type=float, default=1e-5)
    p.add_argument("--min-amplitude", type=float, default=1e-6)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--predictions", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    utils.set_seed(args.seed)
    device = utils.choose_device(args.device)

    if not args.checkpoint:
        args.checkpoint = f"best_model_holdout_{args.holdout_strain}_seed_{args.seed}.pt"
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    graph = utils.load_full_graph(args.data, args.t0)
    if args.holdout_strain not in graph.node2idx:
        raise ValueError(f"--holdout-strain={args.holdout_strain!r} not found among known strains: {graph.idx2node}")

    fit_mask_full, val_mask_full, test_mask_full = utils.build_fold_masks(
        graph, args.holdout_strain, args.val_frac, args.split_seed,
    )
    directed_adjacency_np = utils.build_directed_adjacency(graph, fit_mask_full)

    sl = slice(graph.t0_idx, graph.T)
    target_np = np.nan_to_num(graph.W[sl], nan=0.0).astype(np.float32)
    fit_mask_np = fit_mask_full[sl].copy()
    val_mask_np = val_mask_full[sl].copy()
    test_mask_np = test_mask_full[sl].copy()
    fit_mask_np[0] = False  # t0 is the given initial condition, never a target
    val_mask_np[0] = False
    test_mask_np[0] = False

    scale_mask = fit_mask_np.copy()
    scale_mask[0] = graph.Obs[graph.t0_idx]
    edge_scale = utils.robust_edge_scale(graph.W[sl], scale_mask)

    model_config_dict = {
        "num_nodes": graph.N,
        "hidden_dim": DSGNN_CONFIG["hidden_dim"],
        "num_heads": DSGNN_CONFIG["heads"],
        "init_layers": DSGNN_CONFIG["init_layers"],
        "recurrent_layers": DSGNN_CONFIG["recurrent_layers"],
        "dropout": DSGNN_CONFIG["dropout"],
        "edge_scale": edge_scale,
        "min_rate": args.min_rate,
        "min_amplitude": args.min_amplitude,
        "initial_rate": args.initial_rate,
        "initial_amplitude": args.initial_amplitude,
    }
    model = DSGNN(**model_config_dict).to(device)

    target = torch.as_tensor(target_np, device=device)
    w0 = torch.as_tensor(np.nan_to_num(graph.W[graph.t0_idx], nan=0.0).astype(np.float32), device=device)
    w0_observed = torch.as_tensor(graph.Obs[graph.t0_idx], device=device)
    adjacency = torch.as_tensor(directed_adjacency_np, device=device)
    times = torch.as_tensor(np.asarray(graph.idx2time[graph.t0_idx:], dtype=np.float32), device=device)
    fit_mask = torch.as_tensor(fit_mask_np, device=device)
    val_mask = torch.as_tensor(val_mask_np, device=device)
    test_mask = torch.as_tensor(test_mask_np, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(10, args.patience // 4), min_lr=1e-5,
    )

    print(f"Model: DSGNN  (config: {DSGNN_CONFIG})")
    held_idx = graph.node2idx[args.holdout_strain]
    n_test_edges = int(test_mask_full[graph.t0_idx + 1:].sum())
    print(f"Leave-one-strain-out: '{args.holdout_strain}' (node {held_idx}) -- "
          f"{n_test_edges} directed test interactions post-t0")
    print(f"Device: {device}  seed={args.seed}  split_seed={args.split_seed}")
    print(f"N={graph.N} T={graph.T} rollout_steps={target.shape[0] - 1} edge_scale={edge_scale:.6f}")
    print("Unordered future pairs - "
          f"fit={utils.count_unordered_pairs(fit_mask_np, 0)} "
          f"val={utils.count_unordered_pairs(val_mask_np, 0)} "
          f"test={utils.count_unordered_pairs(test_mask_np, 0)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(w0, w0_observed, adjacency, times, return_aux=True)
        loss, pieces = utils.trajectory_loss(
            output, target, fit_mask,
            huber_beta=args.huber_beta, slope_weight=args.slope_weight,
            amplitude_reg=args.amplitude_reg, rate_reg=args.rate_reg,
            rate_smoothness=args.rate_smoothness,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {float(loss)}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_output = model(w0, w0_observed, adjacency, times, return_aux=False)
            fit_metrics = utils.metrics(eval_output["pred"], target, fit_mask, w0)
            val_metrics = utils.metrics(eval_output["pred"], target, val_mask, w0)
            selection_score = val_metrics["mae"] if not np.isnan(val_metrics["mae"]) else fit_metrics["mae"]
        scheduler.step(selection_score)

        improved = selection_score < best_score - 1e-8
        if improved:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % args.log_every == 0 or improved:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch={epoch:04d} loss={float(loss.detach()):.6f} "
                  f"fit_mae={fit_metrics['mae']:.6f} "
                  f"val_mae={val_metrics['mae']:.6f}")

        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected -- validation set may be empty.")

    torch.save({
        "epoch": best_epoch,
        "model_variant": "DSGNN",
        "model_state": best_state,
        "model_config": model_config_dict,
        "train_args": vars(args),
        "idx2node": list(graph.idx2node),
        "idx2time": list(graph.idx2time),
        "t0_idx": int(graph.t0_idx),
        "best_validation_mae": best_score,
    }, checkpoint_path)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_output = model(w0, w0_observed, adjacency, times, return_aux=True)
        pred = final_output["pred"]
        fit_metrics = utils.metrics(pred, target, fit_mask, w0)
        val_metrics = utils.metrics(pred, target, val_mask, w0)
        test_metrics = utils.metrics(pred, target, test_mask, w0)
        baseline_test = utils.metrics(w0.unsqueeze(0).expand_as(target), target, test_mask, w0)

    elapsed = time.time() - start
    print("\nBest checkpoint evaluation")
    print(utils.format_metrics("fit", fit_metrics))
    print(utils.format_metrics("validation", val_metrics))
    print(utils.format_metrics("test", test_metrics))
    print(utils.format_metrics("W0 carry-forward baseline (test)", baseline_test))
    print(f"Best epoch={best_epoch}; elapsed={elapsed:.1f}s; checkpoint={checkpoint_path}")

    summary = {
        "model_variant": "DSGNN", "holdout_strain": args.holdout_strain, "seed": args.seed,
        "best_epoch": best_epoch, "best_validation_mae": best_score,
        "fit": fit_metrics, "validation": val_metrics, "test": test_metrics, "baseline_test": baseline_test,
        "model_config": model_config_dict, "n_params": sum(p.numel() for p in model.parameters()),
    }
    checkpoint_path.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.predictions:
        predictions_path = Path(args.predictions)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_path,
            pred=pred.detach().cpu().numpy(), target=target.detach().cpu().numpy(),
            fit_mask=fit_mask_np, val_mask=val_mask_np, test_mask=test_mask_np,
            w0=w0.detach().cpu().numpy(), times=times.detach().cpu().numpy(),
            idx2node=np.asarray(graph.idx2node, dtype=object),
        )
        print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()