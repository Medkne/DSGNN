import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older PyTorch
        torch.use_deterministic_algorithms(True)


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


@dataclass
class Graph:
    W: np.ndarray                  # [T,N,N] raw weights (NaN where unobserved)
    Obs: np.ndarray                 # [T,N,N] bool, observed mask
    node2idx: Dict[str, int]
    idx2node: List[str]
    N: int
    T: int
    t0_idx: int
    idx2time: List[float]


def load_full_graph(data_path, t0_value):
    from dataset import process  

    df_raw = pd.read_csv(data_path, sep="\t")
    df = process(df_raw)

    all_nodes = sorted(set(df["Sender"]).union(set(df["Receiver"])))
    node2idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    unique_times = sorted(df["Time (H)"].unique())
    T = len(unique_times)
    time2idx = {t: i for i, t in enumerate(unique_times)}
    idx2time = unique_times

    _df = df.copy()
    _df["i"] = _df["Sender"].map(node2idx)
    _df["j"] = _df["Receiver"].map(node2idx)
    _df["t"] = _df["Time (H)"].map(time2idx)
    W = np.full((T, N, N), np.nan, dtype=np.float32)
    Obs = np.zeros((T, N, N), dtype=bool)
    for (t, i, j, w) in _df[["t", "i", "j", "w"]].itertuples(index=False):
        W[t, i, j] = float(w)
        Obs[t, i, j] = True

    assert t0_value in idx2time, f"t0_value={t0_value} not found on time axis {idx2time}"
    t0_idx = idx2time.index(t0_value)
    return Graph(W, Obs, node2idx, all_nodes, N, T, t0_idx, idx2time)


def detect_hub_strains(data_path):
    df = pd.read_csv(data_path, sep="\t")
    return set(df["Test strain"].unique().tolist())


def make_pairwise_validation_split(
    train_mask,
    t0_idx,
    val_fraction,
    seed):
    fit = train_mask.copy()
    val = np.zeros_like(train_mask, dtype=bool)

    if val_fraction <= 0.0:
        return fit, val
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val-frac must be in [0, 1)")

    future_present = np.any(train_mask[t0_idx + 1:], axis=0)
    ii, jj = np.where(future_present)
    pairs = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(ii, jj)})
    if len(pairs) < 2:
        return fit, val

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(pairs))
    n_val = max(1, int(round(val_fraction * len(pairs))))
    n_val = min(n_val, len(pairs) - 1)
    held_out_pairs = {pairs[k] for k in permutation[:n_val].tolist()}

    for t in range(t0_idx + 1, train_mask.shape[0]):
        edge_i, edge_j = np.where(train_mask[t])
        for i, j in zip(edge_i.tolist(), edge_j.tolist()):
            if (min(i, j), max(i, j)) in held_out_pairs:
                fit[t, i, j] = False
                val[t, i, j] = True

    assert not np.any(fit & val)
    return fit, val


def build_fold_masks(
    graph,
    holdout_strain,
    val_fraction,
    split_seed,
):
    from dataset import make_split_edgewise  # see load_full_graph's note on local imports

    test_idx = graph.node2idx[holdout_strain]
    split = make_split_edgewise(
        graph.Obs, graph.idx2time, t0_value=graph.idx2time[graph.t0_idx],
        test_edge_frac=0.0, seed=split_seed, forced_test_nodes={test_idx},
    )
    fit, val = make_pairwise_validation_split(
        split.M_train, graph.t0_idx, val_fraction, split_seed + 17,
    )
    test = split.M_test

    assert not np.any(fit & val)
    assert not np.any(fit & test)
    assert not np.any(val & test)
    return fit, val, test


def build_directed_adjacency(graph, fit_mask_full):
    directed_adjacency = graph.Obs[graph.t0_idx].copy()
    if graph.t0_idx + 1 < graph.T:
        directed_adjacency |= np.any(fit_mask_full[graph.t0_idx + 1:], axis=0)
    np.fill_diagonal(directed_adjacency, True)
    return directed_adjacency


def count_unordered_pairs(mask, t0_idx):
    present = np.any(mask[t0_idx + 1:], axis=0)
    ii, jj = np.where(present)
    return len({(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(ii, jj)})



def robust_edge_scale(w, mask):
    values = np.abs(w[mask & np.isfinite(w)])
    if values.size == 0:
        return 1.0
    return max(float(np.quantile(values, 0.90)), 1e-3)


def masked_smooth_l1(pred, target, mask, beta):
    if not torch.any(mask):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask], beta=beta, reduction="mean")


def trajectory_loss(
    output,
    target,
    mask,
    *,
    huber_beta,
    slope_weight,
    amplitude_reg,
    rate_reg,
    rate_smoothness):
    pred = output["pred"]
    data_loss = masked_smooth_l1(pred, target, mask, huber_beta)

    if pred.shape[0] > 1:
        slope_mask = mask[1:] & mask[:-1]
        slope_loss = masked_smooth_l1(pred[1:] - pred[:-1], target[1:] - target[:-1], slope_mask, huber_beta)
    else:
        slope_loss = pred.sum() * 0.0

    amp_reg = output["amplitude_positive"].square().mean() + output["amplitude_negative"].square().mean()
    rate_pos, rate_neg = output["rate_positive"], output["rate_negative"]
    rate_reg_val = rate_pos.square().mean() + rate_neg.square().mean() if rate_pos.numel() > 0 else pred.sum() * 0.0
    if rate_pos.shape[0] > 1:
        rate_smooth_val = (rate_pos[1:] - rate_pos[:-1]).square().mean() + (rate_neg[1:] - rate_neg[:-1]).square().mean()
    else:
        rate_smooth_val = pred.sum() * 0.0

    total = (
        data_loss
        + slope_weight * slope_loss
        + amplitude_reg * amp_reg
        + rate_reg * rate_reg_val
        + rate_smoothness * rate_smooth_val
    )
    return total, {"data": float(data_loss.detach()), "slope": float(slope_loss.detach())}


def metrics(pred, target, mask, w0):
    if not torch.any(mask):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "pearson": math.nan,
                "sign_acc": math.nan, "change_sign_acc": math.nan, "n": 0.0}
    p = pred[mask].detach().float()
    y = target[mask].detach().float()
    error = p - y
    mae = error.abs().mean()
    rmse = torch.sqrt(error.square().mean())
    denom = (y - y.mean()).square().sum()
    r2 = 1.0 - error.square().sum() / denom if denom > 0 else y.new_tensor(float("nan"))
    p_centered, y_centered = p - p.mean(), y - y.mean()
    corr_denom = torch.sqrt(p_centered.square().sum() * y_centered.square().sum())
    pearson = (p_centered * y_centered).sum() / corr_denom if p.numel() > 1 and corr_denom > 0 else y.new_tensor(float("nan"))
    sign_acc = (torch.sign(p) == torch.sign(y)).float().mean()

    baseline = w0.unsqueeze(0).expand_as(pred)
    pd_, yd_ = (pred - baseline)[mask], (target - baseline)[mask]
    meaningful = yd_.abs() > 1e-6
    change_sign_acc = (
        (torch.sign(pd_[meaningful]) == torch.sign(yd_[meaningful])).float().mean()
        if meaningful.any() else y.new_tensor(float("nan"))
    )
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson": float(pearson),
            "sign_acc": float(sign_acc), "change_sign_acc": float(change_sign_acc), "n": float(p.numel())}


def format_metrics(name, values):
    return (
        f"{name}: MAE={values['mae']:.6f} RMSE={values['rmse']:.6f} "
        f"R2={values['r2']:.4f} r={values['pearson']:.4f} "
        f"sign={values['sign_acc']:.3f} change-sign={values['change_sign_acc']:.3f} "
        f"n={int(values['n'])}"
    )