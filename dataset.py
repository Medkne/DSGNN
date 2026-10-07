import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


def process(df_raw):
    "it keeps the interaction with the lowest p-value"

    df_d = df_raw[['Test strain', 'Target strain', 'Time (H)', 'Estimate effect on target', 'FDR Adj PValue, effect on target']].copy()
    df_i = df_raw[['Target strain', 'Test strain', 'Time (H)', 'Estimate effect on test', 'FDR Adj PValue, effect on test']].copy()
    df_d.rename(columns={'Test strain': 'Sender', 'Target strain': 'Receiver',
                            'Estimate effect on target': 'w', 'FDR Adj PValue, effect on target': 'fdr'}, inplace=True)
    df_i.rename(columns={'Target strain': 'Sender', 'Test strain': 'Receiver',
                            'Estimate effect on test': 'w', 'FDR Adj PValue, effect on test': 'fdr'}, inplace=True)
    data = pd.concat([df_d, df_i], ignore_index=True, axis=0)
    # drop exact duplicates per (Sender,Receiver,Time)
    data["pair_id"] = data["Sender"] + "__" + data["Receiver"]
    keys = ["pair_id", "Time (H)"]
    idx = data.groupby(keys)["fdr"].idxmin()
    df = data.loc[idx].drop(columns=["pair_id"]).copy()
    return df


# Edge-wise persistent split with t0, PAIRED across (i,j)/(j,i)
@dataclass
class Split:
    M_train: np.ndarray  # bool [T,N,N]
    M_test: np.ndarray   # bool [T,N,N]


def make_split_edgewise(Obs,
                        idx2time,
                        t0_value,
                        test_edge_frac = 0.4,
                        seed = 42,
                        forced_test_nodes = None):
    rng = np.random.default_rng(seed)
    T, N, _ = Obs.shape
    assert t0_value in idx2time, "t0_value not on time axis"
    t0 = idx2time.index(t0_value)

    M_train = np.zeros_like(Obs, dtype=bool)
    M_test = np.zeros_like(Obs, dtype=bool)

    # t <= t0: everything observed is TRAIN (known initial condition)
    for t in range(t0 + 1):
        M_train[t] = Obs[t]

    # Edges that ever appear after t0
    future_present = np.any(Obs[t0 + 1:], axis=0)  # [N,N]
    ii, jj = np.where(future_present)
    directed_edges = list(zip(ii.tolist(), jj.tolist()))
    if len(directed_edges) == 0:
        return Split(M_train, M_test)

    # Group directed edges into undirected pairs so (i,j) and (j,i) travel together
    pair_to_edges = {}
    for (i, j) in directed_edges:
        key = frozenset((i, j))  # {i} for self-loops, {i,j} otherwise
        pair_to_edges.setdefault(key, []).append((i, j))

    pair_keys = list(pair_to_edges.keys())  

    forced_test_nodes = forced_test_nodes or set()
    forced_positions = {
        pos for pos, key in enumerate(pair_keys)
        if any(node in forced_test_nodes for node in key)
    }
    remaining_positions = [pos for pos in range(len(pair_keys)) if pos not in forced_positions]

    perm = rng.permutation(len(remaining_positions))
    k_test_pairs = int(math.ceil(test_edge_frac * len(remaining_positions)))
    random_test_positions = {remaining_positions[k] for k in perm[:k_test_pairs].tolist()}
    test_pair_positions = forced_positions | random_test_positions

    test_edges = set()
    for pos, key in enumerate(pair_keys):
        if pos in test_pair_positions:
            for e in pair_to_edges[key]:
                test_edges.add(e)

    for t in range(t0 + 1, T):
        obs_t = Obs[t]
        train_t = np.zeros_like(obs_t, dtype=bool)
        test_t = np.zeros_like(obs_t, dtype=bool)
        ei, ej = np.where(obs_t)
        for a, b in zip(ei, ej):
            if (a, b) in test_edges:
                test_t[a, b] = True
            else:
                train_t[a, b] = True
        M_train[t] = train_t
        M_test[t] = test_t

    # No overlap
    assert not np.any(M_train & M_test), "Train/Test masks overlap!"

    # Pairing actually holds (whenever both directions are observed post-t0)
    both_dirs_present = future_present & future_present.T
    pi, pj = np.where(both_dirs_present)
    for a, b in zip(pi.tolist(), pj.tolist()):
        in_test_ab = (a, b) in test_edges
        in_test_ba = (b, a) in test_edges
        assert in_test_ab == in_test_ba, f"Pairing violated for ({a},{b})/({b},{a})"

    return Split(M_train, M_test)


# Per-time standardization (using TRAIN edges only)
@dataclass
class StdInfo:
    mean_t: np.ndarray
    std_t: np.ndarray


def standardize_per_time(W, M_train):
    T, N, _ = W.shape
    Wz = np.copy(W)
    mean_t = np.zeros(T, dtype=np.float32)
    std_t = np.ones(T, dtype=np.float32)
    for t in range(T):
        vals = W[t][M_train[t]]
        if vals.size == 0:
            mean_t[t] = 0.0
            std_t[t] = 1.0
        else:
            m = float(vals.mean())
            s = float(vals.std())
            s = 1.0 if s < 1e-6 else s
            mean_t[t] = m
            std_t[t] = s
            obs = ~np.isnan(W[t])
            Wz[t][obs] = (W[t][obs] - m) / s
    return Wz, StdInfo(mean_t, std_t)


def build_adjacency(M_train):
    ever = M_train.any(axis=0)  # [N,N]
    A = ever | ever.T
    np.fill_diagonal(A, True)
    return A


@dataclass
class Dataset:
    W: np.ndarray                 # [T,N,N] raw weights (NaN where unobserved)
    Obs: np.ndarray                # [T,N,N] bool, observed mask
    Wz: np.ndarray                 # [T,N,N] per-time standardized weights
    std_info: StdInfo
    split: Split
    adjacency: np.ndarray          # [N,N] bool, undirected, TRAIN-only, self-loops included
    idx2node: List[str]
    idx2time: List[float]
    node2idx: Dict[str, int]
    time2idx: Dict[float, int]
    N: int
    T: int
    t0_idx: int
    t0_value: float


def load_dataset(data_path,
                 t0_value,
                 test_edge_frac = 0.3,
                 split_seed = 50,
                 holdout_strain = None):
    df_raw = pd.read_csv(data_path, sep="\t")
    df = process(df_raw)

    all_nodes = sorted(set(df['Sender']).union(set(df['Receiver'])))
    node2idx = {n: i for i, n in enumerate(all_nodes)}
    idx2node = all_nodes
    N = len(all_nodes)

    unique_times = sorted(df['Time (H)'].unique())
    T = len(unique_times)
    time2idx = {t: i for i, t in enumerate(unique_times)}
    idx2time = unique_times  # e.g., [12, 24, ..., 168]

    _df = df.copy()
    _df['i'] = _df['Sender'].map(node2idx)
    _df['j'] = _df['Receiver'].map(node2idx)
    _df['t'] = _df['Time (H)'].map(time2idx)

    W = np.full((T, N, N), np.nan, dtype=np.float32)
    Obs = np.zeros((T, N, N), dtype=bool)
    for (t, i, j, w) in _df[['t', 'i', 'j', 'w']].itertuples(index=False):
        W[t, i, j] = float(w)
        Obs[t, i, j] = True

    assert t0_value in idx2time, f"t0_value={t0_value} not found on time axis {idx2time}"
    t0_idx = idx2time.index(t0_value)

    forced_test_nodes = None
    if holdout_strain is not None:
        if holdout_strain not in node2idx:
            raise ValueError(
                f"holdout_strain={holdout_strain!r} not found among known strains: "
                f"{idx2node}"
            )
        forced_test_nodes = {node2idx[holdout_strain]}

    split = make_split_edgewise(Obs, idx2time, t0_value=t0_value,
                                test_edge_frac=test_edge_frac, seed=split_seed,
                                forced_test_nodes=forced_test_nodes)
    assert not split.M_test[t0_idx].any(), "No test edges allowed at t0"

    Wz, std_info = standardize_per_time(W, split.M_train)
    adjacency = build_adjacency(split.M_train)

    return Dataset(
        W=W, Obs=Obs, Wz=Wz, std_info=std_info, split=split, adjacency=adjacency,
        idx2node=idx2node, idx2time=idx2time, node2idx=node2idx, time2idx=time2idx,
        N=N, T=T, t0_idx=t0_idx, t0_value=t0_value,
    )