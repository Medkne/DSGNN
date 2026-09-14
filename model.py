import math

import torch
from torch import nn
import torch.nn.functional as F


def _inverse_softplus(value):
    """Numerically stable inverse of softplus for positive initialization."""
    if value <= 0:
        raise ValueError("softplus initialization value must be positive")
    return math.log(math.expm1(value))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim = None,
        dropout = 0.0,
        final_activation = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        ]
        if final_activation:
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout,
        edge_scale,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_scale = float(edge_scale)
        self.attn_scale = self.head_dim ** -0.5

        self.q_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_in = nn.Linear(hidden_dim, hidden_dim, bias=False)

        edge_dim = 2
        edge_hidden = max(16, hidden_dim // 2)
        self.edge_bias = MLP(edge_dim, num_heads, edge_hidden, dropout)
        self.edge_value = MLP(edge_dim, hidden_dim, edge_hidden, dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_to_sender = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_to_receiver = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm_sender = nn.LayerNorm(hidden_dim)
        self.norm_receiver = nn.LayerNorm(hidden_dim)

    def _edge_features(self, graph):
        w = graph / self.edge_scale
        return torch.stack((F.relu(w), F.relu(-w)), dim=-1)

    def forward(
        self,
        sender,
        receiver,
        graph,
        adjacency,
    ):
        if sender.ndim != 2 or receiver.ndim != 2:
            raise ValueError("sender and receiver states must have shape [N, D]")
        if graph.ndim != 2 or adjacency.ndim != 2:
            raise ValueError("graph and adjacency must have shape [N, N]")
        if adjacency.dtype != torch.bool:
            adjacency = adjacency.bool()

        n = sender.shape[0]
        h = self.num_heads
        d = self.head_dim
        edge = self._edge_features(graph)

        # sender i attends to receiver j along i -> j
        q_out = self.q_out(sender).view(n, h, d)
        k_out = self.k_out(receiver).view(n, h, d)
        v_out = self.v_out(receiver).view(n, h, d)
        logits_out = torch.einsum("ihd,jhd->ijh", q_out, k_out) * self.attn_scale
        logits_out = logits_out + self.edge_bias(edge)
        logits_out = logits_out.masked_fill(~adjacency.unsqueeze(-1), -torch.inf)
        alpha_out = torch.softmax(logits_out, dim=1)
        alpha_out = self.dropout(alpha_out)

        edge_v_out = self.edge_value(edge).view(n, n, h, d)
        values_out = v_out.unsqueeze(0) + edge_v_out
        msg_sender = torch.einsum("ijh,ijhd->ihd", alpha_out, values_out).reshape(n, -1)
        msg_sender = self.out_proj(msg_sender)

        # receiver j attends to sender i along i -> j
        q_in = self.q_in(receiver).view(n, h, d)
        k_in = self.k_in(sender).view(n, h, d)
        v_in = self.v_in(sender).view(n, h, d)
        logits_in = torch.einsum("jhd,ihd->jih", q_in, k_in) * self.attn_scale
        edge_t = edge.transpose(0, 1)
        adjacency_t = adjacency.transpose(0, 1)
        logits_in = logits_in + self.edge_bias(edge_t)
        logits_in = logits_in.masked_fill(~adjacency_t.unsqueeze(-1), -torch.inf)
        alpha_in = torch.softmax(logits_in, dim=1)
        alpha_in = self.dropout(alpha_in)

        edge_v_in = self.edge_value(edge_t).view(n, n, h, d)
        values_in = v_in.unsqueeze(0) + edge_v_in
        msg_receiver = torch.einsum("jih,jihd->jhd", alpha_in, values_in).reshape(n, -1)
        msg_receiver = self.in_proj(msg_receiver)

        sender_out = self.norm_sender(
            sender
            + self.dropout(msg_sender)
            + self.dropout(self.cross_to_sender(receiver))
        )
        receiver_out = self.norm_receiver(
            receiver
            + self.dropout(msg_receiver)
            + self.dropout(self.cross_to_receiver(sender))
        )
        return sender_out, receiver_out


class PairDecoder(nn.Module):
    """Decode directed pair quantities from sender/receiver role embeddings."""

    def __init__(
        self,
        hidden_dim,
        time_dim,
        out_dim,
        dropout,
        include_edge_features,
    ):
        super().__init__()
        self.include_edge_features = include_edge_features
        pair_dim = 4 * hidden_dim + time_dim
        if include_edge_features:
            pair_dim += 2  # W0 value and W0 observed flag
        trunk_dim = max(hidden_dim, 96)
        self.net = nn.Sequential(
            nn.Linear(pair_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, out_dim),
        )

    def forward(
        self,
        sender,
        receiver,
        time_features,
        w0 = None,
        w0_observed = None,
    ):
        n, d = sender.shape
        s = sender[:, None, :].expand(n, n, d)
        r = receiver[None, :, :].expand(n, n, d)
        t = time_features.view(1, 1, -1).expand(n, n, -1)
        parts = [s, r, s * r, (s - r).abs(), t]

        if self.include_edge_features:
            if w0 is None or w0_observed is None:
                raise ValueError("w0 and w0_observed are required by this decoder")
            parts.extend(
                [
                    w0.unsqueeze(-1),
                    w0_observed.to(dtype=w0.dtype).unsqueeze(-1),
                ]
            )
        return self.net(torch.cat(parts, dim=-1))


class DynamicSaturationGNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        hidden_dim = 64,
        num_heads = 4,
        init_layers = 2,
        recurrent_layers = 2,
        dropout = 0.10,
        edge_scale = 0.10,
        min_rate = 1e-5,
        min_amplitude = 1e-6,
        initial_rate = 0.025,
        initial_amplitude = 0.10,
        amplitude_uses_time = True,
        use_identity_embedding = True,
    ):
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if init_layers < 0 or recurrent_layers < 0:
            raise ValueError("init_layers and recurrent_layers must both be >= 0")
        if edge_scale <= 0:
            raise ValueError("edge_scale must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_nodes = num_nodes
        self.edge_scale = edge_scale
        self.min_rate = min_rate
        self.min_amplitude = min_amplitude
        self.initial_rate = initial_rate
        self.initial_amplitude = initial_amplitude
        self._model_config = {
            "num_nodes": num_nodes,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "init_layers": init_layers,
            "recurrent_layers": recurrent_layers,
            "dropout": dropout,
            "edge_scale": edge_scale,
            "min_rate": min_rate,
            "min_amplitude": min_amplitude,
            "initial_rate": initial_rate,
            "initial_amplitude": initial_amplitude,
        }
        n = num_nodes
        d = hidden_dim
        self.amplitude_uses_time = amplitude_uses_time
        self.use_identity_embedding = use_identity_embedding

        if use_identity_embedding:
            self.sender_embedding = nn.Embedding(n, d)
            self.receiver_embedding = nn.Embedding(n, d)
        else:
            self.sender_embedding = None
            self.receiver_embedding = None

        # Directed graph summary features for each node at t0.
        # [degree, mean, std, positive mean, negative magnitude mean, max |w|]
        self.sender_stats = MLP(6, d, d, dropout)
        self.receiver_stats = MLP(6, d, d, dropout)
        self.initial_sender_norm = nn.LayerNorm(d)
        self.initial_receiver_norm = nn.LayerNorm(d)

        self.initial_blocks = nn.ModuleList(
            [
                AttentionBlock(d, num_heads, dropout, edge_scale)
                for _ in range(init_layers)
            ]
        )
        self.recurrent_blocks = nn.ModuleList(
            [
                AttentionBlock(d, num_heads, dropout, edge_scale)
                for _ in range(recurrent_layers)
            ]
        )

        self.time_dim = 16
        self.time_encoder = nn.Sequential(
            nn.Linear(6, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
        )
        self.time_to_sender = nn.Linear(self.time_dim, d)
        self.time_to_receiver = nn.Linear(self.time_dim, d)

        self.sender_gru = nn.GRUCell(d, d)
        self.receiver_gru = nn.GRUCell(d, d)
        self.post_sender_norm = nn.LayerNorm(d)
        self.post_receiver_norm = nn.LayerNorm(d)

        amp_time_dim = self.time_dim if amplitude_uses_time else 0
        self.amplitude_decoder = PairDecoder(
            d,
            amp_time_dim,
            out_dim=2,
            dropout=dropout,
            include_edge_features=True,
        )
        self.rate_positive_decoder = PairDecoder(
            d,
            self.time_dim,
            out_dim=1,
            dropout=dropout,
            include_edge_features=False,
        )
        self.rate_negative_decoder = PairDecoder(
            d,
            self.time_dim,
            out_dim=1,
            dropout=dropout,
            include_edge_features=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.use_identity_embedding:
            nn.init.normal_(self.sender_embedding.weight, std=0.02)
            nn.init.normal_(self.receiver_embedding.weight, std=0.02)

        amp_last = self.amplitude_decoder.net[-1]
        rate_pos_last = self.rate_positive_decoder.net[-1]
        rate_neg_last = self.rate_negative_decoder.net[-1]
        assert isinstance(amp_last, nn.Linear)
        assert isinstance(rate_pos_last, nn.Linear)
        assert isinstance(rate_neg_last, nn.Linear)

        nn.init.normal_(amp_last.weight, std=1e-3)
        nn.init.constant_(amp_last.bias, _inverse_softplus(self.initial_amplitude))
        for layer in (rate_pos_last, rate_neg_last):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.constant_(layer.bias, _inverse_softplus(self.initial_rate))

    @property
    def model_config(self):
        return dict(self._model_config)

    @staticmethod
    def _masked_stats(values, mask, dim):
        mask_f = mask.to(values.dtype)
        raw_count = mask_f.sum(dim=dim)
        count = raw_count.clamp_min(1.0)
        degree = raw_count / max(1, values.shape[dim])
        mean = (values * mask_f).sum(dim=dim) / count
        centered = (values - mean.unsqueeze(dim)) * mask_f
        std = torch.sqrt((centered.square().sum(dim=dim) / count).clamp_min(0.0) + 1e-8)
        pos_mean = (F.relu(values) * mask_f).sum(dim=dim) / count
        neg_mean = (F.relu(-values) * mask_f).sum(dim=dim) / count
        abs_max = (values.abs() * mask_f).amax(dim=dim)
        return torch.stack((degree, mean, std, pos_mean, neg_mean, abs_max), dim=-1)

    def _time_features(self, elapsed_fraction, dt_fraction):
        # Fourier features plus raw normalized elapsed time and step length.
        x = torch.stack(
            (
                elapsed_fraction,
                dt_fraction,
                torch.sin(math.pi * elapsed_fraction),
                torch.cos(math.pi * elapsed_fraction),
                torch.sin(2.0 * math.pi * elapsed_fraction),
                torch.cos(2.0 * math.pi * elapsed_fraction),
            )
        )
        return self.time_encoder(x)

    def encode_initial_state(
        self,
        w0,
        w0_observed,
        adjacency,
    ):
        out_stats = self._masked_stats(w0, w0_observed, dim=1)
        in_stats = self._masked_stats(w0, w0_observed, dim=0)

        if self.use_identity_embedding:
            ids = torch.arange(self.num_nodes, device=w0.device)
            sender = self.initial_sender_norm(
                self.sender_embedding(ids) + self.sender_stats(out_stats)
            )
            receiver = self.initial_receiver_norm(
                self.receiver_embedding(ids) + self.receiver_stats(in_stats)
            )
        else:
            sender = self.initial_sender_norm(self.sender_stats(out_stats))
            receiver = self.initial_receiver_norm(self.receiver_stats(in_stats))
        for block in self.initial_blocks:
            sender, receiver = block(sender, receiver, w0, adjacency)
        return sender, receiver

    def forward(
        self,
        w0,
        w0_observed,
        adjacency,
        times,
        return_aux = True,
    ):
        n = self.num_nodes
        if w0.shape != (n, n):
            raise ValueError(f"w0 must have shape {(n, n)}, got {tuple(w0.shape)}")
        if w0_observed.shape != (n, n) or adjacency.shape != (n, n):
            raise ValueError("w0_observed and adjacency must match w0")
        if times.ndim != 1 or times.numel() < 1:
            raise ValueError("times must be a non-empty 1D tensor")
        if times.numel() > 1 and not torch.all(times[1:] > times[:-1]):
            raise ValueError("times must be strictly increasing")

        w0 = torch.nan_to_num(w0, nan=0.0)
        w0_observed = w0_observed.bool()
        adjacency = adjacency.bool()

        sender, receiver = self.encode_initial_state(w0, w0_observed, adjacency)

        # Amplitudes remain fixed through the trajectory. At t0, elapsed=0.
        zero = w0.new_zeros(())
        if times.numel() > 1:
            total_span = (times[-1] - times[0]).clamp_min(1e-8)
        else:
            total_span = w0.new_tensor(1.0)
        if self.amplitude_uses_time:
            amp_time = self._time_features(zero, zero)
        else:
            amp_time = w0.new_zeros(0)
        amp_raw = self.amplitude_decoder(
            sender,
            receiver,
            amp_time,
            w0=w0 / self.edge_scale,
            w0_observed=w0_observed,
        )
        amplitude_positive = F.softplus(amp_raw[..., 0]) + self.min_amplitude
        amplitude_negative = F.softplus(amp_raw[..., 1]) + self.min_amplitude

        s_positive = torch.zeros_like(w0)
        s_negative = torch.zeros_like(w0)
        graph = w0
        predictions = [graph]
        positive_rates = []
        negative_rates = []

        for step in range(times.numel() - 1):
            dt = times[step + 1] - times[step]
            elapsed = times[step + 1] - times[0]
            elapsed_fraction = elapsed / total_span
            dt_fraction = dt / total_span
            time_features = self._time_features(elapsed_fraction, dt_fraction)

            message_sender, message_receiver = sender, receiver
            for block in self.recurrent_blocks:
                message_sender, message_receiver = block(
                    message_sender, message_receiver, graph, adjacency
                )

            message_sender = message_sender + self.time_to_sender(time_features)
            message_receiver = message_receiver + self.time_to_receiver(time_features)
            sender = self.post_sender_norm(self.sender_gru(message_sender, sender))
            receiver = self.post_receiver_norm(self.receiver_gru(message_receiver, receiver))

            rate_positive_raw = self.rate_positive_decoder(
                sender, receiver, time_features
            ).squeeze(-1)
            rate_negative_raw = self.rate_negative_decoder(
                sender, receiver, time_features
            ).squeeze(-1)
            rate_positive = F.softplus(rate_positive_raw) + self.min_rate
            rate_negative = F.softplus(rate_negative_raw) + self.min_rate

            # 1 - exp(-k*dt), evaluated stably and clipped far from overflow.
            transition_positive = -torch.expm1(
                -(rate_positive * dt).clamp(max=50.0)
            )
            transition_negative = -torch.expm1(
                -(rate_negative * dt).clamp(max=50.0)
            )
            s_positive = s_positive + (
                amplitude_positive - s_positive
            ) * transition_positive
            s_negative = s_negative + (
                amplitude_negative - s_negative
            ) * transition_negative
            graph = w0 + s_positive - s_negative

            predictions.append(graph)
            positive_rates.append(rate_positive)
            negative_rates.append(rate_negative)

        output = {"pred": torch.stack(predictions, dim=0)}
        if return_aux:
            if positive_rates:
                rate_pos_tensor = torch.stack(positive_rates, dim=0)
                rate_neg_tensor = torch.stack(negative_rates, dim=0)
            else:
                rate_pos_tensor = w0.new_empty((0, n, n))
                rate_neg_tensor = w0.new_empty((0, n, n))
            output.update(
                {
                    "amplitude_positive": amplitude_positive,
                    "amplitude_negative": amplitude_negative,
                    "rate_positive": rate_pos_tensor,
                    "rate_negative": rate_neg_tensor,
                    "saturation_positive": s_positive,
                    "saturation_negative": s_negative,
                    "sender_state": sender,
                    "receiver_state": receiver,
                }
            )
        return output


class DSGNN(DynamicSaturationGNN):
    def __init__(
        self,
        num_nodes,
        hidden_dim = 64,
        num_heads = 4,
        init_layers = 2,
        recurrent_layers = 2,
        dropout = 0.10,
        edge_scale = 0.10,
        min_rate = 1e-5,
        min_amplitude = 1e-6,
        initial_rate = 0.025,
        initial_amplitude = 0.10,
    ):
        super().__init__(
            num_nodes,
            hidden_dim,
            num_heads,
            init_layers,
            recurrent_layers,
            dropout,
            edge_scale,
            min_rate,
            min_amplitude,
            initial_rate,
            initial_amplitude,
            amplitude_uses_time=False,
            use_identity_embedding=False,
        )

