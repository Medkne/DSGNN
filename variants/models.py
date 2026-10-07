import torch
from torch import nn

from model import MLP, AttentionBlock, PairDecoder, DynamicSaturationGNN


class DS(DynamicSaturationGNN):
    """Dynamic saturation without graph message passing.

    Same node statistics, GRU recurrence and saturation decode as DSGNN, but
    with zero attention layers.
    """

    def __init__(
        self,
        num_nodes,
        hidden_dim = 32,
        num_heads = 4,
        dropout = 0.20,
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
            0,  # init_layers
            0,  # recurrent_layers
            dropout,
            edge_scale,
            min_rate,
            min_amplitude,
            initial_rate,
            initial_amplitude,
            amplitude_uses_time=True,
            use_identity_embedding=False,
        )


class DGNN(nn.Module):
    """Dynamic GNN without the saturation decode.

    Same attention + GRU recurrence as DSGNN, but each step adds a bounded
    increment directly to the running graph:

        graph_{t+1} = graph_t + max_step_change * tanh(decode(sender, receiver, time))
    """

    def __init__(
        self,
        num_nodes,
        hidden_dim = 64,
        num_heads = 4,
        init_layers = 2,
        recurrent_layers = 2,
        dropout = 0.10,
        edge_scale = 0.10,
        max_step_change = 0.10,
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
        if max_step_change <= 0:
            raise ValueError("max_step_change must be positive")

        self.num_nodes = num_nodes
        self.edge_scale = edge_scale
        self.max_step_change = max_step_change
        self._model_config = {
            "num_nodes": num_nodes,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "init_layers": init_layers,
            "recurrent_layers": recurrent_layers,
            "dropout": dropout,
            "edge_scale": edge_scale,
            "max_step_change": max_step_change,
        }
        d = hidden_dim

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

        self.increment_decoder = PairDecoder(
            d,
            self.time_dim,
            out_dim=1,
            dropout=dropout,
            include_edge_features=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        last = self.increment_decoder.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)  # start close to W0 carry-forward

    @property
    def model_config(self):
        return dict(self._model_config)

    _masked_stats = staticmethod(DynamicSaturationGNN._masked_stats)
    _time_features = DynamicSaturationGNN._time_features

    def encode_initial_state(
        self,
        w0,
        w0_observed,
        adjacency,
    ):
        out_stats = self._masked_stats(w0, w0_observed, dim=1)
        in_stats = self._masked_stats(w0, w0_observed, dim=0)
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

        if times.numel() > 1:
            total_span = (times[-1] - times[0]).clamp_min(1e-8)
        else:
            total_span = w0.new_tensor(1.0)

        graph = w0
        predictions = [graph]
        increments = []

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

            raw_increment = self.increment_decoder(sender, receiver, time_features).squeeze(-1)
            increment = self.max_step_change * torch.tanh(raw_increment)
            graph = graph + increment

            predictions.append(graph)
            increments.append(increment)

        output = {"pred": torch.stack(predictions, dim=0)}
        if return_aux:
            if increments:
                increment_tensor = torch.stack(increments, dim=0)
            else:
                increment_tensor = w0.new_empty((0, n, n))
            output.update(
                {
                    "increment": increment_tensor,
                    "sender_state": sender,
                    "receiver_state": receiver,
                }
            )
        return output