"""Graph neural belief model scaffold for node-wise compromise beliefs.

This is not wired into the Gym environments yet. The intended first use is
supervised belief learning from simulated rollout traces where the true next
state is known and can be used as a BCE target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GraphBeliefBatch:
    """Inputs for one graph-belief update batch.

    Shapes:
    - node_features: (batch, nodes, node_feature_dim)
    - adjacency: (nodes, nodes) or (batch, nodes, nodes)
    - hidden: (batch, nodes, hidden_dim)
    """

    node_features: torch.Tensor
    adjacency: torch.Tensor
    hidden: torch.Tensor


class GraphBeliefGRU(nn.Module):
    """Message-passing GRU for learned factored belief updates.

    The update follows the form

        h'_v = GRU(h_v, [o_v, sum_{u in N(v)} phi(h_u, h_v, e_uv)])

    with binary adjacency as the current edge feature. The output is a node-wise
    compromise probability sigmoid(readout(h'_v)).
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        message_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(node_feature_dim + message_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, belief: torch.Tensor) -> torch.Tensor:
        """Create a simple hidden state from current marginal belief.

        belief shape: (batch, nodes). The first hidden channel stores belief;
        remaining channels start at zero.
        """
        batch_size, num_nodes = belief.shape
        hidden = belief.new_zeros(batch_size, num_nodes, self.hidden_dim)
        hidden[..., 0] = belief
        return hidden

    def forward(self, batch: GraphBeliefBatch) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = batch.node_features
        adjacency = batch.adjacency
        hidden = batch.hidden
        batch_size, num_nodes, _ = node_features.shape

        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        adjacency = adjacency.to(dtype=node_features.dtype, device=node_features.device)

        h_u = hidden.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        h_v = hidden.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        edge_feature = adjacency.unsqueeze(-1)
        message_input = torch.cat([h_u, h_v, edge_feature], dim=-1)
        messages = self.message_mlp(message_input)
        messages = messages * edge_feature
        aggregate = messages.sum(dim=2)

        gru_input = torch.cat([node_features, aggregate], dim=-1)
        next_hidden = self.gru(
            gru_input.reshape(batch_size * num_nodes, -1),
            hidden.reshape(batch_size * num_nodes, -1),
        ).reshape(batch_size, num_nodes, self.hidden_dim)
        belief_logits = self.readout(next_hidden).squeeze(-1)
        belief = torch.sigmoid(belief_logits)
        return belief, next_hidden


class LearnedGNNBeliefUpdater:
    """Stateless wrapper that uses a saved GraphBeliefGRU as a belief updater."""

    def __init__(
        self,
        model_path: str,
        adjacency,
        device: str = "cpu",
    ) -> None:
        import numpy as np

        self.device = torch.device(device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        self.node_feature_dim = int(checkpoint["node_feature_dim"])
        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.message_dim = int(checkpoint["message_dim"])
        self.model = GraphBeliefGRU(
            node_feature_dim=self.node_feature_dim,
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.adjacency = torch.as_tensor(
            np.asarray(adjacency, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        degrees = self.adjacency.sum(dim=1)
        max_degree = torch.clamp(degrees.max(), min=1.0)
        self.degree_feature = (degrees / max_degree).to(self.device)

    def update(self, belief, observation, defense):
        import numpy as np

        belief_np = np.asarray(belief, dtype=np.float32)
        observation_np = np.asarray(observation, dtype=np.float32)
        defense_np = np.asarray(defense, dtype=np.float32)
        if belief_np.ndim != 1:
            raise ValueError("learned GNN belief must be a 1D factored vector.")
        with torch.no_grad():
            belief_t = torch.as_tensor(belief_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            defense_t = torch.as_tensor(defense_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            observation_t = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            degree_t = self.degree_feature.unsqueeze(0)
            node_features = torch.stack([belief_t, defense_t, observation_t, degree_t], dim=-1)
            hidden = self.model.initial_hidden(belief_t)
            next_belief, _ = self.model(GraphBeliefBatch(node_features, self.adjacency, hidden))
        return next_belief.squeeze(0).cpu().numpy().astype(np.float64)
