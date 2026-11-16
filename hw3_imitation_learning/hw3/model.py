"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


class _MLPChunkPolicy(BasePolicy):
    """Shared MLP backbone: state (B, state_dim) -> action chunk (B, chunk_size, action_dim).

    The final linear layer outputs ``chunk_size * action_dim`` values which are reshaped
    into ``(B, chunk_size, action_dim)``; the batch dimension is inferred with ``-1``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 256,
        depth: int = 3,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, d_model), nn.ReLU()]
            in_dim = d_model
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        flat = self.mlp(state)
        return flat.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and target action chunk (both normalized)."""
        pred = self.forward(state)
        return F.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        return self.forward(state)


class ObstaclePolicy(_MLPChunkPolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """


class MultiTaskPolicy(_MLPChunkPolicy):
    """Goal-conditioned policy for the multicube scene.

    Identical network to :class:`ObstaclePolicy`; the goal information
    (cube positions, one-hot target colour, bin position) simply arrives
    as extra columns of ``state`` selected through ``--state-keys``.
    """


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 256,
    depth: int = 3,
) -> BasePolicy:
    """Build a policy.

    ``chunk_size`` / ``d_model`` / ``depth`` are required keyword arguments because
    ``hw3/eval_utils.py:load_checkpoint`` passes exactly these names when rebuilding
    a policy from a checkpoint.
    """
    policy_cls = {"obstacle": ObstaclePolicy, "multitask": MultiTaskPolicy}.get(policy_type)
    if policy_cls is None:
        raise ValueError(f"Unknown policy type: {policy_type}")

    return policy_cls(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        d_model=d_model,
        depth=depth,
    )
