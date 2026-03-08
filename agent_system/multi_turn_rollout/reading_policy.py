"""
Reading Policy (RP) for LatentMAS-RP.

This module implements the learnable agent-level gating mechanism described in
the LatentMAS-RP design.  A lightweight GateNet (3-layer MLP) produces a scalar
gate g_i ∈ [0, 1] for each upstream agent, controlling how much of that agent's
latent representation is forwarded to the Judger.

Key design choices (matching the paper):
  - Agent-level granularity: each agent's full KV segment is one block.
  - Gumbel-Sigmoid sampling during training for differentiable soft gates.
  - Hard threshold at evaluation for deterministic selection.
  - Decoupled training: GateNet is updated with REINFORCE + EMA baseline;
    the backbone LLM is kept frozen.
  - Sparsity-regularized reward: R = R_task - λ * Σ|g_i|.
  - Lazy initialization: GateNet is built on its first forward call so that
    the actual input dimension (which may differ between the HF path, where
    block summaries come from KV-cache keys, and the vLLM path, where they
    come from embedding records) is used automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GateNet
# ---------------------------------------------------------------------------

class GateNet(nn.Module):
    """
    Lazily-initialized 3-layer MLP that maps [query ; block_summary] → scalar logit.

    Architecture (built on first forward call):
        Linear(dim_in, dim_in // 2) → ReLU
        Linear(dim_in // 2, dim_in // 4) → ReLU
        Linear(dim_in // 4, 1)

    The final layer's bias is initialised to +2.0 so that sigmoid(logit) ≈ 0.88
    at initialisation, i.e. gates start near 1 (pass-through mode).
    """

    def __init__(self) -> None:
        super().__init__()
        self._built: bool = False
        self.layers: Optional[nn.Sequential] = None

    # ------------------------------------------------------------------
    def _build(self, dim_in: int, device: torch.device, dtype: torch.dtype) -> None:
        hidden1 = max(dim_in // 2, 1)
        hidden2 = max(hidden1 // 2, 1)
        self.layers = nn.Sequential(
            nn.Linear(dim_in, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        ).to(device=device, dtype=dtype)
        # Bias init: gate starts near 1.0
        nn.init.constant_(self.layers[-1].bias, 2.0)
        self._built = True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, dim_q + dim_s]`` — concatenation of query and block summary.
        Returns:
            logit: ``[B, 1]``
        """
        if not self._built:
            self._build(x.shape[-1], x.device, x.dtype)
        elif self.layers[0].weight.device != x.device or self.layers[0].weight.dtype != x.dtype:
            self.layers = self.layers.to(device=x.device, dtype=x.dtype)
        return self.layers(x)


# ---------------------------------------------------------------------------
# ReadingPolicy
# ---------------------------------------------------------------------------

class ReadingPolicy(nn.Module):
    """
    Learnable reading policy for LatentMAS-RP.

    One GateNet per upstream agent.  The policy is trained via REINFORCE with
    an EMA baseline and a sparsity penalty on the gate values.

    Parameters
    ----------
    n_agents:
        Number of upstream agents (Planner, Critic, Refiner → 3 by default).
    temperature:
        Gumbel-Sigmoid temperature τ (lower = sharper gates).
    hard_threshold:
        Threshold δ used for hard gating at evaluation time.
    sparsity_lambda:
        Weight λ for the L1 sparsity penalty on gate values.
    ema_momentum:
        EMA momentum α for the REINFORCE baseline update.
    lr:
        Adam learning rate for the GateNet parameters.
    """

    def __init__(
        self,
        n_agents: int = 3,
        temperature: float = 1.0,
        hard_threshold: float = 0.5,
        sparsity_lambda: float = 0.01,
        ema_momentum: float = 0.99,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.temperature = temperature
        self.hard_threshold = hard_threshold
        self.sparsity_lambda = sparsity_lambda
        self.ema_momentum = ema_momentum

        # One independent GateNet per upstream agent
        self.gate_nets = nn.ModuleList([GateNet() for _ in range(n_agents)])

        # EMA baseline (scalar, non-trainable)
        self.register_buffer("baseline", torch.tensor(0.0))

        self._lr = lr
        self._optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Lazy optimizer (built after GateNets are lazily initialised)
    # ------------------------------------------------------------------

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return self._optimizer

    # ------------------------------------------------------------------
    # Block-summary extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_block_summary_from_kv(
        past_kv: Tuple,
        start: int,
        end: int,
        mid_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute a block summary by mean-pooling the key vectors of the middle
        Transformer layer over the agent's KV segment.

        Matches Eq. (3) in the design:
            s_i = mean_{t=start..end} K̂^{(i)}_{:,t,:}
        where K̂ is the key tensor reshaped to [B, L_i, H_kv * d_h].

        Args:
            past_kv: Tuple of ``(key, value)`` tensors per Transformer layer.
                     Each key has shape ``[B, H_kv, L_total, d_h]``.
            start: First KV position belonging to this agent's segment.
            end: One past the last position (exclusive).
            mid_layer: Which layer to use; defaults to ``len(past_kv) // 2``.

        Returns:
            summary: ``[B, H_kv * d_h]`` in float32.
        """
        n_layers = len(past_kv)
        if mid_layer is None:
            mid_layer = n_layers // 2

        keys = past_kv[mid_layer][0]          # [B, H_kv, L_total, d_h]
        seg = keys[:, :, start:end, :]        # [B, H_kv, L_i,     d_h]
        B, H_kv, L_i, d_h = seg.shape
        # Reshape to [B, L_i, D] where D = H_kv * d_h
        seg = seg.permute(0, 2, 1, 3).reshape(B, L_i, H_kv * d_h)
        return seg.mean(dim=1).float()         # [B, H_kv * d_h]

    @staticmethod
    def extract_block_summary_from_embeddings(
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a block summary by mean-pooling an agent's embedding sequence.

        Used on the vLLM path where ``embedding_record[i]`` contains the
        concatenated input embeddings produced by the agent (including latent
        step vectors).

        Args:
            embeddings: ``[B, L_i, H]``

        Returns:
            summary: ``[B, H]`` in float32.
        """
        return embeddings.float().mean(dim=1)

    @staticmethod
    def extract_query_repr(
        judger_ids: torch.Tensor,
        embedding_layer: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Judger's query representation by mean-pooling its input
        token embeddings (Eq. 4 in the design).

        Args:
            judger_ids: ``[B, L_judger]`` token IDs.
            embedding_layer: The model's input embedding layer.
            attention_mask: Optional ``[B, L_judger]`` mask (1 = real token).
                            When provided, padding tokens are excluded from
                            the mean.

        Returns:
            query: ``[B, H]`` in float32.
        """
        with torch.no_grad():
            emb = embedding_layer(judger_ids).float()   # [B, L, H]

        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1).to(emb.device)  # [B, L, 1]
            query = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            query = emb.mean(dim=1)

        return query  # [B, H]

    # ------------------------------------------------------------------
    # Gate computation — Eq. (5) training / Eq. (6) evaluation
    # ------------------------------------------------------------------

    def compute_gates(
        self,
        query: torch.Tensor,                     # [B, D_q]
        block_summaries: List[torch.Tensor],     # N tensors, each [B, D_s]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gate values for all upstream agents.

        During training each gate is drawn from Gumbel-Sigmoid (Eq. 5);
        at evaluation a hard threshold is applied (Eq. 6).

        Args:
            query: Judger query representation ``[B, D_q]``.
            block_summaries: Per-agent block summaries (each ``[B, D_s]``).
                             ``D_q`` and ``D_s`` may differ; the GateNet handles
                             the concatenation ``[B, D_q + D_s]`` lazily.
            training: If True use stochastic Gumbel-Sigmoid; else use hard gate.

        Returns:
            gates:  ``[B, N]`` — gate values in [0, 1] (float32).
            logits: ``[B, N]`` — raw logits with grad_fn (needed for REINFORCE).
        """
        gates_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []

        q = query.float()

        for i, (gnet, s_i) in enumerate(zip(self.gate_nets, block_summaries)):
            s = s_i.float().to(q.device)
            x = torch.cat([q, s], dim=-1)          # [B, D_q + D_s]
            logit = gnet(x).squeeze(-1)             # [B]
            logits_list.append(logit)

            if training:
                # Gumbel-Sigmoid reparameterisation (binary concrete)
                u = torch.empty_like(logit).uniform_().clamp(1e-6, 1.0 - 1e-6)
                gumbel_noise = -torch.log(-torch.log(u))
                g = torch.sigmoid((logit + gumbel_noise) / self.temperature)  # [B]
            else:
                # Hard gate at evaluation
                g = (torch.sigmoid(logit) > self.hard_threshold).float()      # [B]

            gates_list.append(g)

        gates = torch.stack(gates_list, dim=-1)    # [B, N]
        logits = torch.stack(logits_list, dim=-1)  # [B, N]
        return gates, logits

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_gates_hf(
        past_kv: Tuple,
        agent_kv_ranges: List[Tuple[int, int]],
        gates: torch.Tensor,   # [B, N]
    ) -> Tuple:
        """
        Scale each agent's KV-cache segment by the corresponding gate value
        in-place (but returning new tensors to preserve the original cache).

        Implements Eq. (7): K̃V^{(i)}_l = g_i · KV^{(i)}_l for all layers l.

        Args:
            past_kv: Accumulated HF KV cache (tuple of per-layer (key, value) pairs).
            agent_kv_ranges: ``[(start_i, end_i), ...]`` for each agent.
            gates: ``[B, N]`` gate values.

        Returns:
            New past_kv with gated segments.
        """
        new_layers = []
        for layer_kv in past_kv:
            k = layer_kv[0].clone()   # [B, H_kv, L_total, d_h]
            v = layer_kv[1].clone()
            for i, (start, end) in enumerate(agent_kv_ranges):
                g = gates[:, i].view(-1, 1, 1, 1).to(k.dtype).to(k.device)
                k[:, :, start:end, :] *= g
                v[:, :, start:end, :] *= g
            new_layers.append((k, v))
        return tuple(new_layers)

    @staticmethod
    def apply_gates_vllm(
        embedding_record: List[torch.Tensor],
        gates: torch.Tensor,   # [B, N]
    ) -> List[torch.Tensor]:
        """
        Scale each agent's embedding block by the corresponding gate value.

        Used on the vLLM path before concatenating the embeddings and feeding
        them to the vLLM engine.

        Args:
            embedding_record: Per-agent embedding tensors (each ``[B, L_i, H]``).
            gates: ``[B, N]`` gate values.

        Returns:
            List of gated tensors with the same shapes.
        """
        return [
            emb * gates[:, i].detach().view(-1, 1, 1).to(device=emb.device, dtype=emb.dtype)
            for i, emb in enumerate(embedding_record)
        ]

    # ------------------------------------------------------------------
    # Log-probabilities — Eq. (10)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_log_probs(
        gates: torch.Tensor,    # [B, N]  — treated as Bernoulli samples
        logits: torch.Tensor,   # [B, N]  — must have grad_fn for REINFORCE
    ) -> torch.Tensor:
        """
        Bernoulli log-probability of the sampled gates under the current policy:

            log π(g_i) = g_i · log σ(l_i) + (1 − g_i) · log(1 − σ(l_i))

        Sum over agents to get one scalar per batch element.

        Returns:
            log_probs: ``[B]`` with grad_fn w.r.t. GateNet parameters.
        """
        lls = (
            gates.detach() * F.logsigmoid(logits)
            + (1.0 - gates.detach()) * F.logsigmoid(-logits)
        )
        return lls.sum(dim=-1)   # [B]

    # ------------------------------------------------------------------
    # Augmented reward — Eq. (8)
    # ------------------------------------------------------------------

    def compute_augmented_rewards(
        self,
        task_rewards: torch.Tensor,   # [B]
        gates: torch.Tensor,          # [B, N]
    ) -> torch.Tensor:
        """
        R = R_task − λ · Σ_i |g_i|
        """
        sparsity_penalty = gates.detach().abs().sum(dim=-1)   # [B]
        return task_rewards - self.sparsity_lambda * sparsity_penalty

    # ------------------------------------------------------------------
    # REINFORCE update — Eq. (9) + Eq. (11)
    # ------------------------------------------------------------------

    def reinforce_step(
        self,
        log_probs: torch.Tensor,          # [B] — with grad_fn
        augmented_rewards: torch.Tensor,  # [B] — detached
    ) -> torch.Tensor:
        """
        Compute the REINFORCE policy-gradient loss and update the EMA baseline.

            L_φ = −(1/B) Σ_j log_probs_j · (R_j − b)
            b  ← α · b + (1 − α) · mean(R)

        Returns:
            Scalar loss tensor (with grad_fn).
        """
        mean_r = augmented_rewards.detach().mean()
        # EMA baseline update (Eq. 11)
        self.baseline = (
            self.ema_momentum * self.baseline
            + (1.0 - self.ema_momentum) * mean_r
        )
        advantages = (augmented_rewards.detach() - self.baseline)
        loss = -(log_probs * advantages).mean()
        return loss

    def update(
        self,
        log_probs: torch.Tensor,      # [B] — with grad_fn
        task_rewards: torch.Tensor,   # [B]
        gates: torch.Tensor,          # [B, N]
    ) -> Dict:
        """
        Full REINFORCE update step.

        1. Compute augmented rewards (task reward − sparsity penalty).
        2. Compute REINFORCE loss.
        3. Backpropagate and step the optimizer.

        Args:
            log_probs: Log-probabilities of the gate samples (must have grad_fn
                       w.r.t. GateNet parameters).
            task_rewards: Task-level correctness signal, shape ``[B]``.
            gates: Gate values used in the forward pass, shape ``[B, N]``.

        Returns:
            Dictionary with training statistics.
        """
        aug_rewards = self.compute_augmented_rewards(task_rewards, gates)
        loss = self.reinforce_step(log_probs, aug_rewards)

        opt = self.get_optimizer()
        opt.zero_grad()
        loss.backward()
        opt.step()

        return {
            "rp_loss": loss.item(),
            "mean_task_reward": task_rewards.detach().mean().item(),
            "mean_augmented_reward": aug_rewards.detach().mean().item(),
            "mean_gate_value": gates.detach().mean().item(),
            "gate_sparsity": (gates.detach() < 0.5).float().mean().item(),
            "baseline": self.baseline.item(),
        }
