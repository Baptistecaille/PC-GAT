"""
PC-GAT: Predictive Coding meets Graph Attention Networks
=========================================================

A biologically-plausible graph learning architecture that combines:
  - Predictive Coding (PC): hierarchical belief inference via local gradient descent
  - Graph Attention Networks (GAT): attention over graph neighborhoods

Key properties:
  - Attention guided by surprise (prediction errors), not raw features
  - Iterative inference: beliefs converge before weights update
  - No global backpropagation: all updates are local to each layer
  - Native anomaly detection: atypical nodes generate high prediction errors

Formal definitions (l = layer index, i = node index, N(i) = neighbors of i):

  Top-down prediction:
    μ̂ᵢˡ = fˡ(Σⱼ∈N(i) αᵢⱼˡ · Wˡ · μⱼˡ⁺¹)

  Prediction error:
    εᵢˡ = μᵢˡ - μ̂ᵢˡ

  Surprise-guided attention:
    eᵢⱼˡ = LeakyReLU(aˡᵀ · [εᵢˡ ‖ εⱼˡ])
    αᵢⱼˡ = softmax_j(eᵢⱼˡ)

  Inference dynamics (gradient descent on free energy F = Σₗ Σᵢ |εᵢˡ|²):
    μ̇ᵢˡ = -εᵢˡ + Σⱼ∈N(i) αᵢⱼˡ⁻¹ · (∂μ̂ⱼˡ⁻¹/∂μᵢˡ)ᵀ · εⱼˡ⁻¹

  Local weight update (Hebbian/PC rule):
    ΔWˡ ∝ Σᵢ εᵢˡ · (Σⱼ∈N(i) αᵢⱼˡ · μⱼˡ⁺¹)ᵀ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List


class PCGATLayer(nn.Module):
    """
    Single PC-GAT layer.

    Each node i maintains:
      - Representation neurons μᵢˡ  : current belief about the node state
      - Error neurons       εᵢˡ  : gap between belief and top-down prediction

    The layer runs an inference loop that iteratively:
      1. Computes top-down predictions using attention-weighted neighbor aggregation
      2. Computes prediction errors
      3. Updates attention scores based on those errors (surprise-guided)
      4. Updates beliefs via local gradient descent (no global backprop)

    After convergence, weights can be updated locally via `local_weight_update`.

    Args:
        in_features:        Dimensionality of the upper (input) layer representations.
        out_features:       Dimensionality of this layer's representations.
        n_inference_steps:  Number of iterative inference steps before convergence.
        inference_lr:       Step size for belief update dynamics.
        dropout:            Dropout rate applied to attention weights.
        activation:         Non-linearity used in the generative model ('relu', 'tanh', 'linear').
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_inference_steps: int = 20,
        inference_lr: float = 0.1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_inference_steps = n_inference_steps
        self.inference_lr = inference_lr

        # Wˡ : generative weight matrix for top-down predictions
        # Shape: [out_features, in_features]
        self.W = nn.Parameter(torch.empty(out_features, in_features))

        # aˡ : attention weight vector operating on concatenated error pairs
        # Shape: [2 * out_features]
        self.a = nn.Parameter(torch.empty(2 * out_features))

        self.dropout = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "linear":
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unknown activation '{activation}'. Choose relu/tanh/linear.")

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W.unsqueeze(0)).squeeze_(0)
        nn.init.xavier_uniform_(self.a.view(1, -1)).squeeze_(0)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def compute_attention(
        self,
        errors: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Compute surprise-guided attention weights.

        eᵢⱼˡ = LeakyReLU(aˡᵀ · [εᵢˡ ‖ εⱼˡ])
        αᵢⱼˡ = softmax over neighbors j of node i

        Nodes that share similar prediction errors attract each other.
        A neighbor that surprises the current node heavily gets more weight.

        Args:
            errors:     Prediction errors  [N, out_features]
            edge_index: Graph edges        [2, E]  (edge_index[0]=src, edge_index[1]=dst)

        Returns:
            alpha: Attention weights [E]
        """
        src, dst = edge_index  # message flows src → dst

        # Concatenate error vectors of (source, destination) for each edge
        e_concat = torch.cat([errors[src], errors[dst]], dim=-1)  # [E, 2*out_features]

        # Raw attention score
        e_scores = self.leaky_relu(e_concat @ self.a)  # [E]
        e_scores = self.dropout(e_scores)

        n_nodes = errors.shape[0]
        alpha = self._sparse_softmax(e_scores, dst, n_nodes)
        return alpha

    def _sparse_softmax(self, scores: Tensor, dst: Tensor, n_nodes: int) -> Tensor:
        """Numerically stable softmax grouped by destination node."""
        # Subtract per-node maximum for numerical stability
        max_scores = torch.full((n_nodes,), float("-inf"), device=scores.device)
        max_scores.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        # Replace -inf (isolated nodes) with 0
        max_scores = max_scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        shifted = scores - max_scores[dst]
        exp_scores = torch.exp(shifted)

        sum_exp = torch.zeros(n_nodes, device=scores.device)
        sum_exp.scatter_add_(0, dst, exp_scores)

        alpha = exp_scores / (sum_exp[dst] + 1e-8)
        return alpha

    def top_down_prediction(
        self,
        mu_upper: Tensor,
        alpha: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Generate top-down predictions via attention-weighted neighbor aggregation.

        μ̂ᵢˡ = fˡ(Σⱼ∈N(i) αᵢⱼˡ · Wˡ · μⱼˡ⁺¹)

        Args:
            mu_upper:   Upper layer beliefs  [N, in_features]
            alpha:      Attention weights    [E]
            edge_index: Graph edges          [2, E]

        Returns:
            mu_hat: Top-down predictions [N, out_features]
        """
        src, dst = edge_index
        n_nodes = mu_upper.shape[0]

        # Linear transform of upper-layer representations: Wˡ · μⱼˡ⁺¹
        transformed = mu_upper @ self.W.t()  # [N, out_features]

        # Weight each message by its attention score and aggregate at destination
        messages = alpha.unsqueeze(-1) * transformed[src]  # [E, out_features]
        agg = torch.zeros(n_nodes, self.out_features, device=mu_upper.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Apply generative non-linearity fˡ
        mu_hat = self.activation(agg)
        return mu_hat

    def inference_step(
        self,
        mu: Tensor,
        errors: Tensor,
        errors_lower: Optional[Tensor],
        alpha_lower: Optional[Tensor],
        edge_index: Tensor,
    ) -> Tensor:
        """
        One gradient-descent step on the variational free energy.

        μ̇ᵢˡ = -εᵢˡ  +  Σⱼ∈N(i) αᵢⱼˡ⁻¹ · (∂μ̂ⱼˡ⁻¹/∂μᵢˡ)ᵀ · εⱼˡ⁻¹

        First term:  local correction toward the top-down prediction.
        Second term: bottom-up error propagation from lower-layer neighbors,
                     weighted by the attention of that lower layer.

        Args:
            mu:           Current beliefs at this layer      [N, out_features]
            errors:       Current prediction errors          [N, out_features]
            errors_lower: Prediction errors from layer l-1   [N, lower_features] or None
            alpha_lower:  Attention from layer l-1            [E] or None
            edge_index:   Graph edges                         [2, E]

        Returns:
            mu_new: Updated beliefs [N, out_features]
        """
        # Local correction: push belief toward prediction
        delta = -errors  # [N, out_features]

        # Bottom-up error backpropagation from lower layer
        if errors_lower is not None and alpha_lower is not None:
            src, dst = edge_index

            # Weighted lower-layer errors flowing into each edge (dst receives from src)
            # (∂μ̂ⱼˡ⁻¹/∂μᵢˡ)ᵀ · εⱼˡ⁻¹ approximated by Wˡᵀ · (αᵢⱼˡ⁻¹ · εⱼˡ⁻¹)
            weighted_err = alpha_lower.unsqueeze(-1) * errors_lower[dst]  # [E, lower_features]

            # Accumulate at source nodes (upper layer)
            lower_features = errors_lower.shape[-1]
            backprop = torch.zeros(mu.shape[0], lower_features, device=mu.device)
            backprop.scatter_add_(0, src.unsqueeze(-1).expand_as(weighted_err), weighted_err)

            # Project through Wˡᵀ to match current layer dimensionality
            delta = delta + backprop @ self.W  # [N, out_features]

        mu_new = mu + self.inference_lr * delta
        return mu_new

    def local_weight_update(
        self,
        errors: Tensor,
        mu_upper: Tensor,
        alpha: Tensor,
        edge_index: Tensor,
    ) -> None:
        """
        Accumulate the local Hebbian/PC weight gradient.

        ΔWˡ ∝ Σᵢ εᵢˡ · (Σⱼ∈N(i) αᵢⱼˡ · μⱼˡ⁺¹)ᵀ

        This rule requires only quantities local to layer l.
        No global backpropagation through the computation graph is needed.

        The gradient is stored in self.W.grad for use with a standard optimizer.

        Args:
            errors:     Converged prediction errors              [N, out_features]
            mu_upper:   Upper layer beliefs                       [N, in_features]
            alpha:      Converged attention weights               [E]
            edge_index: Graph edges                               [2, E]
        """
        src, dst = edge_index
        n_nodes = errors.shape[0]

        # Attention-weighted aggregation of upper-layer representations
        weighted_upper = alpha.unsqueeze(-1) * mu_upper[src]  # [E, in_features]
        agg_upper = torch.zeros(n_nodes, self.in_features, device=mu_upper.device)
        agg_upper.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_upper), weighted_upper)

        # ΔW = εᵀ · agg_upper  shape: [out_features, in_features]
        delta_W = errors.t() @ agg_upper

        # Store as negative gradient (optimizer will subtract the gradient)
        if self.W.grad is None:
            self.W.grad = -delta_W.detach()
        else:
            self.W.grad.add_(-delta_W.detach())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        mu_upper: Tensor,
        edge_index: Tensor,
        mu_init: Optional[Tensor] = None,
        errors_lower: Optional[Tensor] = None,
        alpha_lower: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run iterative inference to convergence for this layer.

        The loop alternates between:
          1. Top-down prediction  →  prediction error
          2. Surprise-guided attention update
          3. Belief update (local gradient descent)

        Args:
            mu_upper:     Upper layer beliefs                      [N, in_features]
            edge_index:   Graph edges                              [2, E]
            mu_init:      Initial beliefs at this layer            [N, out_features] or None
            errors_lower: Prediction errors from layer below       [N, lower_dim] or None
            alpha_lower:  Attention weights from layer below       [E] or None

        Returns:
            mu:     Converged beliefs       [N, out_features]
            errors: Final prediction errors [N, out_features]
            alpha:  Final attention weights [E]
        """
        n_nodes = mu_upper.shape[0]

        # Initialise beliefs
        mu = (
            mu_init.clone()
            if mu_init is not None
            else torch.zeros(n_nodes, self.out_features, device=mu_upper.device)
        )

        # Uniform attention as starting point
        alpha = self._uniform_attention(edge_index, n_nodes, mu_upper.device)

        # ── Inference loop ──────────────────────────────────────────────
        for _ in range(self.n_inference_steps):
            # 1. Top-down prediction
            mu_hat = self.top_down_prediction(mu_upper, alpha, edge_index)

            # 2. Prediction error:  εᵢˡ = μᵢˡ − μ̂ᵢˡ
            errors = mu - mu_hat

            # 3. Update attention based on current prediction errors
            alpha = self.compute_attention(errors, edge_index)

            # 4. Update beliefs via local gradient descent on free energy
            mu = self.inference_step(mu, errors, errors_lower, alpha_lower, edge_index)
        # ────────────────────────────────────────────────────────────────

        # Final error after convergence
        mu_hat = self.top_down_prediction(mu_upper, alpha, edge_index)
        errors = mu - mu_hat

        return mu, errors, alpha

    def _uniform_attention(
        self, edge_index: Tensor, n_nodes: int, device: torch.device
    ) -> Tensor:
        """Initialise with uniform (1/degree) attention weights."""
        src, dst = edge_index
        n_edges = edge_index.shape[1]
        degree = torch.zeros(n_nodes, device=device)
        degree.scatter_add_(0, dst, torch.ones(n_edges, device=device))
        alpha = 1.0 / (degree[dst] + 1e-8)
        return alpha


# ===========================================================================
# Full PC-GAT model
# ===========================================================================

class PCGAT(nn.Module):
    """
    Full PC-GAT model: a stack of PCGATLayer modules.

    The model is driven entirely by local inference and Hebbian-style weight
    updates — no global backpropagation is required to train the internal layers.

    Optionally, the output representations can be fed into a standard task head
    (e.g., softmax classifier) trained with backprop, matching the hybrid
    strategy common in predictive-coding literature.

    Args:
        layer_dims:         List of feature dimensions from input to output.
                            E.g. [128, 64, 32] → 2-layer PC-GAT,
                            input dim 128, hidden 64, output 32.
        n_inference_steps:  Inference iterations per layer per forward pass.
        inference_lr:       Step size for belief updates.
        dropout:            Attention dropout rate.
        activation:         Generative activation ('relu', 'tanh', 'linear').
    """

    def __init__(
        self,
        layer_dims: List[int],
        n_inference_steps: int = 20,
        inference_lr: float = 0.1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least 2 elements.")

        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1

        # Layer l predicts layer l from layer l+1.
        # in_features  = dimension of the upper layer (layer_dims[l+1])
        # out_features = dimension of this layer      (layer_dims[l])
        self.layers = nn.ModuleList(
            [
                PCGATLayer(
                    in_features=layer_dims[l + 1],
                    out_features=layer_dims[l],
                    n_inference_steps=n_inference_steps,
                    inference_lr=inference_lr,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(self.n_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Run inference through all PC-GAT layers.

        `x` is treated as the highest-level (top) representation (layer L).
        Each lower layer l infers its beliefs against predictions from layer l+1.

        Args:
            x:          Input node features (top-level)  [N, layer_dims[-1]]
            edge_index: Graph connectivity               [2, E]

        Returns:
            output:     Beliefs at the bottom layer      [N, layer_dims[0]]
            all_errors: List of prediction errors per layer (bottom → top)
            all_alpha:  List of attention weights per layer (bottom → top)
        """
        all_mu: List[Tensor] = []
        all_errors: List[Tensor] = []
        all_alpha: List[Tensor] = []

        mu_upper = x
        errors_lower: Optional[Tensor] = None
        alpha_lower: Optional[Tensor] = None

        for layer in self.layers:
            mu, errors, alpha = layer(
                mu_upper=mu_upper,
                edge_index=edge_index,
                errors_lower=errors_lower,
                alpha_lower=alpha_lower,
            )
            all_mu.append(mu)
            all_errors.append(errors)
            all_alpha.append(alpha)

            # Pass information downward for the next layer
            mu_upper = mu
            errors_lower = errors
            alpha_lower = alpha

        return mu, all_errors, all_alpha

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def total_free_energy(self, all_errors: List[Tensor]) -> Tensor:
        """
        Variational free energy: F = Σₗ Σᵢ |εᵢˡ|²

        This is the quantity minimised implicitly during inference.
        A lower value indicates better model-data fit.
        """
        return sum(e.pow(2).sum() for e in all_errors)

    def anomaly_scores(self, all_errors: List[Tensor]) -> Tensor:
        """
        Per-node anomaly score = sum of squared prediction errors across layers.

        Atypical nodes that deviate from the model's expectations produce
        high errors and therefore high anomaly scores — giving the model
        native anomaly-detection capability without any extra objective.

        Returns:
            scores: Anomaly score per node [N]
        """
        return sum(e.pow(2).mean(dim=-1) for e in all_errors)

    def local_update_all(
        self,
        all_errors: List[Tensor],
        x: Tensor,
        all_mu: List[Tensor],
        all_alpha: List[Tensor],
        edge_index: Tensor,
    ) -> None:
        """
        Accumulate local Hebbian weight gradients for all layers.

        Call this after `forward`, then call `optimizer.step()` to apply.

        Args:
            all_errors: Prediction errors per layer (output of forward)
            x:          Top-level input features
            all_mu:     Beliefs per layer (output of forward)
            all_alpha:  Attention weights per layer (output of forward)
            edge_index: Graph edges
        """
        for l, layer in enumerate(self.layers):
            mu_upper = x if l == 0 else all_mu[l - 1]
            layer.local_weight_update(
                errors=all_errors[l],
                mu_upper=mu_upper,
                alpha=all_alpha[l],
                edge_index=edge_index,
            )
