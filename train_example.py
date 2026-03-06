"""
PC-GAT Training Example
========================

Demonstrates the PC-GAT algorithm on a simple node-classification task using a
synthetic graph (Barabási–Albert random graph with two-class node labels).

Two training modes are shown:

  1. Local PC learning   — weights updated with the Hebbian/PC rule (no backprop).
  2. Hybrid learning     — PC inference + standard backprop on a task head.

Run:
    python train_example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from pc_gat import PCGAT


# ---------------------------------------------------------------------------
# Synthetic graph generator
# ---------------------------------------------------------------------------

def make_synthetic_graph(
    n_nodes: int = 200,
    feat_dim: int = 16,
    n_edges_per_node: int = 3,
    seed: int = 42,
) -> tuple:
    """
    Create a random Barabási–Albert graph with binary node labels.

    Returns:
        x:          Node features  [N, feat_dim]
        edge_index: Edges          [2, E]  (undirected, stored as both directions)
        labels:     Node labels    [N]  (0 or 1)
    """
    torch.manual_seed(seed)

    # Random node features
    x = torch.randn(n_nodes, feat_dim)

    # Labels: nodes in the first half are class 0, second half class 1
    labels = torch.zeros(n_nodes, dtype=torch.long)
    labels[n_nodes // 2 :] = 1

    # Make features class-discriminative
    x[labels == 1] += 1.0

    # Build a simple ring + random edges (approximating BA topology)
    edges = []
    for i in range(n_nodes):
        # Ring
        edges.append((i, (i + 1) % n_nodes))
        edges.append(((i + 1) % n_nodes, i))
        # Random extra edges
        for _ in range(n_edges_per_node - 1):
            j = torch.randint(n_nodes, (1,)).item()
            if j != i:
                edges.append((i, j))
                edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index, labels


# ---------------------------------------------------------------------------
# Mode 1 – Pure local PC learning (no global backprop)
# ---------------------------------------------------------------------------

def train_local_pc(
    model: PCGAT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> None:
    """
    Train PC-GAT with purely local Hebbian weight updates.

    No loss.backward() is called.  Weight gradients are accumulated by
    `local_update_all` according to the PC rule and applied by the optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n=== Mode 1: Local PC learning (no global backprop) ===")
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        output, all_errors, all_alpha = model(x, edge_index)

        # Collect beliefs per layer (forward returns only the last; rebuild list)
        all_mu = []
        mu_upper = x
        for layer in model.layers:
            mu, errors, alpha = layer(
                mu_upper=mu_upper,
                edge_index=edge_index,
            )
            all_mu.append(mu)
            mu_upper = mu

        # Accumulate local weight gradients
        model.local_update_all(all_errors, x, all_mu, all_alpha, edge_index)

        optimizer.step()

        if epoch % 10 == 0:
            fe = model.total_free_energy(all_errors).item()
            print(f"  Epoch {epoch:4d}  |  Free energy: {fe:.4f}")

    print("Local PC training complete.\n")


# ---------------------------------------------------------------------------
# Mode 2 – Hybrid: PC inference + backprop on classifier head
# ---------------------------------------------------------------------------

class PCGATClassifier(nn.Module):
    """PC-GAT backbone with a linear classification head."""

    def __init__(self, layer_dims: list, n_classes: int, **kwargs):
        super().__init__()
        self.backbone = PCGAT(layer_dims, **kwargs)
        self.head = nn.Linear(layer_dims[0], n_classes)

    def forward(self, x, edge_index):
        # PC-GAT inference (no global backprop inside the backbone needed)
        output, all_errors, all_alpha = self.backbone(x, edge_index)
        logits = self.head(output)
        return logits, all_errors, all_alpha


def train_hybrid(
    model: PCGATClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> None:
    """
    Hybrid training: PC-GAT backbone (local rules) + backprop on the task head.
    """
    # Optimizer only touches the head parameters here;
    # backbone weights are updated via local PC rule.
    head_optimizer = optim.Adam(model.head.parameters(), lr=lr)
    backbone_optimizer = optim.Adam(model.backbone.parameters(), lr=lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    print("=== Mode 2: Hybrid (PC backbone + backprop task head) ===")
    for epoch in range(1, n_epochs + 1):
        model.train()
        head_optimizer.zero_grad()
        backbone_optimizer.zero_grad()

        logits, all_errors, all_alpha = model(x, edge_index)

        # Task loss (backprop through the head only)
        loss = criterion(logits, labels)
        loss.backward()
        head_optimizer.step()

        # Free-energy metric (no backward used)
        fe = model.backbone.total_free_energy(all_errors).item()

        if epoch % 20 == 0:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            print(
                f"  Epoch {epoch:4d}  |  Loss: {loss.item():.4f}  "
                f"|  Acc: {acc:.3f}  |  Free energy: {fe:.4f}"
            )

    print("Hybrid training complete.\n")


# ---------------------------------------------------------------------------
# Anomaly detection demo
# ---------------------------------------------------------------------------

def demo_anomaly_detection(
    model: PCGAT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_anomalies: int = 10,
) -> None:
    """
    Inject anomalous nodes (random features) and measure anomaly scores.

    High-scoring nodes should correspond to the injected anomalies.
    """
    model.eval()
    x_noisy = x.clone()
    anomaly_idx = torch.randperm(x.shape[0])[:n_anomalies]
    x_noisy[anomaly_idx] = torch.randn(n_anomalies, x.shape[1]) * 5.0  # outliers

    with torch.no_grad():
        _, all_errors, _ = model(x_noisy, edge_index)
        scores = model.anomaly_scores(all_errors)

    top_k = scores.topk(n_anomalies).indices
    injected = set(anomaly_idx.tolist())
    detected = set(top_k.tolist())
    overlap = len(injected & detected)

    print("=== Anomaly detection demo ===")
    print(f"  Injected anomaly nodes : {sorted(injected)}")
    print(f"  Top-{n_anomalies} by PC error    : {sorted(detected)}")
    print(f"  Recall                 : {overlap}/{n_anomalies}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_NODES = 200
    FEAT_DIM = 16
    LAYER_DIMS = [8, 16, FEAT_DIM]   # output=8, hidden=16, input=FEAT_DIM
    N_CLASSES = 2

    x, edge_index, labels = make_synthetic_graph(n_nodes=N_NODES, feat_dim=FEAT_DIM)

    print(f"Graph: {N_NODES} nodes, {edge_index.shape[1]} edges, {FEAT_DIM}-dim features\n")

    # ---- Mode 1: pure local PC learning ----
    pc_model = PCGAT(
        layer_dims=LAYER_DIMS,
        n_inference_steps=10,
        inference_lr=0.05,
        dropout=0.1,
    )
    train_local_pc(pc_model, x, edge_index, labels)

    # ---- Anomaly detection ----
    demo_anomaly_detection(pc_model, x, edge_index)

    # ---- Mode 2: hybrid training ----
    hybrid_model = PCGATClassifier(
        layer_dims=LAYER_DIMS,
        n_classes=N_CLASSES,
        n_inference_steps=10,
        inference_lr=0.05,
        dropout=0.1,
    )
    train_hybrid(hybrid_model, x, edge_index, labels)
