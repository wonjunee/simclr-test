import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- Step 1: Create donut clusters ---- #
def make_donut_clusters(n=1000, num_clusters=4, width=0.2, seed=0):
    torch.manual_seed(seed)
    samples_per_cluster = n // num_clusters
    X = []
    y = []
    radii = torch.linspace(1.0, 2.0, num_clusters)  # Increasing radius

    for i, r in enumerate(radii):
        theta = 2 * torch.pi * torch.rand(samples_per_cluster)
        radius = r + width * (torch.rand(samples_per_cluster) - 0.5)
        x = radius * torch.cos(theta)
        y_ = radius * torch.sin(theta)
        points = torch.stack([x, y_], dim=1)
        X.append(points)

    return torch.cat(X, dim=0)

# ---- Step 2: Embedding map f: R^2 -> R^1 ---- #
class Embed1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---- Step 3: NT-Xent Loss ---- #
def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: shape (n, d), should be normalized
    """
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # shape (2n, d)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T)  # (2n, 2n)
    sim /= temperature

    # Mask to exclude self-similarity
    labels = torch.cat([torch.arange(n), torch.arange(n)], dim=0)
    labels = labels.to(z.device)

    # Create positive pairs mask
    pos_mask = torch.eye(2 * n, dtype=torch.bool).roll(shifts=n, dims=0).to(z.device)

    # Compute loss
    exp_sim = torch.exp(sim)
    pos_sim = exp_sim[pos_mask].view(2 * n, 1)
    neg_sim = exp_sim.masked_fill(pos_mask, 0).sum(dim=1, keepdim=True)

    loss = -torch.log(pos_sim / (pos_sim + neg_sim))
    return loss.mean()

# ---- Training ---- #
def train():
    # Generate dataset
    X = make_donut_clusters(n=1024)
    dataset = X

    model = Embed1D()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    temperature = 0.5

    for epoch in range(200):
        model.train()

        # Data augmentation (Gaussian noise)
        noise = 0.05 * torch.randn_like(dataset)
        x1 = dataset + noise
        x2 = dataset + 0.05 * torch.randn_like(dataset)

        z1 = model(x1)
        z2 = model(x2)

        loss = nt_xent_loss(z1, z2, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, dataset

# ---- Run and visualize ---- #
if __name__ == "__main__":
    model, dataset = train()
    with torch.no_grad():
        z = model(dataset).squeeze()

    plt.scatter(dataset[:, 0], dataset[:, 1], c=z.numpy(), cmap='viridis')
    plt.colorbar(label='1D embedding')
    plt.title("2D Donuts Colored by 1D Embedding")
    plt.axis("equal")
    plt.show()
