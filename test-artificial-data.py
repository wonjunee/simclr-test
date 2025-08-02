import argparse
import os
import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch import nn, optim, Tensor
from typing import Tuple
import matplotlib.gridspec as gridspec
import torch.autograd as autograd
import tqdm
import matplotlib.cm as cm
import random

import warnings
from torchvision import datasets, transforms
from PIL import Image

warnings.filterwarnings("ignore")


save_folder = "figures"
os.makedirs(save_folder, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("figures/no-nn", exist_ok=True)
os.makedirs("figures/with-nn", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20_000, help="number of epochs of training")
parser.add_argument("--tau",      type=float, default=0.1, help="temperature for the eta function in the loss")
parser.add_argument("--lr",      type=float, default=1e-3, help="learning rate")
parser.add_argument("--plot_every", type=int, default=1000, help="plotting frequency")
args = parser.parse_args()
print(args)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
cmap = plt.cm.viridis

input_dim = 2


# np.random.seed(0)
# torch.manual_seed(1)

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, tau=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.register_buffer("zeros", (torch.zeros(batch_size * 2, batch_size * 2)).float())

        self.relu = nn.ReLU()



    def forward_att_only(self, Tx1: Tensor, Tx2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Contrastive‐style loss between two batches of embeddings.

        For each i:
            repulsion_i = (1/n) ∑_j exp( -||Tx1_i - Tx2_j||² / (2*tau) )
            attraction_i = exp( -||Tx1_i - Tx2_i||² / (2*tau) )
        loss = (1/n) ∑_i log(1 + repulsion_i / (attraction_i + eps))
            + lambda_reg * mean((Tx1 + 1)^2)

        Returns:
            loss:        scalar loss
            avg_attr:    mean_i attraction_i
            avg_repel:   mean_i repulsion_i
        """
        eps = 1e-8
        n = Tx1.size(0)
        # Positive pairs (diagonal): squared distances
        pos_sq = (Tx1 - Tx2).pow(2).sum(dim=1)                        # (n,)
        # Contrastive term: log(1 + repulsion/attraction)
        loss_terms = pos_sq / (2 * self.tau)
        loss = loss_terms.mean()

        return loss, 0, 0


    def forward_rep_only(self, Tx1: Tensor, Tx2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Contrastive‐style loss between two batches of embeddings.

        For each i:
            repulsion_i = (1/n) ∑_j exp( -||Tx1_i - Tx2_j||² / (2*tau) )
            attraction_i = exp( -||Tx1_i - Tx2_i||² / (2*tau) )
        loss = (1/n) ∑_i log(1 + repulsion_i / (attraction_i + eps))
            + lambda_reg * mean((Tx1 + 1)^2)

        Returns:
            loss:        scalar loss
            avg_attr:    mean_i attraction_i
            avg_repel:   mean_i repulsion_i
        """
        eps = 1e-8
        n = Tx1.size(0)

        # Pairwise squared L2 distances, shape (n, n)
        sq_dists = torch.cdist(Tx1, Tx2, p=2).pow(2)

        # Repulsion: average over all j
        repulsion = torch.exp(-sq_dists / (2 * self.tau)).mean(dim=1)  # (n,)
        # Contrastive term: log(1 + repulsion/attraction)
        loss_terms = torch.log1p(repulsion)      # (n,)
        loss = loss_terms.mean()

        return loss, 0, 0



    def forward(self, Tx1: Tensor, Tx2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Contrastive‐style loss between two batches of embeddings.

        For each i:
            repulsion_i = (1/n) ∑_j exp( -||Tx1_i - Tx2_j||² / (2*tau) )
            attraction_i = exp( -||Tx1_i - Tx2_i||² / (2*tau) )
        loss = (1/n) ∑_i log(1 + repulsion_i / (attraction_i + eps))
            + lambda_reg * mean((Tx1 + 1)^2)

        Returns:
            loss:        scalar loss
            avg_attr:    mean_i attraction_i
            avg_repel:   mean_i repulsion_i
        """
        eps = 1e-8
        n = Tx1.size(0)

        # Pairwise squared L2 distances, shape (n, n)
        sq_dists = torch.cdist(Tx1, Tx2, p=2).pow(2)

        # Repulsion: average over all j
        repulsion = torch.exp(-sq_dists / (2 * self.tau)).mean(dim=1)  # (n,)

        # Positive pairs (diagonal): squared distances
        pos_sq = (Tx1 - Tx2).pow(2).sum(dim=1)                        # (n,)
        attraction = torch.exp(-pos_sq / (2 * self.tau))              # (n,)

        # Contrastive term: log(1 + repulsion/attraction)
        loss_terms = torch.log1p(repulsion / (attraction + eps))      # (n,)
        loss = loss_terms.mean()

        return loss, attraction.mean(), repulsion.mean()



    def forward2(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        numerator = torch.exp(positives / self.tau)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.tau)

        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))

        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss, loss, loss

class FixedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        # store as buffer so it's on the right device but not trainable
        self.register_buffer('weight', weight)
    def forward(self, x):
        # x has shape [..., N], weight is [N, 2] → output [..., 2]
        return x.matmul(self.weight)
    
# Define your network architecture here
class Classifier(nn.Module):
    def __init__(self, out_dim=2, noise=0.1, data=None):
        super().__init__()
        
        # noise for the transform
        self.out_dim=out_dim
        self.noise = noise
        self.max_angle = 2*np.pi * 1.0

        self.A = torch.zeros((2,2))
        self.A.requires_grad = False
        self.lam = 0.1
        # self.A[1,:] = self.lam
        self.A[1,1] = self.lam
        self.model = None

        self.b = torch.randn(2000, 1, device=device) *0.02 # worked with tau = 0.1
        # self.b = torch.randn(2000, 1, device=device) * 10 # doesn't work with tau = 1e-5

        self.target_size = (28,28)
        self.blur_kernel = 5
        self.blur_sigma = 1.0
        self.blur_sigma_range = (0.1,2.0)
        self.scale_range = (0.8, 1.2)

        # if data is not None:
        #     self.b = data
        #     norms = self.b.norm(dim=1, keepdim=True)
        #     # 2) create the direction vector (0,1)
        #     direction = torch.tensor([0.0, 1.0], device=self.b.device)
        #     # 3) form B
        #     self.b = 0.05 * self.b + norms * direction * 0.03

        N = 100
        self.f =  nn.Sequential(
            nn.Linear(2, N),
            nn.LeakyReLU(0.2),
            nn.Linear(N, 1)
        )

    def transform(self, x):
        # offset = (torch.rand(x.shape[0],1 , device=device)*2-1)
        offset = torch.randn(x.shape[0],1 , device=device)
        # offset[:,0] *= 0
        return x + offset
    
    def transform_rotate(self, x):
        """
        x: Tensor of shape (batch_size, 2)
        returns: Tensor of same shape, where each point has been rotated
                by a random angle in [-max_angle, +max_angle].
        """
        # sample one angle per example in [-max_angle, +max_angle]
        angles = (torch.rand(x.shape[0], device=x.device) * 2 - 1) * self.max_angle
        # compute cos and sin, shape (batch_size, 1)
        c = torch.cos(angles).unsqueeze(1)
        s = torch.sin(angles).unsqueeze(1)

        # split x into x and y components
        x0 = x[:, 0:1]
        x1 = x[:, 1:2]

        # apply rotation: [ x'; y' ] = [cos -sin; sin cos] [x; y]
        x_rot = torch.cat([c * x0 - s * x1,
                        s * x0 + c * x1], dim=1)
        return x_rot
    
    def transform_both(self, x):
        """
        Apply a random rescale *then* a random rotation to each 2D point in x.
        """
        # first apply your original random‐scale
        x_scaled = self.transform(x)

        # then apply your random‐rotate
        x_scaled_and_rotated = self.transform_rotate(x_scaled)

        return x_scaled_and_rotated
    
    def transform2(self, x):
        """
        Adds uniform noise (sampled uniformly from a ball of radius `self.noise`)
        to each vector in x.

        Args:
            x (torch.Tensor): Tensor of shape (*, d), where d is the feature dimension.
        
        Returns:
            torch.Tensor: The tensor x with added uniform noise.
        """
        # Get feature dimension.
        d = x.shape[-1]
        
        # Sample random directions by drawing from a normal distribution and normalizing.
        noise = torch.randn_like(x)
        norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        unit_noise = noise / norm
        
        # Sample random radii for each vector.
        # To sample uniformly inside a ball, scale u^(1/d) by the desired radius, where u ~ Uniform(0,1).
        u = torch.rand(x.shape[:-1], device=x.device)
        scale = u.pow(1.0/d) * self.noise
        scale = scale.unsqueeze(-1)
        
        # Compute the noise and add it to x.
        noise_ball = unit_noise * scale
        noise_ball[:,0] *= 0
        return x + noise_ball
    
    def Phi(self, z):
        """
        return the cumulative Gaussian function.
        Output ranges from 0 to 1.
        """
        return z
        # return 0.5 * (1 + torch.erf(z/np.sqrt(2)))
    
    def T(self, x):
        fx = self.f(x)
        fx_model = self.model.f(x)
        return fx - fx_model + self.b[:x.shape[0],:] # + x # self.b[:x.shape[0],:]

        
    def forward(self, x):
        # # apply transformations
        x1 = self.transform(x)
        x2 = self.transform(x)

        x1 = x1.reshape((x1.shape[0],-1))
        x2 = x2.reshape((x2.shape[0],-1))


        Tx1 = self.T(x1)
        Tx2 = self.T(x2)

        return Tx1, Tx2

    def save_model(self, model):
        self.model = copy.deepcopy(model)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight.data)
        # nn.init.constant_(m.weight.data, 0)
        nn.init.normal_(m.weight.data, 0, 0.1)
        nn.init.constant_(m.bias.data, 0)


# creating data
tau =args.tau
out_dim = 1
delta      = 0.3

if True:
    split = 4
    N_per_cluster = 400
    N = split * N_per_cluster
    # Define cluster centers (equally around the origin)
    centers = torch.tensor([[-1.0,  -1.0],
                            [-1.0,   1.0],
                            [ 1.0,  -1.0],
                            [ 1.0,   1.0]], device=device)

    # Preallocate dataset and label tensors
    val_circle = torch.empty(N, 2, device=device)
    labels = torch.empty(N, dtype=torch.long, device=device)

    # Generate uniformly distributed points in a disk of radius 0.3 for each cluster
    for i in range(split):
        start_ind = i * N_per_cluster
        end_ind = (i + 1) * N_per_cluster
        
        # Uniformly sample angles between 0 and 2pi
        angles = torch.rand(N_per_cluster, device=device) * 2 * math.pi
        # To achieve uniform distribution in the disk, sample radius as sqrt(u)*R with u ~ Uniform(0,1)
        radii = torch.sqrt(torch.rand(N_per_cluster, device=device)) * 0.8
        offsets = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
        val_circle[start_ind:end_ind] = centers[i] + offsets
        labels[start_ind:end_ind] = i


if False:
    split = 2
    N_per_cluster = 100
    N = split * N_per_cluster

    val_circle = torch.randn(N, input_dim, device=device)  # 10000-D Gaussian samples
    val_circle[:N//2, 0] = torch.abs(val_circle[:N//2, 0])      # first half: positive first axis
    val_circle[N//2:, 0] = -torch.abs(val_circle[N//2:, 0])     # second half: negative first axis
    labels = torch.zeros(N, dtype=torch.long, device=device)
    labels[N//2:] = 1



if False:
    split = 4
    N_per_cluster = 200
    N = split * N_per_cluster
    # Define cluster centers (equally around the origin)
    centers = torch.tensor([
                            [-3.0,   0.0],
                            [-1.0,   0.0],
                            [ 1.0,   0.0],
                            [ 3.0,   0.0],], device=device)

    # Preallocate dataset and label tensors
    val_circle = torch.empty(N, 2, device=device)
    labels = torch.empty(N, dtype=torch.long, device=device)

    # Generate uniformly distributed points in a disk of radius 0.3 for each cluster
    for i in range(split):
        start_ind = i * N_per_cluster
        end_ind = (i + 1) * N_per_cluster
        offsets = (torch.rand(N_per_cluster, 2, device=device)*2 - 1) * 0.6
        offsets[:,1] *= 2.0
        val_circle[start_ind:end_ind] = centers[i] + offsets
        labels[start_ind:end_ind] = i

if False:
    
    # parameters
    split         = 6        # number of clusters
    N_per_cluster = 200      # points per cluster
    radius        = 5.0      # base circle radius
    sigma_rad     = 0.9      # radial noise std
    sigma_ang     = 0.2     # angular noise std (radians)

    # 1) generate cluster angles
    angles = torch.linspace(0, 2*math.pi, split+1, device=device)[:-1]  

    # 2) preallocate
    N = split * N_per_cluster
    val_circle = torch.empty(N, 2, device=device)
    labels     = torch.empty(N, dtype=torch.long, device=device)

    # 3) sample points for each cluster
    for i, theta in enumerate(angles):
        start = i * N_per_cluster
        end   = (i + 1) * N_per_cluster

        # a) sample angular offsets Δθ ~ N(0, sigma_ang)
        dtheta = torch.randn(N_per_cluster, device=device) * sigma_ang
        thetas = theta + dtheta  # shape (N_per_cluster,)

        # b) sample radial offsets r_noise ~ N(0, sigma_rad)
        r_noise = torch.randn(N_per_cluster, device=device) * sigma_rad
        radii   = radius + r_noise  # shape (N_per_cluster,)

        # c) convert polar → Cartesian
        x = radii * torch.cos(thetas)
        y = radii * torch.sin(thetas)
        val_circle[start:end, 0] = x
        val_circle[start:end, 1] = y

        labels[start:end] = i

if False: # donut data
    split         = 4         # number of clusters
    N_per_cluster = 300       # points per cluster
    width         = 0.2       # donut thickness (total radial span)
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) specify the radii for each cluster
    radii = [0.2 + 0.3*i for i in range(split)]  
    # => [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]

    # 2) preallocate
    N = split * N_per_cluster
    val_circle = torch.empty(N, 2, device=device)
    labels     = torch.empty(N, dtype=torch.long, device=device)

    # 3) for each cluster, sample uniformly in the annulus [r - width/2, r + width/2]
    for i, r in enumerate(radii):
        start = i * N_per_cluster
        end   = start + N_per_cluster

        # sample angles ∼ Uniform[0, 2π)
        thetas = torch.rand(N_per_cluster, device=device) * 2 * math.pi

        # sample radial offsets ∼ Uniform[–width/2, +width/2]
        r_offsets = (torch.rand(N_per_cluster, device=device) - 0.5) * width
        radii_i   = r + r_offsets

        # convert to Cartesian
        x = radii_i * torch.cos(thetas)
        y = radii_i * torch.sin(thetas)

        val_circle[start:end, 0] = x
        val_circle[start:end, 1] = y
        labels[start:end] = i

    val_circle_evenly = torch.empty(N, 2, device=device)
    labels_evenly     = torch.empty(N, dtype=torch.long, device=device)
    for i, r in enumerate(radii):
        start = i * N_per_cluster
        end   = start + N_per_cluster

        # sample angles ∼ Uniform[0, 2π)
        thetas = torch.linspace(0, 2*math.pi, N_per_cluster+1, device=device)[:-1] + (i*math.pi/2)

        # sample radial offsets ∼ Uniform[–width/2, +width/2]
        radii_i   = r

        # convert to Cartesian
        x = radii_i * torch.cos(thetas)
        y = radii_i * torch.sin(thetas)

        val_circle_evenly[start:end, 0] = x
        val_circle_evenly[start:end, 1] = y
        labels_evenly[start:end] = i
    

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(val_circle[:,0].detach().cpu(),val_circle[:,1].detach().cpu(),c=labels.detach().cpu(),s=100)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# ax.set_ylim([-1.0,1.0])
fig.savefig(f'{save_folder}/data.png', bbox_inches='tight')

batch_size = N
model = Classifier(out_dim=out_dim, noise = 0.1, data=val_circle).to(device)
# model.apply(initialize_weights)
criterion = ContrastiveLoss(batch_size=batch_size, tau=tau).to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr)

random_ind = torch.randperm(val_circle.shape[0])
val_circle = val_circle[random_ind]
labels = labels[random_ind]

# TODO Train the network here
epochs = args.epochs
count = 0

model.save_model(model)

### With NN
if True:
    skip = 10
    val_circle = val_circle.to(device).detach()
    fig_count = 0

    particle_array = []
    pbar = tqdm.tqdm(range(epochs))
    for e in pbar:
        optimizer.zero_grad()
        Tx1, Tx2 = model(val_circle)
        loss, loss_at, loss_re = criterion.forward(Tx1, Tx2)
        loss.backward()
        optimizer.step()

        if e == 0:
            Tx = Tx1.clone().detach().cpu()
            # Plot the 1D particles as points on a vertical line, colored by label
            fig_particles, ax_particles = plt.subplots(figsize=(8, 8))
            y = Tx[:, 0].numpy()
            x = np.zeros_like(y)
            label_vals = labels.cpu().numpy()
            unique_labels = np.unique(label_vals)
            scatter = ax_particles.scatter(label_vals, y, c=label_vals, s=300, edgecolor='k', linewidth=0.7)
            ax_particles.set_xlabel("Labels", fontsize=24)
            ax_particles.set_ylabel("1D Embedding", fontsize=24)
            ax_particles.set_xticks([])
            ax_particles.set_title("1D Particles (Initial)", fontsize=16)
            ax_particles.grid(True, axis='y')
            ax_particles.tick_params(axis='y', labelsize=24)
            fig_particles.tight_layout()
            fig_particles.savefig(f'{save_folder}/with-nn/particles-initial.png', dpi=200)
            plt.close(fig_particles)


        if e % skip== 0:
            with torch.no_grad():
                Tx = Tx1.clone().detach().cpu()
                particle_array.append(Tx[:, 0])  # store only 1D embedding

                fig = plt.figure(figsize=(24, 8))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 0.1])  # Make second plot wider

                ax_data = fig.add_subplot(gs[0])
                ax_particle = fig.add_subplot(gs[1])

                # Left plot: original data
                ax_data.scatter(
                    val_circle[:, 0].detach().cpu(),
                    val_circle[:, 1].detach().cpu(),
                    c=labels.detach().cpu(),
                    s=100
                )
                ax_data.set_title('Input Data', fontsize=18)
                ax_data.set_xlabel('x', fontsize=16)
                ax_data.set_ylabel('y', fontsize=16)
                ax_data.tick_params(axis='both', which='major', labelsize=14)
                ax_data.grid()
                ax_data.set_aspect('equal', adjustable='box')

                # Right plot: evolution of 1D embeddings over iterations
                embedding_matrix = torch.stack(particle_array)  # shape: (T, n)
                T, n = embedding_matrix.shape
                time = torch.arange(T) * skip # adjust if you save every skipepochs

                normalized_labels = (labels / labels.max()).cpu()

                for i in range(n):
                    ax_particle.plot(
                        time.numpy(),
                        embedding_matrix[:, i].numpy(),
                        color=plt.cm.viridis(normalized_labels[i].item()),
                        alpha=0.5
                    )

                ax_particle.set_title("1D Embedding over Iterations", fontsize=18)
                ax_particle.set_xlabel("Iteration", fontsize=16)
                ax_particle.set_ylabel(r"$f(w,x)$", fontsize=16)
                ax_particle.tick_params(axis='both', which='major', labelsize=14)
                ax_particle.grid(True)
                # ax_particle.set_yscale('log')

                fig.tight_layout()
                fig.savefig(f'{save_folder}/with-nn/particle-{e//skip}.png', dpi=300)
                plt.close(fig)
                
                pbar.set_description(f"figure saved in {save_folder}/with-nn/particle-{e//skip}.png")
