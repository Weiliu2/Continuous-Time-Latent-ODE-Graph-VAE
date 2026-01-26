import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torchdiffeq import odeint
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Utilities -----------------------------
def normalize_adjacency(A, eps=1e-6):
    # A: (N, N) adjacency (weighted). Returns sym normalized A_hat = D^{-1/2}(A+I)D^{-1/2}
    N = A.shape[0]
    A_hat = A + torch.eye(N, device=A.device)
    deg = A_hat.sum(dim=1)  # degree
    deg_inv_sqrt = deg.clamp(min=eps).pow(-0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

# Simple RK4 ODE solver for batches: y0: (B, N, D)
def rk4_step(f, y0, t0, t1, steps=1):
    # integrate from t0 to t1 with fixed number of internal steps
    h = (t1 - t0) / steps
    y = y0
    for _ in range(steps):
        k1 = f(t0, y)
        k2 = f(t0 + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t0 + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t0 + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 = t0 + h
    return y

def ode_propagate(odefunc, z0, t0, t1, steps=3, RK4=False):
    if RK4 == False:
        times = torch.tensor([t0, t1], dtype=z0.dtype, device=z0.device)
        zt = odeint(odefunc, z0, times, method="dopri5", rtol=1e-4, atol=1e-5)
        return zt[1]
    else:
        return rk4_step(odefunc, z0, t0, t1, steps=steps)

# ----------------------------- Model pieces -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias) # define the shape of hidden neural W. self.lin(X)=X@W 
    def forward(self, X, A_norm):
        # X: (N, d); A_norm: (N, N)
        return A_norm @ (self.lin(X))


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim) # define the shape of W1
        self.gc2 = GCNLayer(hidden_dim, out_dim) # define the shape of W2
        self.act = nn.ReLU() 
    def forward(self, X, A):
        A_norm = normalize_adjacency(A)
        h1 = self.act(self.gc1(X, A_norm))
        h2 = self.gc2(h1, A_norm)
        return h2  # (N, out_dim) node embeddings


class ODEFunc(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        # small MLP with spectral norm for Lipschitz control
        self.fc1 = nn.Linear(dim + 1, hidden_dim)  # +1 for time
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.softplus = nn.Softplus(beta=1.0)  # smooth activation
        # small scaling to avoid large vector field magnitude
        # self.scale = nn.Parameter(torch.tensor(0.1))  # learnable scaling (small init)
    def forward(self, t, z):
        # z: (B, N, D) or (N, D)
        # We'll support batch dim optional
        has_batch = (z.dim() == 3)
        if not has_batch:
            z = z.unsqueeze(0)  # (1, N, D)
        B, N, D = z.shape
        t_feat = torch.full((B, N, 1), float(t), device=z.device)
        inp = torch.cat([z, t_feat], dim=-1)  # (B, N, D+1)
        h = self.softplus(self.fc1(inp))
        dz = self.fc2(h) # * self.scale
        if not has_batch:
            dz = dz.squeeze(0)
        return dz


class PriorNet(nn.Module):
    """Predicts prior variance for the ODE prior p(z | z^-)."""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
            nn.Softplus()
        )
    def forward(self, z_minus, dt):
        # z_minus: (N, D) or (B,N,D)
        if z_minus.dim() == 2:
            z_minus = z_minus.unsqueeze(0)
        B, N, D = z_minus.shape
        dt_feat = torch.full((B, N, 1), math.log(1.0 + dt), device=z_minus.device)
        inp = torch.cat([z_minus, dt_feat], dim=-1)
        var = self.net(inp) + 1e-6  # ensure positive
        return var.squeeze(0) if var.size(0) == 1 else var  # (N, D) or (B,N,D)


class PosteriorNet(nn.Module):
    """Shared MLP producing mu and logvar per node given z^- and node embedding h."""
    def __init__(self, dim, hdim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim + hdim, hidden),
            nn.Tanh()
        )
        self.mu = nn.Linear(hidden, dim)
        self.logvar = nn.Linear(hidden, dim)
    def forward(self, z_minus, h):
        # z_minus, h: (N, D) and (N, hdim)
        if z_minus.dim() == 3:
            z_minus = z_minus.squeeze(0)
        if h.dim() == 3:
            h = h.squeeze(0)
        inp = torch.cat([z_minus, h], dim=-1)  # (N, dim+hdim)
        feat = self.shared(inp)
        mu = self.mu(feat)
        logvar = self.logvar(feat).clamp(min=-10, max=10)  # numerical safety
        return mu, logvar


class DecoderEdge(nn.Module):
    """Shared decoder MLP for edge means. Outputs scalar mean for each edge pair."""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, z, symmetry=True):
        if z.dim() == 2:
            # z: (N, D) node latents -> compute pairwise means
            N, D = z.shape
            # Efficient pairwise: expand and cat
            zi = z.unsqueeze(1).expand(N, N, D) # node i repeated across rows: z.unsqueeze(1)->(N,1,D), expand(N,N,D)->(N,N,D)
            zj = z.unsqueeze(0).expand(N, N, D) # node j repeated across columns: z.unsqueeze(0)->(1,N,D)
            pair = torch.cat([zi, zj], dim=-1)  # (N, N, 2D)
            mu = self.net(pair).squeeze(-1)     # (N, N)
            if symmetry == True:
                # ensure symmetry by averaging with transpose (optional)
                mu = 0.5 * (mu + mu.t())
            return mu
        elif z.dim() == 3:
            B, N, D = z.shape
            zi = z.unsqueeze(2).expand(-1, N, N, D)
            zj = z.unsqueeze(1).expand(-1, N, N, D)
            pair = torch.cat([zi, zj], dim=-1) # (B, N, N, 2D)
            mu = self.net(pair).squeeze(-1)    # (B, N, N)
            if symmetry == True:
                mu = 0.5 * (mu + mu.transpose(1, 2))
            return mu
        else: 
            raise ValueError(f"Expected z of shape (N,D) or (B,N,D), got {z.shape}")


# ----------------------------- Full model -----------------------------
class LatentODEGraphVAE(nn.Module):
    def __init__(self, node_feat_dim, gcn_hid, latent_dim, ode_hidden=64):
        super().__init__()
        torch.manual_seed(0)
        self.gcn = GCNEncoder(node_feat_dim, gcn_hid, gcn_hid)
        self.ode_func = ODEFunc(latent_dim, hidden_dim=ode_hidden)
        self.prior_net = PriorNet(latent_dim)
        self.post_net = PosteriorNet(latent_dim, gcn_hid)
        self.decoder = DecoderEdge(latent_dim)
        # map node embeddings to initial latent
        self.init_lin = nn.Linear(gcn_hid, latent_dim) # nn.Linear function randomly initialize its coefficients

    def init_latent(self, h):
        # h: (N, gcn_hid)
        return self.init_lin(h)  # (N, latent_dim)
    
    def ode_predict(self, z_t, t0, t1):
        return ode_propagate(self.ode_func, z_t, t0, t1, steps=4, RK4=True)
    
    def bernoulli_nll(self, A, logits):
        # A: (N, N) binary {0,1}, logits: (B, N, N)
        A = A.unsqueeze(0).expand_as(logits)
        # BCE with logits = -log Bernoulli likelihood
        nll = F.binary_cross_entropy_with_logits(logits, A, reduction='none')
        # sum over edges, mean over batch
        return nll.sum(dim=(1,2)).mean()
    
    def gaussian_nll(self, x, mu, sigma2=1.0):
        # x, mu: (N,N), (MC_n,N,N)
        B, N, D = mu.shape
        x = x.unsqueeze(0).expand(B, N, D)
        return 0.5 * ((x - mu) ** 2 / sigma2 + math.log(2 * math.pi * sigma2)).sum() / B
    
    def kl_diag_gauss(self, mu_q, logvar_q, mu_p, var_p):
        # all: (N,D)
        var_q = logvar_q.exp()
        term = (var_q / var_p) + ((mu_q - mu_p) ** 2) / var_p - 1.0 - torch.log(var_q / var_p)
        return 0.5 * term.sum()
    
    def forward(self, X, A, z_prev, t_prev, t_cur, MC_n=10, symmetry=True):
        # param symmetry: Whether the encoded mean is symmetric or not.
        # X: (N, d), A: (N, N)

        N = z_prev.shape[0]
        D = z_prev.shape[1]
        h = self.gcn(X, A)  # (N, gcn_hid)
        z_minus = self.ode_predict(z_prev, t_prev, t_cur)  # (N, D)
        dt = float(t_cur - t_prev)
        prior_var = self.prior_net(z_minus, dt)  # (N, D)
        mu_post, logvar_post = self.post_net(z_minus, h)  # (N, D) each
        std_post = (0.5 * logvar_post).exp()
        eps = torch.randn_like(std_post.unsqueeze(0).expand(MC_n, N, D))
        z_sample = mu_post.unsqueeze(0).expand(MC_n, N, D) + std_post.unsqueeze(0).expand(MC_n, N, D) * eps  # reparam
        logits = self.decoder(z_sample, symmetry)  # (MC_n, N, N)

        # r_loss = self.bernoulli_nll(A, logits)
        r_loss = self.gaussian_nll(A, logits)
        kl_loss = self.kl_diag_gauss(mu_post, logvar_post, z_minus, prior_var)

        return r_loss, kl_loss, mu_post


# ----------------------------- Model learning -----------------------------
def fit(seq, lr=1e-3, weight_decay=1e-5, max_iter=250, MC_n=20, symmetry=True, showstep=False):
    """
    :param seq: Graph time series (t, X, A)
    :param M: Transport cost matrix
    :param lr: learning rate for adam optimizer
    :param weight_decay: Weight decay for adam optimizer
    :param max_iter: Maximum iteration number
    :param MC_n: Sample times of Monte Carol approximation to the expectation loss
    :param symmetry: Whether the decoder mean is symmetric or not
    :param showstep: Whether to show loss during iterating or not
    """
    # get network structure
    t0, X0, A0 = seq[0]
    node_feat_dim = X0.shape[1]
    region_num = X0.shape[0]
    latent_dim = region_num * 2
    gcn_hid = max(region_num*2, node_feat_dim//2)
    model = LatentODEGraphVAE(node_feat_dim=node_feat_dim, gcn_hid=gcn_hid, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize latent from first node embeddings
    with torch.no_grad():
        h0 = model.gcn(X0.to(device), A0.to(device))
    z0 = model.init_latent(h0).detach()  # initial latent (N, D)

    total_loss = []
    # KL annealing schedule
    for it in range(max_iter):
        opt.zero_grad()
        z_prev = z0.clone().to(device)
        t_prev = seq[0][0]
        recon_loss = 0.0
        kl_loss = 0.0
        for k in range(1, len(seq)):
            t_cur, X_cur, A_cur = seq[k]
            X_cur = X_cur.to(device)
            A_cur = A_cur.to(device)
            rloss, ploss, mu_post = model(X_cur, A_cur, z_prev, t_prev, t_cur, MC_n, symmetry)

            recon_loss = recon_loss + rloss
            kl_loss = kl_loss + ploss

            # next initial latent for ODE is posterior mean (teacher forcing style)
            # z_prev = mu_post.detach()  # detach to avoid backprop through time unbounded but make the optimization short on temporal memory
            z_prev = mu_post
            t_prev = t_cur
        
        if it == max_iter-1:
            z_last = z_prev
        loss = recon_loss + kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if showstep == True:
            if it % 50 == 0 or it == max_iter - 1:
                print(f"iter {it:03d} | loss {loss.item():.3f} | recon {recon_loss.item():.3f} | kl {kl_loss.item():.3f}")
        total_loss.append([loss.item(), recon_loss.item(), kl_loss.item()])

    return model, total_loss, z_last

# ----------------------------- Prediction -----------------------------
def prediction(model, t_next, X_next, t_prev, X_prev, A_prev, z_prev, symmetry=True):
    N = X_next.shape[0]
    I = torch.zeros(N, N)
    with torch.no_grad():
        # h_prev = model.gcn(X_prev.to(device), A_prev.to(device))
        # z_prev = model.init_latent(h_prev).to(device)
        h_next = model.gcn(X_next.to(device), I.to(device))
        z_pred_minus = model.ode_predict(z_prev, t_prev, t_next)
        mu_post, logvar_post = model.post_net(z_pred_minus, h_next)
        # std_post = (0.5 * logvar_post).exp()
        # eps = torch.randn_like(std_post)
        # z_sample = mu_post + std_post * eps  # reparam
        logits = model.decoder(mu_post, symmetry) # Gaussian_nll predicts this
        probs = torch.sigmoid(logits) # Bernoulli_nll predicts this
    return logits


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from Synthetic_data_geneation import ComplicatedGraph, SimpleGraph

    # generator = SimpleGraph(N=5, d=10, T=1000)
    # seq = generator.generate_sequence()
    generator = ComplicatedGraph(N=4, dx=10)
    seq = generator.generate_sequence(T=20)
    A=torch.zeros(4,4)
    for i in range(len(seq)):
        A += seq[i][2]
    print(A/len(seq))

    max_iter = 1000
    model, total_loss, z_prev = fit(seq=seq[0:-1], max_iter=max_iter, showstep=True)

    # Run a single prediction step to demonstrate usage
    t_prev, X_prev, A_prev = seq[-2]
    t_next, X_next, A_next_true = seq[-1]
    logits = prediction(model, t_next, X_next, t_prev, X_prev, A_prev, z_prev)
    # print("Predicted adjacency matrix (mean):", logits.cpu().numpy())
    print("Predicted adjacency matrix (mean):", logits)
    print("True adjacency matrix:", A_next_true.to(torch.float32))