import torch
import math
import random

class ComplicatedGraph:
    def __init__(
        self,
        N=30,
        dz=4,
        dx=50,
        C=3,
        alpha=0.4,
        beta=0.5,
        omega=0.6,
        gamma=1.2,
        lambda_comm=15,
        noise_z=0.02,
        noise_x=0.1,
        seed=0
    ):
        # torch.manual_seed(seed)
        # random.seed(seed)

        self.N = N
        self.dz = dz
        self.dx = dx
        self.C = C

        # community assignments
        self.c = torch.randint(0, C, (N,))

        # parameters
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.gamma = gamma
        self.lambda_comm = lambda_comm

        # embeddings
        self.Wc = torch.randn(C, dz)
        self.M = torch.randn(dz, dx)
        self.U = torch.randn(C, dx)

        self.noise_z = noise_z
        self.noise_x = noise_x

        self.phi = 2 * math.pi * torch.rand(N)

    def latent_dynamics(self, z, t):
        drift = -self.alpha * z
        comm = self.Wc[self.c]
        periodic = self.beta * torch.sin(self.omega * t + self.phi).unsqueeze(1)
        noise = self.noise_z * torch.randn_like(z)
        return drift + comm + periodic + noise

    def step_latent(self, z, t, dt):
        # simple Euler (ground truth only)
        return z + dt * self.latent_dynamics(z, t)

    def sample_adjacency(self, z, A_prev=None):
        N = self.N
        dist = torch.cdist(z, z, p=2)
        # logits = -dist # also too complicated for the algorithm, so remove
        logits = torch.zeros(N, N)

        # community bias
        # same_comm = (self.c.unsqueeze(0) == self.c.unsqueeze(1)).float()
        # logits += self.lambda_comm * same_comm

        # # temporal persistence
        if A_prev is not None:
            logits += self.gamma * A_prev

        logits = 0.25 * logits # scale penalty
        logits += 4 * torch.randn(self.N, self.N)
        logits = 0.5 * (logits + logits.T)
        # # logits += -1.0  # sparsity bias
        probs = torch.sigmoid(logits)
        # A = probs >= 0.5 # not correct because there is a uncertain term of latent variable distance
        # A = A.to(dtype=logits.dtype)
        # A = torch.bernoulli(probs)

        # # no self loops
        # A.fill_diagonal_(0)
        # return A
        return probs

    def sample_features(self, z):
        X = z @ self.M + self.U[self.c]
        X += self.noise_x * torch.randn_like(X)
        return X

    def generate_sequence(
        self,
        T=20,
        t0=0.0,
        dt_shape=2.0,
        dt_scale=0.5
    ):
        N, dz = self.N, self.dz

        # irregular times
        times = [t0]
        for _ in range(T - 1):
            dt = torch.distributions.Gamma(dt_shape, dt_scale).sample().item()
            times.append(times[-1] + dt)

        z = torch.randn(N, dz)
        A_prev = None

        data = []

        for k in range(T):
            t = times[k]
            X = self.sample_features(z)
            A = self.sample_adjacency(z, A_prev)

            data.append((t, X, A))

            if k < T - 1:
                dt = times[k+1] - times[k]
                z = self.step_latent(z, t, dt)
                A_prev = A
        return data
    
class SimpleGraph():
    def __init__(self, N=7, d=471, T=4):
        self.N = N
        self.d = d
        self.T = T

    def generate_sequence(self):
        torch.manual_seed(0)
        # create synthetic irregular timestamps and graphs
        times = sorted([0.0] + [random.random() * 5.0 for _ in range(self.T-1)])
        graphs = []
        for t in times:
            X = 0.2 * torch.ones(self.N, self.d) + torch.sqrt(torch.rand(self.N, self.d))
            A = torch.randn(self.N, self.N) * 0.2
            A = 0.5 * (A + A.t())
            A = torch.sigmoid(A)  # centering around 0 with small variance
            # A = A >= 0.5
            # A = A.to(dtype=X.dtype)
            graphs.append((t, X, A))
        return graphs    