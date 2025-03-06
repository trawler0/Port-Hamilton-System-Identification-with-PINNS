import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from kan import KAN


class VTU(nn.Module):

    def __init__(self, N, strict=False):
        super(VTU, self).__init__()

        idx = []
        for j in range(N):
            rng = range(j + 1, N) if strict else range(j, N)
            for i in rng:
                idx.append(j * N + i)
        self.register_buffer('idx', torch.tensor(idx, dtype=torch.long))
        self.N = N
        self.register_buffer("zeros", torch.zeros(N * N))

    def forward(self, x):
        B, L = x.shape
        assert len(self.idx) == L
        zeros = self.zeros.unsqueeze(0).expand(B, -1)
        idx = self.idx.unsqueeze(0).expand(B, -1)

        M = torch.scatter(zeros, 1, idx, x)
        M = M.view(B, self.N, self.N)
        return M



class MLP(nn.Sequential):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            depth,
            batchnorm_input=False,
            activation=partial(nn.SiLU, inplace=True),
    ):
        layers = [nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
        )]
        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation(),
            ))
        layers.append(nn.Linear(hidden_dim, output_dim))
        if batchnorm_input:
            layers = [nn.BatchNorm1d(input_dim)] + layers
        super(MLP, self).__init__(*layers)


class Baseline(nn.Module):

    def __init__(self, input_dim, hidden_dim, u_dim, depth):
        super(Baseline, self).__init__()
        self.mlp = MLP(input_dim + u_dim, hidden_dim, input_dim + u_dim, depth=depth)

    def forward(self, x, u):
        u_dim = u.shape[-1]
        xu = torch.cat([x, u], dim=-1)
        out = self.mlp(xu)
        return out[:, :-u_dim], out[:, -u_dim:]



class JLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.J_ = nn.Parameter(torch.randn(dim, dim))

        nn.init.kaiming_normal_(self.J_)

    def forward(self, x, grad_H):
        J = self.J_ - self.J_.T
        return F.linear(grad_H, J)

    def reparam(self, x):
        J = (self.J_ - self.J_.T)
        return J.unsqueeze(0).expand(x.size(0), -1, -1).permute(0, 2, 1)


class RLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.R_ = nn.Parameter(torch.randn(dim, dim))

        nn.init.kaiming_normal_(self.R_)

    def forward(self, x, grad_H):
        return F.linear(F.linear(grad_H, self.R_), self.R_.T) / math.sqrt(self.R_.size(0))

    def reparam(self, x):
        R = self.R_ @ self.R_.T / math.sqrt(self.R_.size(0))
        return R.unsqueeze(0).expand(x.size(0), -1, -1)


# G linear in the sense of corresponding to linear PH-system, but it means it is constant
class GLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.G_ = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        return self.G_.expand(x.size(0), -1)


# quadratic Hamiltonian
class Grad_HLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.Q_ = nn.Parameter(torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.randn(dim))

        nn.init.kaiming_normal_(self.Q_)


    def forward(self, x):
        Q = self.Q_ @ self.Q_.T
        return F.linear(x, Q, bias=self.bias)

    def H(self, x):
        Q = self.Q_ @ self.Q_.T
        return (F.linear(x, Q, bias=self.bias) @ x.unsqueeze(-1)).squeeze(-1)


class JDefault(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            depth,
            arch="mlp"
    ):
        super().__init__()
        if arch == "mlp":
            self.J_ = MLP(input_dim, hidden_dim, ((input_dim - 1) * input_dim) // 2, depth)
        elif arch == "kan":
            self.J_ = KAN([input_dim] + [hidden_dim] * depth + [((input_dim - 1) * input_dim) // 2])
        self.vsu = VTU(input_dim, strict=True)

    def forward(self, x, grad_H):
        J_ = self.vsu(self.J_(x))
        J = J_ - J_.permute(0, 2, 1)
        return (grad_H.unsqueeze(1) @ J).squeeze(1)

    def reparam(self, x):
        J_ = self.vsu(self.J_(x))
        J = J_ - J_.permute(0, 2, 1)
        return J.permute(0, 2, 1)


class RDefault(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            depth,
            arch="mlp"
    ):
        super().__init__()
        if arch == "mlp":
            self.R_ = MLP(input_dim, hidden_dim, input_dim ** 2, depth)
        elif arch == "kan":
            self.R_ = KAN([input_dim] + [hidden_dim] * depth + [input_dim ** 2])

    def forward(self, x, grad_H):
        R_ = self.R_(x)
        d = int(math.sqrt(R_.size(-1)))
        R_ = R_.view(R_.size(0), d, d)
        return (grad_H.unsqueeze(1) @ R_ @ R_.permute(0, 2, 1)).squeeze(1) / math.sqrt(d)

    def reparam(self, x):
        R_ = self.R_(x)
        d = int(math.sqrt(R_.size(-1)))
        R_ = R_.view(R_.size(0), d, d)
        R = R_ @ R_.permute(0, 2, 1) / math.sqrt(d)
        return R


class Grad_H(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth, arch="mlp"):
        super().__init__()
        if arch == "mlp":
            self.H = MLP(input_dim, hidden_dim, 1, depth)
        elif arch == "kan":
            self.H = KAN([input_dim] + [hidden_dim] * depth + [1])

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad = True
            H = self.H(x)
            grad_H = torch.autograd.grad(
                inputs=x,
                outputs=H,
                grad_outputs=torch.ones_like(H),
                create_graph=True,
                retain_graph=True,
            )[0]
        return grad_H

# just for the spring
class RQuadratic(nn.Module):

    def __init__(self, D):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(D+1, D**2))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x, grad_H):
        B, D = x.size()
        x = torch.cat([x, torch.ones((B, 1), device=x.device)], dim=1)
        x = (x @ self.weight).view(-1, D, D)
        R = x @ x.transpose(1, 2) / math.sqrt(D)
        return (grad_H.unsqueeze(1) @ R).squeeze(1)

    def reparam(self, x):
        B, D = x.size()
        x = torch.cat([x, torch.ones((B, 1), device=x.device)], dim=1)
        x = (x @ self.weight).view(-1, D, D)
        R = x @ x.transpose(1, 2) / math.sqrt(D)
        return R

# one can often guess J from calculations with Poisson brackets
class JSpring(nn.Module):

    def __init__(self):
        super().__init__()
        J = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=torch.float32)
        self.register_buffer("J", J)

    def forward(self, x, grad_H):
        return F.linear(grad_H, self.J.T)

    def reparam(self, x):
        return self.J

class J_Sigmoid(nn.Module):

    def __init__(self, input_dim, hidden_dim, excitation="linear"):
        super().__init__()
        self.input_dim = input_dim
        self.J1 = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        self.J2 = nn.Parameter(torch.randn(input_dim + hidden_dim, input_dim).T)
        if excitation == "linear":
            self.J_sigma = nn.Linear(input_dim, hidden_dim)
        elif excitation == "mlp":
            self.J_sigma = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )

        nn.init.kaiming_normal_(self.J1)
        nn.init.kaiming_normal_(self.J2)

    def forward(self, x, grad_H):
        sigma = torch.sigmoid(self.J_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        J_left = F.linear((F.linear(grad_H, self.J1) * sigma), self.J2)
        J_right = F.linear((F.linear(grad_H, self.J2.T) * sigma), self.J1.T)
        return J_left - J_right

    def reparam(self, x):
        sigma = torch.sigmoid(self.J_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)  # B x hidden
        sigma = torch.diag_embed(sigma)  # B x hidden x hidden
        J1 = self.J1.T.unsqueeze(0).expand(x.size(0), -1, -1)  # B x input x hidden
        J2 = self.J2.T.unsqueeze(0).expand(x.size(0), -1, -1)
        J_left = J1 @ sigma @ J2  # B x input x input
        J_right = J2.permute(0, 2, 1) @ sigma @ J1.permute(0, 2, 1)  # B x input x input
        return J_left - J_right

class R_Sigmoid(nn.Module):

    def __init__(self, input_dim, hidden_dim, excitation="linear"):
        super().__init__()
        self.input_dim = input_dim
        self.R_ = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        if excitation == "linear":
            self.R_sigma = nn.Linear(input_dim, hidden_dim)
        elif excitation == "mlp":
            self.R_sigma = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )

        nn.init.kaiming_normal_(self.R_)

    def forward(self, x, grad_H):
        sigma = torch.sigmoid(self.R_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        x = F.linear(F.linear(grad_H, self.R_) * sigma, self.R_.T)
        return x

    def reparam(self, x):
        sigma = torch.sigmoid(self.R_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        sigma = torch.diag_embed(sigma)
        R_ = self.R_.T.unsqueeze(0).expand(x.size(0), -1, -1)
        R = R_ @ sigma @ R_.permute(0, 2, 1)
        return R

class PHNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth, J="default", R="default", grad_H="gradient", G="mlp", excitation="mlp",
                 u_dim=1):
        super().__init__()
        if J == "linear":
            self.J = JLinear(input_dim)
        elif J == "default":
            self.J = JDefault(input_dim, hidden_dim, depth)
        elif J == "default_kan":
            self.J = JDefault(input_dim, hidden_dim, depth, arch="kan")
        elif J == "sigmoid":
            self.J = J_Sigmoid(input_dim, hidden_dim, excitation)
        elif J == "spring":
            self.J = JSpring()
        else:
            raise ValueError("Unknown J")

        if R == "linear":
            self.R = RLinear(input_dim)
        elif R == "quadratic":
            self.R = RQuadratic(input_dim)
        elif R == "default":
            self.R = RDefault(input_dim, hidden_dim, depth)
        elif R == "sigmoid":
            self.R = R_Sigmoid(input_dim, hidden_dim, excitation)
        elif R == "default_kan":
            self.R = RDefault(input_dim, hidden_dim, depth, arch="kan")
        else:
            raise ValueError("Unknown R")

        if grad_H == "gradient":
            self.grad_H = Grad_H(input_dim, hidden_dim, depth)
        elif grad_H == "gradient_kan":
            self.grad_H = Grad_H(input_dim, hidden_dim, depth, arch="kan")
        elif grad_H == "linear":
            self.grad_H = Grad_HLinear(input_dim)
        else:
            raise ValueError("Unknown grad_H")

        if G == "mlp":
            self.G = MLP(input_dim, hidden_dim, input_dim * u_dim, depth)
        elif G == "kan":
            self.G = KAN([input_dim] + [hidden_dim] * depth + [input_dim * u_dim])
        elif G == "linear":
            self.G = GLinear(input_dim * u_dim)
        else:
            raise ValueError("Unknown G")

    def forward(self, x, u):
        grad_H = self.grad_H(x)
        J_hat = self.J(x, grad_H)
        R_hat = self.R(x, grad_H)
        G = self.G(x)

        G = G.view(G.size(0), G.size(1) // u.size(-1), u.size(-1))

        y = (G.transpose(1, 2) @ grad_H.unsqueeze(-1)).squeeze(-1)
        input_term = (G @ u.unsqueeze(-1)).squeeze(-1)

        xdot = J_hat - R_hat + input_term

        return xdot, y

    def reparam(self, x):
        J = self.J.reparam(x)
        R = self.R.reparam(x)
        return J, R

    def H(self, x):
        return self.grad_H.H(x)


if __name__ == "__main__":
    model = PHNNModel(4, 4, 3, u_dim=2)
    x = torch.randn(256, 4)
    u = torch.randn(256, 2)
    print(sum(p.numel() for p in model.parameters()))
    print(model(x, u))