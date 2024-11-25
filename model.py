import torch
from torch import nn
import torch.nn.functional as F
from functools import partial


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


def functional_calculus(A: torch.Tensor, func) -> torch.Tensor:
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("The last two dimensions of A must be equal (square matrices).")

    if not torch.allclose(A, A.transpose(-1, -2), atol=1e-6):
        raise ValueError("Input matrix A must be symmetric.")

    eigenvalues, eigenvectors = torch.linalg.eigh(A)

    f_eigenvalues = func(eigenvalues)

    f_eigenvalues_diag = torch.diag_embed(f_eigenvalues)
    transformed_A = torch.matmul(eigenvectors, torch.matmul(f_eigenvalues_diag, eigenvectors.transpose(-1, -2)))

    return transformed_A


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


class ModelOld(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(ModelOld, self).__init__()
        self.input_dim = input_dim
        self.J1 = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        self.J2 = nn.Parameter(torch.randn(input_dim + hidden_dim, input_dim).T)
        self.J_sigma = nn.Linear(input_dim, hidden_dim)

        self.R1 = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        self.R_sigma = nn.Linear(input_dim, hidden_dim)

        self.H = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.G = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

        nn.init.kaiming_normal_(self.J1)
        nn.init.kaiming_normal_(self.J2)
        nn.init.kaiming_normal_(self.R1)

    def _R(self, x, grad_H):
        sigma = torch.sigmoid(self.R_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        x = F.linear(F.linear(grad_H, self.R1) * sigma, self.R1.T)
        return x

    def _J(self, x, grad_H):
        sigma = torch.sigmoid(self.J_sigma(x))
        sigma = torch.cat([sigma, torch.ones(sigma.size(0), self.input_dim, device=sigma.device)], dim=-1)
        J_left = F.linear((F.linear(grad_H, self.J1) * sigma), self.J2)
        J_right = F.linear((F.linear(grad_H, self.J2.T) * sigma), self.J1.T)
        return J_left - J_right

    def grad_H(self, x):
        return torch.autograd.grad(
            inputs=x,
            outputs=self.H(x),
            grad_outputs=torch.ones_like(self.H(x)),
            create_graph=True,
            retain_graph=True,
        )[0]

    def forward(self, x, u):
        x.requires_grad = True
        grad_H = self.grad_H(x)
        J_R = self._J(x, grad_H) - self._R(x, grad_H)
        G = self.G(x)

        return J_R + G * u


class J_Sigmoid(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth, excitation="linear"):
        super().__init__()
        self.input_dim = input_dim
        self.J1 = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        self.J2 = nn.Parameter(torch.randn(input_dim + hidden_dim, input_dim).T)
        if excitation == "linear":
            self.J_sigma = nn.Linear(input_dim, hidden_dim)
        elif excitation == "mlp":
            self.J_sigma = MLP(input_dim, hidden_dim, hidden_dim, depth=depth)

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

    def __init__(self, input_dim, hidden_dim, depth, excitation="linear"):
        super().__init__()
        self.input_dim = input_dim
        self.R_ = nn.Parameter(torch.randn(input_dim, input_dim + hidden_dim).T)
        if excitation == "linear":
            self.R_sigma = nn.Linear(input_dim, hidden_dim)
        elif excitation == "mlp":
            self.R_sigma = MLP(input_dim, hidden_dim, hidden_dim, depth=depth)

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


class JLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.J_ = nn.Parameter(torch.randn(dim, dim))

        nn.init.kaiming_normal_(self.J_)

    def forward(self, x, grad_H):
        J = self.J_ - self.J_.T
        return F.linear(grad_H, J)

    def reparam(self, x):
        J = self.J_ - self.J_.T
        return J.unsqueeze(0).expand(x.size(0), -1, -1)


class RLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.R_ = nn.Parameter(torch.randn(dim, dim))

        nn.init.kaiming_normal_(self.R_)

    def forward(self, x, grad_H):
        return F.linear(F.linear(grad_H, self.R_), self.R_.T)

    def reparam(self, x):
        R = self.R_ @ self.R_.T
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

        nn.init.kaiming_normal_(self.Q_)

    def forward(self, x):
        Q = self.Q_ @ self.Q_.T
        return F.linear(x, Q)

    def H(self, x):
        Q = self.Q_ @ self.Q_.T
        return (F.linear(x, Q) @ x.unsqueeze(-1)).squeeze(-1)


class JMatmul(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            depth,
    ):
        super().__init__()
        self.J_ = MLP(input_dim, hidden_dim, ((input_dim - 1) * input_dim) // 2, depth)
        self.vsu = VTU(input_dim, strict=True)

    def forward(self, x, grad_H):
        J_ = self.vsu(self.J_(x))
        J = J_ - J_.permute(0, 2, 1)
        return (grad_H.unsqueeze(1) @ J).squeeze(1)

    def reparam(self, x):
        J_ = self.vsu(self.J_(x))
        J = J_ - J_.permute(0, 2, 1)
        return J


class RMatmul(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            depth
    ):
        super().__init__()
        self.R_ = MLP(input_dim, hidden_dim, ((input_dim + 1) * input_dim) // 2, depth)
        self.vtu = VTU(input_dim)

    def forward(self, x, grad_H):
        R_ = self.vtu(self.R_(x))
        return (grad_H.unsqueeze(1) @ R_ @ R_.permute(0, 2, 1)).squeeze(1)

    def reparam(self, x):
        R_ = self.vtu(self.R_(x))
        R = R_ @ R_.permute(0, 2, 1)
        return R


class RMatmulRescale(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            depth
    ):
        super().__init__()
        self.R_ = MLP(input_dim, hidden_dim, ((input_dim + 1) * input_dim) // 2, depth)
        self.log_tij = torch.nn.Parameter(torch.randn(1, input_dim, input_dim))
        self.bij = torch.nn.Parameter(torch.randn(1, input_dim, input_dim))
        self.vtu = VTU(input_dim)

    def forward(self, x, grad_H):
        R_ = self.vtu(self.R_(x))
        R = R_ @ R_.permute(0, 2, 1) * torch.exp(self.log_tij) + self.bij
        return (grad_H.unsqueeze(1) @ R).squeeze(1)

    def reparam(self, x):
        R_ = self.vtu(self.R_(x))
        R = R_ @ R_.permute(0, 2, 1) * torch.exp(self.log_tij) + self.bij
        return R


# do not know what this represents, it is not a quadratic Hamiltonian
class Grad_HMatmul(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth):
        super().__init__()
        self.Q = MLP(input_dim, hidden_dim, ((input_dim + 1) * input_dim) // 2, depth)
        self.vtu = VTU(input_dim)

    def forward(self, x):
        Q_ = self.vtu(self.Q(x))
        Q = Q_.permute(0, 2, 1) @ Q_
        return (x.unsqueeze(1) @ Q).squeeze(1)

    def H(self, x):
        return None


class Grad_H(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth):
        super().__init__()
        self.H = MLP(input_dim, hidden_dim, 1, depth)

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


class PHNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth, J="sigmoid", R="sigmoid", grad_H="gradient", G="mlp", excitation="linear",
                 u_dim=1):
        super().__init__()
        if J == "sigmoid":
            self.J = J_Sigmoid(input_dim, hidden_dim, depth, excitation)
        elif J == "linear":
            self.J = JLinear(input_dim)
        elif J == "matmul":
            self.J = JMatmul(input_dim, hidden_dim, depth)
        else:
            raise ValueError("Unknown J")

        if R == "sigmoid":
            self.R = R_Sigmoid(input_dim, hidden_dim, depth, excitation)
        elif R == "linear":
            self.R = RLinear(input_dim)
        elif R == "matmul":
            self.R = RMatmul(input_dim, hidden_dim, depth)
        elif R == "matmul_rescale":
            self.R = RMatmulRescale(input_dim, hidden_dim, depth)
        else:
            raise ValueError("Unknown R")

        if grad_H == "gradient":
            self.grad_H = Grad_H(input_dim, hidden_dim, depth)
        elif grad_H == "linear":
            self.grad_H = Grad_HLinear(input_dim)
        # ?
        elif grad_H == "matmul":
            self.grad_H = Grad_HMatmul(input_dim, hidden_dim, depth)
        else:
            raise ValueError("Unknown grad_H")

        if G == "mlp":
            self.G = MLP(input_dim, hidden_dim, input_dim * u_dim, depth)
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
    model = PHNNModel(3, 64, J="linear", R="linear", grad_H="gradient", G="linear", excitation="mlp", u_dim=2)
    x = torch.randn(10, 3)
    u = torch.randn(10, 2)
    print(model(x, u))