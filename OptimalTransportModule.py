import torch
import torch.nn as nn
import ot

class EOT(nn.Module):
    def __init__(self, 
                 n_iters: int = 50, 
                 epsilon: float = 0.1,
                 tol = 1e-2):
        super().__init__()
        self.n_iters = n_iters
        self.eps = epsilon
        self.tol = tol

    def forward(self,
                C: torch.Tensor,
                a: torch.Tensor = None,
                b: torch.Tensor = None, normalize= True) -> torch.Tensor:
        """
        C: cost matrix of shape (n, m)
        a: source marginal, shape (n,)
        b: target marginal, shape (m,)

        Returns:
            T: transport plan of shape (n, m)
        """
        n, m = C.shape

        # Default to uniform marginals
        if normalize:
            if a is None:
                a = torch.full((n,), 1.0 / n, device=C.device, dtype=C.dtype)
            else:
                a = a / a.sum()
            if b is None:
                b = torch.full((m,), 1.0 / m, device=C.device, dtype=C.dtype)
            else:
                b = b / b.sum()
        else:
            a = a.float()
            b = b.float()

        # Gibbs kernel
        K = torch.exp(-C / self.eps)  # (n, m)

        # Initialize dual variables
        u = torch.ones_like(a)        # (n,)
        v = torch.ones_like(b)        # (m,)
        u_prev = u.clone()
        v_prev = v.clone()

        for i in range(self.n_iters):
            u = a / (K @ v)            # (n,) = (n,m) @ (m,)
            v = b / (K.t() @ u)        # (m,) = (m,n) @ (n,)

            # convergence check
            diff = (u - u_prev).pow(2).mean() + (v - v_prev).pow(2).mean()
            if diff < self.tol:
                break

            u_prev.copy_(u)
            v_prev.copy_(v)

        # Build transport plan: diag(u) @ K @ diag(v)
        T = (u.unsqueeze(1) * K) * v.unsqueeze(0)  # (n, m)

        return T

class EPOT(nn.Module):
    def __init__(self,epsilon: float = 0.1,tol = 1e-2,n_iters: int = 50, s =0.9):
        super().__init__()
        self.s = s
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.tol = tol

    def forward(self, 
                C: torch.Tensor, 
                a: torch.Tensor = None, 
                b: torch.Tensor = None, 
                ) -> torch.Tensor:
        """
        C: cost matrix of shape (n, m)
        a: source marginal, shape (n,)
        b: target marginal, shape (m,)

        Returns:
            T: transport plan of shape (n, m)
        """
        n, m = C.shape
        if a is None:
            a = torch.full((n,), 1.0 / n, device=C.device, dtype=C.dtype)
        else:
            a = a / a.sum()
        if b is None:
            b = torch.full((m,), 1.0 / m, device=C.device, dtype=C.dtype)
        else:
            b = b / b.sum()
        C_np = C.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy() 
        # Use POT's partial earth mover's distance solver (partial OT)
        T_partial = ot.partial.entropic_partial_wasserstein(a_np, b_np, C_np, m=self.s, reg = self.epsilon, stopThr =self.tol, numItermax = self.n_iters, log=False)

        # Convert back to torch tensor
        return torch.tensor(T_partial, dtype=torch.float32)
    
class POT(nn.Module):
    def __init__(self, s =0.9):
        super().__init__()
        self.s = s

    def forward(self, 
                C: torch.Tensor, 
                a: torch.Tensor = None, 
                b: torch.Tensor = None, 
                ) -> torch.Tensor:
        """
        C: cost matrix of shape (n, m)
        a: source marginal, shape (n,)
        b: target marginal, shape (m,)

        Returns:
            T: transport plan of shape (n, m)
        """
        n, m = C.shape
        if a is None:
            a = torch.full((n,), 1.0 / n, device=C.device, dtype=C.dtype)
        else:
            a = a / a.sum()
        if b is None:
            b = torch.full((m,), 1.0 / m, device=C.device, dtype=C.dtype)
        else:
            b = b / b.sum()
        C_np = C.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy() 
        # Use POT's partial earth mover's distance solver (partial OT)
        T_partial = ot.partial.partial_wasserstein(a_np, b_np, C_np, m=self.s, nb_dummies=1, log=False)

        # Convert back to torch tensor
        return torch.tensor(T_partial, dtype=torch.float32)
    
class OT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                C: torch.Tensor, 
                a: torch.Tensor = None, 
                b: torch.Tensor = None, 
                ) -> torch.Tensor:
        """
        C: cost matrix of shape (n, m)
        a: source marginal, shape (n,)
        b: target marginal, shape (m,)

        Returns:
            T: transport plan of shape (n, m)
        """
        n, m = C.shape
        if a is None:
            a = torch.full((n,), 1.0 / n, device=C.device, dtype=C.dtype)
        else:
            a = a / a.sum()
        if b is None:
            b = torch.full((m,), 1.0 / m, device=C.device, dtype=C.dtype)
        else:
            b = b / b.sum()
        C_np = C.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy() 
        T = ot.emd(a_np, b_np, C_np)

        # Convert back to torch tensor
        return torch.tensor(T, dtype=torch.float32)

