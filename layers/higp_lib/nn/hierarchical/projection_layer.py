from torch import nn
import torch


class ProjectionLayer(nn.Module):
    def forward(self, y, Q):
        """
        Project y onto the null space of Q.
        """
        # projection matrix: I - Q^T(QQ^T)^{-1}Q = I - pinv(Q)Q
        # Q_s = Q.T @ torch.linalg.inv(Q @ Q.T) @ Q
        Q_s = torch.linalg.lstsq(Q, Q).solution
        P = torch.eye(Q.size(-1), device=Q.device) - Q_s
        # project
        if P.dim() == 3:
            P = P.unsqueeze(1)
        y = torch.matmul(P, y)
        return y
