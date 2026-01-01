import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNetNoEmbeddings

class TabNetModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, n_d=8, n_a=8, n_steps=3, gamma=1.5):
        super(TabNetModel, self).__init__()
        self.tabnet = TabNetNoEmbeddings(
            input_dim   = input_dim,
            output_dim  = output_dim,
            n_d         = n_d,
            n_a         = n_a,
            n_steps     = n_steps,
            gamma       = gamma,
            epsilon     = 1e-2,       # ← much larger than the 1e‑15 default
            mask_type   = "sparsemax" # the default
        )
        self.sigmoid_ne = True


    def forward(self, x):
        output, _ = self.tabnet(x)
        return output
