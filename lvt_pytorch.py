import torch
import torch.nn as nn
import einops as eops


class CSA_Trans(nn.Module):
    def __init__(self) -> None:
        super(CSA_Trans, self).__init__()
        self.ln_norm = nn.LayerNorm()

    def forward(self, x):
        """
        args:
            x: image feature from previous

        """

        pass


class ASA(nn.Module):
    def __init__(self) -> None:
        super(ASA, self).__init__()

    def forward(self):
        pass


class LVT(nn.Module):
    def __init__(self) -> None:
        super(LVT, self).__init__()

    def forward(self, x):
        pass
