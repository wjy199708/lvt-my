import torch
import torch.nn as nn
import einops as eops


class CSA(nn.Module):
    def __init__(self, pre_ch) -> None:
        super(CSA, self).__init__()
        self.ln_norm = nn.LayerNorm(pre_ch)
        self.unflod = nn.Unfold(2, 2, stride=2)

    def forward(self, x):
        """
        args:
            x: image feature from previous (h,w,c)


        """

        _x = x
        x = self.ln_norm(x)  # -> (B,H,W,C)
        


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
