import torch
from torch._C import parse_ir
import torch.nn as nn
import torch.nn.functional as F
import einops as eops


class CSA(nn.Module):
    def __init__(self, pre_ch, unfold_kernel_size=2):
        super(CSA, self).__init__()
        self.in_ch = pre_ch
        # self.ln_norm = nn.LayerNorm(pre_ch)
        self.kernel_size = unfold_kernel_size
        self.unflod = nn.Unfold(unfold_kernel_size, stride=2)
        self.conv2d_1 = nn.Conv2d(pre_ch, pre_ch, kernel_size=1, stride=2)
        self.linear1 = nn.Linear(pre_ch, unfold_kernel_size**4)

        self.BMM = nn.Conv2d(in_channels=pre_ch,
                             out_channels=pre_ch**2,
                             kernel_size=1,
                             stride=1)
        self.out_ch = self.BMM.out_channels

    def forward(self, x):
        """
        args:
            x: image feature from previous (b,c,h,w)


        """
        b, c, h, w = x.shape
        _x = x
        x = nn.LayerNorm([c, h, w])(x)  # -> (B,C,H,W)

        x_unfold = self._unfold(x)

        x_similiarity = self._similarity(_x).reshape(-1, self.kernel_size**2,
                                                     self.kernel_size**2)

        param_v = self.BMM(
            x_unfold.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(
                -1, self.kernel_size**2,
                self.out_ch)  # (h/2*w/2)*k*k*c_in -> (h/2*w/2)*k*k*c_out

        x = torch.bmm(x_similiarity,
                      param_v).reshape(b, -1, self.kernel_size**2,
                                       self.out_ch).permute(0, 3, 2, 1)
        x = x.reshape(b, self.out_ch * self.kernel_size**2, -1)
        print('x===', x.shape,
              b * self.out_ch * self.kernel_size**2 * x.shape[-1],
              b * h * w * self.out_ch)
        x = F.fold(x,
                   output_size=(h, w),
                   kernel_size=self.kernel_size,
                   stride=2)

        x = x.repeat(3) + _x
        return x

    def _unfold(self, x):
        b, c, h, w = x.shape
        x = self.unflod(x)  # -> (B,C*kenel_size,L)
        x = x.view(1, c, self.kernel_size, self.kernel_size, int(h / 2),
                   int(w / 2)).permute(0, 4, 5, 2, 3,
                                       1)  # b, h/2, w/2, k_size, k_size , c
        x = x.reshape(b, -1, self.kernel_size * self.kernel_size, c)
        return x

    def _similarity(self, x):
        b, c, h, w = x.shape
        x_sim = self.conv2d_1(x)
        x_sim = self.linear1(x_sim.permute(
            0, 2, 3, 1))  # b,c,h,w -> b, h, w, kernel_size**4
        x_sim = x_sim.reshape(b, (int(h / 2) * int(w / 2)),
                              self.kernel_size**2,
                              self.kernel_size**2)  # b, h/2, w/2, k**2, k**2

        return x_sim  # b, (h/2 * w/2), k**2, k**2


class ASA(nn.Module):
    def __init__(self):
        super(ASA, self).__init__()

    def forward(self):
        pass


class LVT(nn.Module):
    def __init__(self):
        super(LVT, self).__init__()

    def forward(self, x):
        pass


class Downsample(nn.Module):
    """
    Image to Patch Embedding, downsampling between stage1 and stage2
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, kernel_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim,
                              out_embed_dim,
                              kernel_size=kernel_size,
                              stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 3, 1024, 1024)
    # print(x.shape)
    model = CSA(3)

    print(model(x).shape)
