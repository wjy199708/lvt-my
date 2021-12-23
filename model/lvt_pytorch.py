import torch
import torch.nn as nn
import einops as eops


class CSA(nn.Module):
    def __init__(self, pre_ch, unfold_kernel_size=2):
        super(CSA, self).__init__()
        # self.ln_norm = nn.LayerNorm(pre_ch)
        self.kernel_size = unfold_kernel_size
        self.unflod = nn.Unfold(unfold_kernel_size, stride=2)
        self.simliarity = nn.Linear(pre_ch, unfold_kernel_size ** 4)
        self.BMM=nn.Conv2d()
    
    def forward(self, x):
        """
        args:
            x: image feature from previous (b,c,h,w)


        """
        b, c, h, w = x.shape
        _x = x
        x = nn.LayerNorm([c, h, w])(x)  # -> (B,C,H,W)
        
        x_unfold=self._unfold(x)
        
        x_similiarity=self._similarity(_x)
        
        
        
        return x

    def _unfold(self,x):
        x = self.unflod(x)  # -> (B,C*kenel_size,L)
        x = x.view(
            1, c, self.kernel_size, self.kernel_size, int(h / 2), int(w / 2)
        ).permute(0, 4, 5, 2, 3, 1) # b, h/2, w/2, k_size, k_size , c
        return x
    
    def _similarity(self,x):
        x_sim = self.simliarity(
            _x.permute(0, 2, 3, 1)
        )  # b,c,h,w -> b,h,w,kernel_size**4
        x_sim = x_sim.view()
        
        return x_sim

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


if __name__ == "__main__":
    x = torch.rand(1, 3, 1024, 1024)
    # print(x.shape)
    model = CSA(3)

    print(model(x).shape)
