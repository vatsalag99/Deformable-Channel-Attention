import torch
from torch import nn

from .deform_offsets_module import dca_offsets_layer
from .deform_conv_1d import DeformConv1D

from einops import rearrange, reduce, repeat

class dca_layer(nn.Module):
    """Constructs a Deformable ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=512, k_size=3, use_cov=False):
        super(dca_layer, self).__init__()

        self.use_cov = use_cov

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if use_cov:
            self.conv_offset = dca_offsets_layer(channel, k_size)
        else:
            self.conv_offset = nn.Conv1d(1, k_size, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.deform_conv = DeformConv1D(1, 1, kernel_size=k_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y_reshaped = rearrange(y, 'b c h w -> b c (h w)')
        y_reshaped = rearrange(y_reshaped, 'b c n -> b n c')

        if self.use_cov:
            offset = self.conv_offset(x)
        else:
            offset = self.conv_offset(y_reshaped)

#         offset = repeat(torch.tensor([0, 100, 212]), 'k -> b c k', c=512, b=b)
#         offset = rearrange(offset, 'b c k -> b k c')

        y = self.deform_conv(y, offset)
        y = rearrange(y, 'b n c -> b c n')
        y = rearrange(y, 'b c (h w) -> b c h w', h=1, w=1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
