import torch
from torch import nn

from .deform_offsets_module import dca_offsets_layer
from .deform_conv_1d import DeformConv1D

from einops import rearrange, reduce, repeat

import math 

class dca_layer(nn.Module):
    """Constructs a Deformable ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        attn_type: Types of attention - ["use_local_deform", "use_nonlocal_deform", 
                                         "use_both_weighted_all_zeros", "use_both_weighted_nonlocal_zero"] 
    """
    def __init__(self, channel=512, k_size=3, attn_type='use_local_deform', use_shuffle=False):
        super(dca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if attn_type == 'use_local_deform':
            self.conv_offset_local = nn.Conv1d(1, k_size, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
       
        
        if use_shuffle:
            self.groups = int(2 ** (math.ceil(math.log(k_size, 2))))

        self.attn_type = attn_type
        self.use_shuffle = use_shuffle 

        self.dropout = nn.Dropout(p=0.2)
        self.deform_conv = DeformConv1D(1, 1, kernel_size=k_size)
        self.sigmoid = nn.Sigmoid()


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_grp = c // groups 
        x = x.view(b, groups, channels_per_grp, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        if self.use_shuffle:
            y = self.channel_shuffle(y, self.groups)

        y_reshaped = rearrange(y, 'b c h w -> b c (h w)')
        y_reshaped = rearrange(y_reshaped, 'b c n -> b n c')

        if self.attn_type == 'use_local_deform':
            offset_local = self.conv_offset_local(y_reshaped)
            y_local = self.deform_conv(y, offset_local)
            
            # y_local = self.dropout(y_local)
            y_local = rearrange(y_local, 'b n c -> b c n')
            y_local = rearrange(y_local, 'b c (h w) -> b c h w', h=1, w=1)
            
            if self.use_shuffle:
                y_local = self.channel_shuffle(y, groups = c // self.groups)
            
            attended = self.sigmoid(y_local).expand_as(x) * x

        return attended 
