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
        attn_type: Types of attention - ["use_local_deform", "use_nonlocal_deform", 
                                         "use_both_weighted_all_zeros", "use_both_weighted_nonlocal_zero"] 
    """
    def __init__(self, channel=512, k_size=3, attn_type='use_local_deform'):
        super(dca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        uses_local = attn_type != "use_nonlocal_deform"
        uses_nonlocal = attn_type != "use_local_deform"
        uses_weighted = attn_type == "use_both_weighted_all_zeros" or attn_type == "use_both_weighted_nonlocal_zero" 


        if uses_nonlocal:
            self.conv_offset_nonlocal = dca_offsets_layer(channel, k_size)
            
            if attn_type == "use_nonlocal_deform":
                self.weight_nonlocal = nn.Parameter(torch.zeros(1, channel, 1, 1))
                self.bias = nn.Parameter(torch.ones(1, channel, 1, 1))
        
        if uses_local:
            self.conv_offset_local = nn.Conv1d(1, k_size, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        
        if uses_weighted: 
            if attn_type == "use_both_weighted_nonlocal_zero":
                self.weight_local = nn.Parameter(torch.ones(1, channel, 1, 1))
            else:
                self.weight_local = nn.Parameter(torch.zeros(1, channel, 1, 1))
            self.weight_nonlocal = nn.Parameter(torch.zeros(1, channel, 1, 1))
            self.bias = nn.Parameter(torch.ones(1, channel, 1, 1))

        self.uses_local = uses_local 
        self.uses_nonlocal = uses_nonlocal
        self.uses_weighted = uses_weighted 
    
        self.attn_type = attn_type

        self.deform_conv = DeformConv1D(1, 1, kernel_size=k_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y_reshaped = rearrange(y, 'b c h w -> b c (h w)')
        y_reshaped = rearrange(y_reshaped, 'b c n -> b n c')

        if self.uses_nonlocal:
            offset_nonlocal = self.conv_offset_nonlocal(x)
            y_nonlocal = self.deform_conv(y, offset_nonlocal)
            y_nonlocal = rearrange(y_nonlocal, 'b n c -> b c n')
            y_nonlocal = rearrange(y_nonlocal, 'b c (h w) -> b c h w', h=1, w=1)

        if self.uses_local:
            offset_local = self.conv_offset_local(y_reshaped)
            y_local = self.deform_conv(y, offset_local)
            y_local = rearrange(y_local, 'b n c -> b c n')
            y_local = rearrange(y_local, 'b c (h w) -> b c h w', h=1, w=1)

        if self.uses_weighted:
            y = y_local * self.weight_local + y_nonlocal * self.weight_nonlocal + self.bias
            attended = self.sigmoid(y).expand_as(x) * x
        else:
            if self.uses_local:
                assert self.attn_type == "use_local_deform"
                attended = self.sigmoid(y_local).expand_as(x) * x
            if self.uses_nonlocal:
                assert self.attn_type == "use_nonlocal_deform"
                y_nonlocal = self.weight_nonlocal * y_nonlocal + self.bias 
                attended = self.sigmoid(y_nonlocal).expand_as(x) * x

        return attended 
