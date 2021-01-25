import torch
from torch import nn

from einops import rearrange, reduce, repeat

import math 

class csa_layer(nn.Module):
    """Constructs a Channel Shuffle Channel Attention Module
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=512, k_size=3, use_prototype=False, channels_per_grp=32):
        super(csa_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
       
        self.use_prototype = use_prototype
        self.channels_per_grp = channels_per_grp
        
        if self.use_prototype:
             self.conv = nn.Conv2d(channel // self.channels_per_grp, 1, kernel_size=1, bias=False)
    
        else:
            self.groups = int(2 ** (math.ceil(math.log(k_size, 2))))

            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                                  padding=(k_size - 1) // 2, bias=False)
        
        self.dropout = nn.Dropout(p=0.2)
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

        if self.use_prototype:
            grp = F.avg_pool1d(x, kernel_size=self.channels_per_grp, stride=self.channels_per_grp)
            relation_matrix = torch.bmm(x.permute(0,2,1), grp).permute(0,2,1).unsqueeze(2)
            channel_attention = self.conv(relation_matrix)
            attended = self.sigmoid(channel_attention).expand_as(x) * x
            return attended 

        y = self.channel_shuffle(y, self.groups)

        y = rearrange(y, 'b c h w -> b c (h w)')
        y = rearrange(y, 'b c n -> b n c')

        y = self.conv(y)
        y = rearrange(y, 'b n c -> b c n') 
        y = rearrange(y, ' b c (h w) -> b c h w' , h=1, w=1)

        y = self.channel_shuffle(y, groups = c // self.groups)
        attended = self.sigmoid(y).expand_as(x) * x

        return attended 
