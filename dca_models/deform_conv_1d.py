import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
from einops import rearrange, reduce, repeat


class DeformConv1D(nn.Module):
    def __init__(self, inc=1, outc=1, kernel_size=3, padding=1, bias=None):
        super(DeformConv1D, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ConstantPad1d(kernel_size // 2, value=0)
        self.conv_kernel = nn.Conv1d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        # x - b, c, 1, 1
        # offset - b k c (k = kernel_size)
       
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = rearrange(x, 'b c n -> b n c')

        offset = offset.float()
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) # Kernel size (only x direction)

        assert N == ks, "Offset size is wrong!"

        if self.padding:
            x = self.zero_padding(x)

        # print('x: ', x.shape)

        # (b, N,  w) - w = c
        p = self._get_p(offset, dtype)

        # (b, w, N)
        p = p.contiguous().permute(0, 2, 1)

        # print('p: ', p.shape)

        q_left = Variable(p.data, requires_grad=False).floor()
        q_right = q_left + 1

        q_left = torch.clamp(q_left[..., :N], 0, x.size(2)-1).long()
        q_right = torch.clamp(q_right[..., :N], 0, x.size(2)-1).long()

        # print('q_left: ', q_left.shape)
        # print('q_right: ', q_right.shape)


        # (b, h, w, N)
        mask = (p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding)).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.clamp(p[..., :N], 0, x.size(2)-1)

        # print('pnew: ', p.shape)

        # linear kernel (b, h, w, N)
        g_left = (1 + (q_left[..., :N].type_as(p) - p[..., :N]))
        g_right = (1 - (q_right[..., :N].type_as(p) - p[..., :N]))

        # print('g_left: ', g_left.shape)
        # print('g_right: ', g_right.shape)

        # (b, c, h, w, N)
        x_q_left = self._get_x_q(x, q_left, N)
        x_q_right = self._get_x_q(x, q_right, N)

        # (b, c, h, w, N)
        x_offset = g_left.unsqueeze(dim=1) * x_q_left + \
                   g_right.unsqueeze(dim=1) * x_q_right

        # print('x_offset: ', x_offset.shape)

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))[0]
        # (N, 1)
        p_n = p_n_x.flatten()
        p_n = np.reshape(p_n, (1, N, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)
        return p_n

    @staticmethod
    def _get_p_0(w, N, dtype):
        p_0_x = np.meshgrid(range(1, w+1))[0]
        p_0 = p_0_x.flatten().reshape(1, 1, w).repeat(N, axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)
        return p_0

    def _get_p(self, offset, dtype):
        N, w = offset.size(1), offset.size(2)

        # (1, N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, N, h, w)
        p_0 = self._get_p_0(w, N, dtype)

        # print(p_n.shape, p_0.shape)
        p = p_0.to(offset.device) + p_n.to(offset.device) + offset

        return p

    def _get_x_q(self, x, q, N):
        b, channels, kernel_size = q.size()

        x = x.contiguous()

        # (b, h, w, N)
        index = q[..., :N] # offset_x

        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, 1, -1, -1).contiguous().view(b, 1, -1)

        # print('x: ', x.shape, ' index: ', index.shape, ' q: ', q.shape)
        # print(index)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, 1, channels, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, w*ks) for s in range(0, N, ks)], dim=-1)
        # print(x_offset.shape)
        x_offset = x_offset.contiguous().view(b, c, w*ks)

        return x_offset
