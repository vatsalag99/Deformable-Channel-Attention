import torch
from torch import nn
from torch.nn.parameter import Parameter

from einops import rearrange, reduce, repeat

class dca_offsets_layer(nn.Module):
    """Constructs a Offset Generation module.
    """
    def __init__(self, channel, n_offsets):
        super(dca_offsets_layer, self).__init__()

        self.channel = channel
        self.n_offsets = n_offsets

    def covariance_features(self, x):
        """
        Takes in a feature map and returns the unnormalized covariance matrix
        """
        m_batchsize, C, height, width = x.size()
        x = x - x.mean(dim=1, keepdim=True) / (x.std(dim=1, keepdim=True) + 1e-5)
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        return energy

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        cov_matrix = self.covariance_features(x).reshape(m_batchsize, C, 1, C)

        _, locations = torch.topk(cov_matrix, self.n_offsets, dim=1)
        delta = torch.stack(self.n_offsets*[torch.arange(0, self.channel)], dim=0)
        delta = torch.stack(m_batchsize * [delta], dim=0)
        offsets = locations.squeeze() - delta.cuda()
        return offsets 
