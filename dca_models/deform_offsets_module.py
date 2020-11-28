import torch
from torch import nn
from torch.nn.parameter import Parameter

class dca_offsets_layer(nn.Module):
    """Constructs a Offset Generation module.
    """
    def __init__(self, channel, n_offsets):
        super(dca_offsets_layer, self).__init__()

        self.channel = channel
        self.n_offsets = n_offsets

        self.conv = nn.Conv2d(channel, n_offsets, kernel_size=1)
            
    def covariance_features(self, x):
        """
        Takes in a feature map and returns the unnormalized covariance matrix
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        return energy

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        cov_matrix = self.covariance_features(x).reshape(m_batchsize, C, 1, C)
        offsets = self.conv(cov_matrix).squeeze()
        return offsets
