import torch 
from torch import nn


class AttentionLayer(nn.Module):
    """
    Attention module from PECAN.
    
    Dot product attention as in: `Effective Approaches to Attention-based Neural Machine Translation
    <https://arxiv.org/pdf/1508.04025.pdf>`__
    
    Mathematically it is defined as follows:
    h_i = W_a p'_i
    h_j = W_a s'_j
    a_{ij} = \sigma(h_i^T h_j)
    c_i = \sum_{j=1}^M s'_j \frac{a_{ij}}{\sqrt{\sum_{i,j=1}^{i=N,j=M}a_{ij}^2}}
                 
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    """
    def __init__(self, in_feats, out_feats):
        super(AttentionLayer, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        
        self.weight_a = nn.Parameter(torch.Tensor(in_feats, out_feats))
        
        self.reset_parameters()
        
    
    def reset_parameters(self):
        r"""
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/vamships/PECAN>`__
        where the weight :math:`W^{(l)}` is initialized using He initialization
        and the bias is initialized to be zero.

        """

        torch.nn.init.kaiming_uniform_(self.weight_a)
    
    
    def forward(self, feat_p, feat_s):
        """
        Forward computation

        Parameters
        ----------
        feat_p  : torch.Tensor
            The primary input node feature.
        feat_s  : torch.Tensor
            The secondary input node feature.
        
        Returns
        -------
        torch.Tensor
            The output feature
        """
        
        h_i = torch.matmul(feat_p, self.weight_a)
        h_j = torch.matmul(feat_s, self.weight_a)
        sig = torch.matmul(h_i, h_j.T)  # mistake in paper!

        a_ij = nn.functional.relu(sig)
        α_ij = a_ij / torch.linalg.matrix_norm(a_ij)

        c_i = torch.matmul(α_ij, feat_s)
        return c_i