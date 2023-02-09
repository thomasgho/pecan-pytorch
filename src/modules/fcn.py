import torch 
from torch import nn


class FCLayer(nn.Module):
    """
    Fully connected module from PECAN.
    
    Concactenate + linear for binary classification.
    
    Mathematically it is defined as follows:
    y_i = W^T(c_i||p'_i) + b
                 
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    third : bool
        Option to concactenate a third tensor
    """
    
    def __init__(self, in_feats, out_feats, third=False):
        super(FCLayer, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        
        self.weight_c = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_p = nn.Parameter(torch.Tensor(in_feats, out_feats))
        
        if third:
            self.weight_g = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight_g", None)
            
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        
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

        torch.nn.init.kaiming_uniform_(self.weight_c)
        torch.nn.init.kaiming_uniform_(self.weight_p)
        
        if self.weight_g is not None:
            torch.nn.init.kaiming_uniform_(self.weight_g)
        
        torch.nn.init.zeros_(self.bias)
    
    
    def forward(self, feat_c, feat_p, feat_g=None):
        """
        Forward computation

        Parameters
        ----------
        feat_c  : torch.Tensor
            The context node feature.
        feat_p  : torch.Tensor
            The primary node feature.
        feat_g  : torch.Tensor
            Optional. The primary node geometric feature.
            
        Returns
        -------
        torch.Tensor
            The output feature
        """
        out_c = torch.matmul(feat_c, self.weight_c)
        out_p = torch.matmul(feat_p, self.weight_p)
        
        if self.weight_g is not None and feat_g is not None:
            out_g = torch.matmul(feat_g, self.weight_g)
            return out_c + out_p + out_g + self.bias
        
        return out_c + out_p + self.bias