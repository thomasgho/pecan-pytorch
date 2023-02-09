import torch 
from torch import nn
import dgl.function as fn
from dgl.utils import expand_as_pair


class GCNLayer(nn.Module):
    """
    Graph convolution module from PECAN.
    
    Follows operation used in: `Protein Interface Prediction using Graph Convolutional Networks 
    <https://proceedings.neurips.cc/paper/2017/file/f507783927f2ec2737ba40afbd17efb5-Paper.pdf>`__
    
    Mathematically it is defined as follows:
    m_i^{(l+1)} = \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}h_j^{(l)}W_u^{(l)})
    h_i^{(l+1)} = \sigma(b^{(l)} + h_i^{(l)}W_v^{(l)} + m_i^{(l+1)})
                 
                 
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    """
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        
        self.weight_u = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_v = nn.Parameter(torch.Tensor(in_feats, out_feats))
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

        torch.nn.init.kaiming_uniform_(self.weight_u)
        torch.nn.init.kaiming_uniform_(self.weight_v)
        torch.nn.init.zeros_(self.bias)
    

    def forward(self, graph, feat):
        """
        Forward computation

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat  : torch.Tensor
            The input node feature.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        
        with graph.local_scope():
            feat_u, feat_v = expand_as_pair(feat, graph)
            
            # centre node transform
            rst_v = torch.matmul(feat_v, self.weight_v)
            
            # neighbour node aggregation
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                feat_u = torch.matmul(feat_u, self.weight_u)
                graph.srcdata['h'] = feat_u
                graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
                rst_u = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_u
                graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
                rst_u = graph.dstdata['h']
                rst_u = torch.matmul(rst_u, self.weight_u)

            degs = graph.in_degrees().to(feat_v).clamp(min=1)
            norm = 1.0 / degs
            shape = norm.shape + (1,) * (feat_v.dim() - 1)
            norm = torch.reshape(norm, shape)

            # update
            rst_v = rst_v + (norm * rst_u) + self.bias
            rst_v = nn.functional.relu(rst_v)
            
            return rst_v
        

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout, n_layers):
        super(GCN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            self.layers.append(
                GCNLayer(
                    in_feats = in_feats if i == 0 else hid_feats, 
                    out_feats = hid_feats if i < (n_layers - 1) else out_feats
                )
            )
        
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, graph, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h