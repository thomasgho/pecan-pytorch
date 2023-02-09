import torch 
from torch import nn

from src.modules.pointnet import PointNet
from src.modules.gcn import GCN
from src.modules.attention import AttentionLayer
from src.modules.fcn import FCLayer


class PECAN_PN(torch.nn.Module):
    """
    PECAN model with 2xGCN + 1xPointNet + 1xAttention.
    
    The model is created as in the
    `original implementation <https://github.com/vamships/PECAN>`__
    with the addition of a coordinate processing module.
                 
    Parameters
    ----------
    in_feats : int
        Input feature size.
    hid_feats : int
        Hidden feature size.
    out_feats : int
        Output feature size.
    dropout  : float
        Node dropout for GCN.
    """
    
    def __init__(self, in_feats, hid_feats, out_feats, dropout):
        super(PECAN_PN, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        
        self.pn_p = PointNet(
            in_feats=3,
            hid_feats=hid_feats,
            out_feats=hid_feats, 
            dropout=dropout)
        
        self.gcn_p = GCN(
            in_feats=in_feats, 
            hid_feats=hid_feats, 
            out_feats=hid_feats, 
            dropout=dropout,
            n_layers=2)
        
        self.gcn_s = GCN(
            in_feats=in_feats, 
            hid_feats=hid_feats, 
            out_feats=hid_feats, 
            dropout=dropout,
            n_layers=2)
        
        self.attention = AttentionLayer(
            in_feats=hid_feats,
            out_feats=hid_feats)
        
        self.fcn = FCLayer(
            in_feats=hid_feats,
            out_feats=1)
    
    
    def loss(self, pred, target, weight):
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        return criterion(pred, target)
    
    
    def forward(self, graph_p, graph_s):
        geom_p = self.pn_p(graph_p.ndata["coord"])
        feat_p = self.gcn_p(graph_p, graph_p.ndata['feat'])
        feat_s = self.gcn_p(graph_s, graph_s.ndata['feat'])
        c = self.attention(feat_p, feat_s)
        out = self.fcn(c, feat_p, geom_p)
        return out
