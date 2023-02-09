import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class PointNet(nn.Module):
    """
    PointNet module.
    
    Original paper: `PointNet: Deep Learning on Point Sets for 3D 
    Classification and Segmentation <https://arxiv.org/abs/1612.00593>`__
    
    Implementation inspiration: `DGL 
    <https://github.com/dmlc/dgl/blob/master/examples/pytorch/pointcloud/pointnet/pointnet_cls.py>`__
    
    Parameters
    ----------
    in_feats : int
        Input feature size.
    hid_feats : int
        Hidden feature size.        
    out_feats : int
        Output feature size.
    dropout : bool
        Dropout probibility
    single_sample: bool
        If batch size is 1. Applies instance norm instead of batch norm.
    """
    
    def __init__(self, in_feats=3, hid_feats=64, out_feats=20, dropout=0.5):
        super(PointNet, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        
        self.trans1 = TransformNet(in_feats)
        self.trans1_bn = nn.BatchNorm1d(in_feats)
        
        self.conv1 = nn.Conv1d(in_feats, hid_feats, 1)
        self.conv1_bn = nn.BatchNorm1d(hid_feats)
        
        self.trans2 = TransformNet(hid_feats)
        self.trans2_bn = nn.BatchNorm1d(hid_feats)

        self.conv2 = nn.Conv1d(hid_feats, hid_feats * 2, 1)
        self.conv2_bn = nn.BatchNorm1d(hid_feats * 2)
        
        self.conv3 = nn.Conv1d(hid_feats * 2, hid_feats * 16, 1)
        self.conv3_bn = nn.BatchNorm1d(hid_feats * 16)
        
        self.conv4 = nn.ModuleList()
        self.conv4.append(nn.Conv1d(hid_feats * 20, hid_feats * 8, 1))
        self.conv4.append(nn.Conv1d(hid_feats * 8, hid_feats * 4, 1))
        self.conv4.append(nn.Conv1d(hid_feats * 4, hid_feats * 2, 1))
        
        self.conv4_bn = nn.ModuleList()
        self.conv4_bn.append(nn.BatchNorm1d(hid_feats * 8))
        self.conv4_bn.append(nn.BatchNorm1d(hid_feats * 4))
        self.conv4_bn.append(nn.BatchNorm1d(hid_feats * 2))

        self.conv5 = nn.Conv1d(hid_feats * 2, out_feats, 1)


    def forward(self, h):        
        npoints = h.shape[0]
        
        h = h.transpose(1, 0)
        h = h.unsqueeze(0)
        
        trans = self.trans1(h)
        h = h.transpose(2, 1)
        h = torch.bmm(h, trans)
        h = h.transpose(2, 1)
        h = nn.functional.relu(self.trans1_bn(h))

        mid_feat = []
        
        h = self.conv1(h)
        h = self.conv1_bn(h)
        h = nn.functional.relu(h)
        mid_feat.append(h)
            
        trans = self.trans2(h)
        h = h.transpose(2, 1)
        h = torch.bmm(h, trans)
        h = h.transpose(2, 1)
        h = nn.functional.relu(self.trans2_bn(h))
        mid_feat.append(h)

        h = self.conv2(h)
        h = self.conv2_bn(h)
        h = nn.functional.relu(h)
        mid_feat.append(h)
        
        h = self.conv3(h)
        h = self.conv3_bn(h)
        h = torch.max(h, 2, keepdim=True)[0]
        h = h.view(1, -1, 1)
        h = h.repeat(1, 1, npoints)
        mid_feat.append(h)
        
        h = torch.cat(mid_feat, 1)
        for conv, bn in zip(self.conv4, self.conv4_bn):
            h = conv(h)
            h = bn(h)
            h = nn.functional.relu(h)

        out = self.conv5(h)
        
        out = out.squeeze(0)
        out = out.transpose(1, 0)
        
        return out


class TransformNet(nn.Module):
    def __init__(self, in_feats=3, hid_feats=64):
        super(TransformNet, self).__init__()
        
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv1d(in_feats, hid_feats, 1))
        self.conv.append(nn.Conv1d(hid_feats, hid_feats * 2, 1))
        self.conv.append(nn.Conv1d(hid_feats * 2, hid_feats * 16, 1))

        self.conv_bn = nn.ModuleList()
        self.conv_bn.append(nn.BatchNorm1d(hid_feats))
        self.conv_bn.append(nn.BatchNorm1d(hid_feats * 2))
        self.conv_bn.append(nn.BatchNorm1d(hid_feats * 16))

        self.pool_feat_len = hid_feats * 16

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(hid_feats * 16, hid_feats * 8))
        self.mlp.append(nn.Linear(hid_feats * 8, hid_feats * 4))
        
        self.mlp_bn = nn.ModuleList()
        self.mlp_bn.append(nn.InstanceNorm1d(hid_feats * 8))
        self.mlp_bn.append(nn.InstanceNorm1d(hid_feats * 4))

        self.in_feats = in_feats
        self.mlp_out = nn.Linear(hid_feats * 4, in_feats * in_feats)

        
    def forward(self, h):
        for conv, bn in zip(self.conv, self.conv_bn):
            h = conv(h)
            h = bn(h)
            h = nn.functional.relu(h)

        h = torch.max(h, 2, keepdim=True)[0]
        h = h.view(-1, self.pool_feat_len)
        
        for mlp, bn in zip(self.mlp, self.mlp_bn):
            h = mlp(h)
            h = bn(h)
            h = nn.functional.relu(h)

        out = self.mlp_out(h)

        iden = torch.eye(self.in_feats, requires_grad=True).flatten()
        iden = iden.view(1, self.in_feats * self.in_feats)
        if out.is_cuda:
            iden = iden.cuda()
        out = out + iden
        out = out.view(-1, self.in_feats, self.in_feats)
        return out