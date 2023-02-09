### PECAN ported to PyTorch

For default settings, run: 
```
python main.py
```

Common flags:
```
--model         choose one of "PECAN" or "PECAN_PN"
--epochs        number of train epochs
--hid_feats     hidden layer dimensions
```

For full list, run:
```
python main.py --help
```


**TODO:** 
- [ ] add pointnet augmentation (as in [original](https://github.com/charlesq34/pointnet/blob/master/provider.py))*
- [ ] add SE(3) transformer module (see [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer))
- [ ] add EGNN module (see [here](https://docs.dgl.ai/en/1.0.x/generated/dgl.nn.pytorch.conv.EGNNConv.html)). Neural message passing mechnism (specifically the aggregate function) *may* need to be modified if to be kept similar to PECAN.


