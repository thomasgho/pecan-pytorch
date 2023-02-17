### PECAN ported to PyTorch

To run an experiment, create a `.json` file with desired configuration paramteres in the `configs` folder. Then, run: 
```
python main.py -f <your config .json file>
```

---

The available configuration parameters are as follows:
```
"model"                 str           Choose one of "PECAN" or "PECAN_PN"
"in_feats"              int           Input data feature size
"hid_feats"             int           Hidden feature size
"dropout"               float         Dropout probability for GCN module
"lr"                    float         SGD learning rate
"weight_decay"          float         L2 regularizarion factor
"momentum"              float         Nesterov momentum factor
"batch_size"            int           Training batch size
"accumulation_steps"    int           Gradient accumulation steps (=1 means no accumulation)
"num_augment"           int           Number of random 3D rotation augmentations for protein coordinates
"epochs"                int           Number of model training epochs
"cuda"                  bool          Whether to use CUDA acceleration
"mixed_precision"       bool          Whether to use mixed precision floating points to reduce memory overhead
"save_model"            bool          Whether to save model weights
```

---

**TODO:** 
- [x] add pointnet augmentation (as in [original](https://github.com/charlesq34/pointnet/blob/master/provider.py))
- [ ] add SE(3) transformer module (see [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer))
- [ ] add EGNN module (see [here](https://docs.dgl.ai/en/1.0.x/generated/dgl.nn.pytorch.conv.EGNNConv.html)). Neural message passing mechnism (specifically the aggregate function) *may* need to be modified if to be kept similar to PECAN.


