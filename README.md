## PECAN - PyTorch

A simple pytorch implementation of [PECAN](https://doi.org/10.1101/658054).
Environment is constructed with Docker.

To run an experiment, create a `.json` file with desired configuration paramteres in the `configs` folder. Then, run: 
```
python main.py -f <your config .json file>
```

---

The available configuration parameters are as follows:
```
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

### Citations

```bibtex

@article{10.1093/bioinformatics/btaa263,
    author = {Pittala, Srivamshi and Bailey-Kellogg, Chris},
    title = "{Learning context-aware structural representations to predict antigen and antibody binding interfaces}",
    journal = {Bioinformatics},
    volume = {36},
    number = {13},
    pages = {3996-4003},
    year = {2020},
    month = {04},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa263},
}
```


