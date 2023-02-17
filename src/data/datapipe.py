import os
import json

import torch
import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterableWrapper, Collator

from dgl import batch
from dgl.nn.pytorch.factory import RadiusGraph

from src.data.transforms import rotate_point_cloud

# descriptor order: label, molecular weight, hydrophobicity, pI
with open('src/data/descriptors.json', 'r') as fp:
    descriptors = json.load(fp)

    
# datapipe parses *one row* of the csvs at a time
# because of this we need to keep track of which pdb csv the row came from
def preprocess_row(row):
    # parsed rows are in the form: pdb_path, row_data
    pdb, row_data = row

    # one hot encode amino acid
    residue_id = row_data[6]
    residue_onehot = torch.zeros(20, dtype=torch.float32)
    residue_onehot[descriptors[residue_id][0]] = 1. # descriptors[residue_id][2]
        
    # residue coordinates
    coord = list(map(float, row_data[1:4]))
    coord = torch.tensor(coord, dtype=torch.float32)
    
    # PSSM
    pssm = list(map(float, row_data[9:29]))
    pssm = torch.tensor(pssm, dtype=torch.float32)
    
    # solvent accessibility
    solv_access = list(map(float, row_data[29:31]))
    solv_access = torch.tensor(solv_access, dtype=torch.float32)
    
    # neighbourhood
    nhood = list(map(float, row_data[31:51]))
    nhood = torch.tensor(nhood, dtype=torch.float32)
    
    # label
    label = float(row_data[-1])
    label = torch.tensor(label, dtype=torch.float32)
    
    return {
        "id": os.path.basename(pdb).split(".")[0],
        "coord": coord,
        "node_feat": torch.hstack((residue_onehot, pssm, solv_access, nhood)),
        "node_label": label,
    }        


# function to batch rows from same pdb together
def group_fn(preprocess_dict):
    return preprocess_dict["id"]


# batch rows belonging to same pdb & format into DGL graph object
def preprocess_group(group):
    # collect node coordinates
    coords = torch.vstack([d["coord"] for d in group])
    
    # collect node features
    node_feats = torch.vstack([d["node_feat"] for d in group])
    
    # collect labels
    node_labels = torch.vstack([d["node_label"] for d in group])
    
    # create graph
    radius_graph = RadiusGraph(8.)
    graph = radius_graph(coords)
    graph.ndata["coord"] = coords
    graph.ndata["feat"] = node_feats
    graph.ndata["label"] = node_labels
    
    return {
        "id": group[0]["id"],
        "graph": graph,
    }


# collate function to batch DGL graphs of different pdbs
def custom_collate(data): 
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {
            "id": [p["id"] for p in data[0]],
            "graph": batch([p["graph"] for p in data[0]]),
        }


def build_datapipe(root_dir, num_augment=0):
    # list and open csv files
    datapipe = dp.iter.FileLister(root_dir, recursive=True)
    datapipe = dp.iter.FileOpener(datapipe, mode='rt')
    
    # parse each row from all the csvs
    # keep track of which pdb each row came from with return_path=True
    datapipe = datapipe.parse_csv(delimiter=',', skip_lines=1, return_path=True)
    
    # preprocess each row
    datapipe = datapipe.map(preprocess_row)
    
    # batch (group) from same pdb together
    datapipe = datapipe.groupby(group_key_fn=group_fn, buffer_size=1e6)
    
    # process batch into required format
    datapipe = datapipe.map(preprocess_group)
    
    # order by pdb id
    datapipe = IterableWrapper(sorted(datapipe, key=lambda d: d["id"]))
    
    # repeat pdb for augmentation
    if num_augment > 0:
        datapipe = datapipe.repeat(num_augment + 1)
        datapipe = datapipe.map(rotate_point_cloud)
    
    # collate function for when batch size > 1
    datapipe = Collator(datapipe, collate_fn=custom_collate)
    
    return datapipe

