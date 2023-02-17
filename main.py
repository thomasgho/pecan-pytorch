import argparse
import json
import os

import torch
from torchdata.dataloader2 import DataLoader2
from torchmetrics.classification import BinaryAveragePrecision

from src.data.datapipe import build_datapipe
from src.models.baseline import PECAN
from src.trainer import Trainer

os.environ['TENSORBOARD_BINARY'] = '/data/conda/bin/tensorboard'

    
def train(args):
    
    # load data
    for dataset in ["train", "test"]:
        dp_ag = build_datapipe("data/ag/" + dataset, args.num_augment)
        dp_ab = build_datapipe("data/ab/" + dataset, args.num_augment)
        
        # to iterate over two dataloaders simultaneously
        dp_zip = dp_ag.zip(dp_ab) 

        # prepare dataloaders
        if dataset == "train":
            dp_zip = dp_zip.shuffle()
            dp_zip = dp_zip.batch(batch_size=args.batch_size)
            train_loader = DataLoader2(datapipe=dp_zip)
        elif dataset == "test":
            test_loader = DataLoader2(datapipe=dp_zip)

        # free memory
        del dp_ag, dp_ab
        
    # model instance
    if args.model == "PECAN_PN":
        model = PECAN(
            in_feats=args.in_feats,
            hid_feats=args.hid_feats,
            out_feats=1,
            dropout=args.dropout,
        )
    elif args.model == "PECAN":
        model = PECAN(
            in_feats=args.in_feats,
            hid_feats=args.hid_feats,
            out_feats=1,
            dropout=args.dropout,
        )
    else: 
        raise ValueError("Please choose one of \"PECAN\" or \"PECAN_PN\" for model")

    # optimizer instance
    # optimizer chosen by original PECAN authors
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True,
    )

    # metric instance
    metric = BinaryAveragePrecision()

    # train routine instance
    # batch size chosen here for memory reasons
    # (sequential batch feed)
    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        metric = metric,
        accumulation_steps = args.accumulation_steps,
        log_dir = "runs/" + args.exp_name,
        cuda = args.cuda,
        mixed_precision = args.mixed_precision,
    )

    # run train routine
    trainer.fit(
        train_loader,
        test_loader,
        args.epochs,
    )
    
    # save model
    if args.save_model:
        torch.save(model.state_dict(), "runs/" + args.exp_name)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--exp_name", help="configuration json file name", type=str, default="exp1")
    args = parser.parse_args()
    
    with open("configs/" + args.exp_name + ".json", "r") as fp:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(fp))
        args = parser.parse_args(namespace=t_args)

    train(args)
