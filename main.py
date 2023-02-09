import argparse
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
        dp_ag = build_datapipe("data/ag/" + dataset)
        dp_ab = build_datapipe("data/ab/" + dataset)
        
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
        log_dir = args.log_dir,
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
        torch.save(model.state_dict(), args.log_dir)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="choose one of \"PECAN\" or \"PECAN_PN\"", type=str, default="PECAN_PN")
    parser.add_argument(
        "--in_feats", help="input feature dimension", type=int, default=62)
    parser.add_argument(
        "--hid_feats", help="hidden layers feature dimension", type=int, default=32)
    parser.add_argument(
        "--dropout", help="hidden layer dropout probability", type=float, default=0.5)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument(
        "--weight_decay", help="L2 regularization", type=float, default=0.)
    parser.add_argument(
        "--momentum", help="Nesterov momentum factor", type=float, default=0.9)
    parser.add_argument(
        "--batch_size", help="train batch size", type=int, default=1)
    parser.add_argument(
        "--accumulation_steps", help="number of gradient accumilation steps", type=int, default=1)
    parser.add_argument(
        "--cuda", help="use CUDA acceleration", type=bool, default=True)
    parser.add_argument(
        "--mixed_precision", help="train with 16-bit and 32-bit floats to reduce memory overhead", type=bool, default=False)
    parser.add_argument(
        "--epochs", help="number of train epochs", type=int, default=100)
    parser.add_argument(
        "--log_dir", help="path to log tensorboard results", type=str, default="runs/exp1")
    parser.add_argument(
        "--save_model", help="whether to save model weights", type=bool, default=False)
    args = parser.parse_args()

    train(args)