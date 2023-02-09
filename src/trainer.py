import dgl
import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Train routine helper.
    """
    
    def __init__(self, model, optimizer, metric, accumulation_steps, 
                 log_dir, cuda=True, mixed_precision=False):
        super().__init__()
        # cuda
        if cuda:
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        else:
            self.device = torch.device("cpu")
        
        # training instances
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.metric = metric
        
        # loss weighting
        self.pn_ratio = torch.tensor([(100.-8.9)/8.9], device=self.device)        
        
        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        
        # tensorboard
        self.log_dir = log_dir
        
        # automatic mixed precision training
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler(enabled=self.mixed_precision)

    
    def fit(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            writer = SummaryWriter(log_dir=self.log_dir)
            
            train_loss, train_metric = self._train(train_loader)
            test_loss, test_metric = self._test(test_loader)
            
            # print updates
            print("Epoch {}: \n \
            -> Train Loss = {} \n \
            -> Test Loss = {} \n \
            -> Train Metric = {} \n \
            -> Test Metric = {}".format(
                epoch,
                train_loss,
                test_loss,
                train_metric,
                test_metric))
            
            # update tensorboard
            writer.add_scalars("Loss", {"Train": train_loss,
                                        "Test": test_loss}, epoch)
            writer.add_scalars("Metric", {"Train": train_metric,
                                        "Test": test_metric}, epoch)
        
        # write pending to disk
        writer.flush()
        writer.close()

            
    def _train(self, loader):
        self.model.train()
        
        batch_losses = []
        batch_metrics = []
        for batch, batch_data in enumerate(loader):
            
            # unzip ag ab data
            batch_ag = [i[0] for i in batch_data]
            batch_ab = [i[1] for i in batch_data]
            
            # check synchronisation
            assert [ag["id"] for ag in batch_ag] == [ab["id"] for ab in batch_ab], (
                "Clear .ipynb_checkpoints cache.")
            
            # merge batch into one dgl graph + send to cuda if available
            batch_ag = dgl.batch([ag["graph"] for ag in batch_ag]).to(self.device)            
            batch_ab = dgl.batch([ab["graph"] for ab in batch_ab]).to(self.device)

            # forward propagation (mixed precision)
            with torch.autocast(device_type=str(self.device), 
                                dtype=torch.float16, 
                                enabled=self.mixed_precision):
                
                pred_label = self.model(batch_ag, batch_ab)
                loss = self.model.loss(pred_label, batch_ag.ndata["label"], self.pn_ratio)
            
            # backward pass on scaled gradients
            self.scaler.scale(loss).backward()
            
            # evaluate metric
            metric = self.metric(pred_label, batch_ag.ndata["label"])
            
            # update weights after accumulation_steps iterations
            # effective batch size is increased 
            if (batch + 1) % self.accumulation_steps == 0 or (batch + 1) == len(loader):
                # update and reset (unscaled) gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
                # append to list
                batch_losses.append(loss.item())
                batch_metrics.append(metric.item())
        
        # compute average over batches
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_metric = sum(batch_metrics) / len(batch_metrics)
        
        return avg_loss, avg_metric

    
    def _test(self, loader):
        self.model.eval()

        with torch.no_grad():
            batch_losses = []
            batch_metrics = [] 
            for ag, ab in loader:
                
                # check synchronisation
                assert ag["id"] == ab["id"], (
                    "Clear .ipynb_checkpoints cache.")

                # send to cuda if available
                for p in [ag, ab]:
                    p["graph"] = p["graph"].to(self.device)
                
                # forward propagation
                pred_label = self.model(ag["graph"], ab["graph"])
                loss = self.model.loss(pred_label, ag["graph"].ndata["label"], self.pn_ratio)
                
                # evaluate metric
                metric = self.metric(pred_label, ag["graph"].ndata["label"])
                
                # append to list
                batch_losses.append(loss.item())
                batch_metrics.append(metric.item())
                
            # compute average over batches
            avg_loss = sum(batch_losses) / len(batch_losses)
            avg_metric = sum(batch_metrics) / len(batch_metrics)
            
        return avg_loss, avg_metric
        
        
