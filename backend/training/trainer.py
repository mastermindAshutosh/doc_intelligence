import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class DocumentTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 criterion, optimizer, device="cpu", use_wandb=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_wandb = use_wandb

        if use_wandb:
            import wandb
            # wandb.init() should be called in outer script
            
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (feats, targets) in enumerate(self.train_loader):
            feats, targets = feats.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # MultiExitClassifier training_loss expects focal_loss_fn
            if hasattr(self.model, "training_loss"):
                loss = self.model.training_loss(feats, targets, self.criterion)
            else:
                logits = self.model(feats)
                loss = self.criterion(logits, targets)
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self) -> dict:
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for feats, targets in self.val_loader:
                feats, targets = feats.to(self.device), targets.to(self.device)
                
                if hasattr(self.model, "forward_inference"):
                    logits, _ = self.model.forward_inference(feats)
                else:
                    logits = self.model(feats)
                    
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
        acc = correct / total if total > 0 else 0.0
        return {"val_accuracy": acc}
