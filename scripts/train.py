import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from backend.classification.model import MultiExitClassifier
from backend.training.losses import FocalLoss
from backend.training.trainer import DocumentTrainer

def main():
    print("🚀 Initializing Training Pipeline Scaffold...")
    
    # Synthetic dataset for showcase
    feats = torch.randn(100, 256)
    labels = torch.randint(0, 5, (100,))
    
    dataset = TensorDataset(feats, labels)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MultiExitClassifier(n_classes=5)
    
    # Alpha weights for focal loss
    alpha = torch.ones(5)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = DocumentTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader, # Dummy validation
        criterion=criterion,
        optimizer=optimizer,
        use_wandb=False
    )
    
    print("⏳ Running single epoch training pass...")
    loss = trainer.train_epoch()
    print(f"✅ Epoch completed. Train Loss: {loss:.4f}")
    
    stats = trainer.validate()
    print(f"✅ Validation Stats: {stats}")

if __name__ == "__main__":
    main()
