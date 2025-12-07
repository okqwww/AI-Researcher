import os
import json
import torch
from torch.utils.data import DataLoader
from data_processing.cifar10 import get_cifar10_dataloaders
from model.vqvae_rr import VQVAE_RR
from training.train import train_vqvae
from testing.test import evaluate_reconstructions

def main():
    # Configuration
    config = {
        "data_dir": os.path.join(os.path.dirname(__file__), "data"),
        "epochs": 3,
        "batch_size": 128,
        "num_workers": 2,
        "image_size": 32,
        "codebook_size": 1024,
        "embedding_dim": 64,
        "beta": 0.25,
        "ema_decay": 0.99,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "log_every": 100
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_cifar10_dataloaders(
        root=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"]
    )

    # Model
    model = VQVAE_RR(
        in_channels=3,
        hidden_channels=128,
        embedding_dim=config["embedding_dim"],
        codebook_size=config["codebook_size"],
        ema_decay=config["ema_decay"],
        beta=config["beta"]
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Train
    history = train_vqvae(model, train_loader, optimizer, device, epochs=config["epochs"], log_every=config["log_every"])

    # Evaluate
    results = evaluate_reconstructions(model, test_loader, device, stats_path=os.path.join("/workplace", "dataset_candidate", "cifar10-32x32.npz"))

    # Save logs
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Training history (last epoch):", {k: (v[-1] if isinstance(v, list) and len(v)>0 else v) for k,v in history.items()})
    print("Evaluation results:", results)

if __name__ == "__main__":
    main()
