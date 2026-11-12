"""Tek bir deneyi eğitir ve sonuçları results/<name>.json içine yazar.

Örnek:
    python src/train.py --name baseline
    python src/train.py --name aug --augment
    python src/train.py --name aug_bn_drop --augment --bn --dropout 0.3
"""
import argparse
import json
import re
from pathlib import Path
import time

import torch
import torch.nn as nn

from data import get_training_loaders, get_test_loader
from checkpoint import load_model
from model import SmallCNN


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CIFAR-10 üzerinde tek bir CNN deneyi eğitir.")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--bn", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--data-dir", default="data", help="CIFAR-10 veri dizini")
    p.add_argument("--results-dir", default="results", help="JSON sonuç dizini")
    p.add_argument("--models-dir", default="models", help="Model kayıt dizini")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2, help="Veri yükleme işçisi; 0 ana süreç")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Eğitim verisinden doğrulamaya ayrılan oran; varsayılan 0.1")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        p.error("--name yalnızca harf, rakam, alt çizgi ve tire içerebilir.")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        p.error("--epochs ve --batch-size pozitif, --workers sıfır veya pozitif olmalı.")
    if not 0 < args.lr < float("inf") or not 0 <= args.dropout < 1:
        p.error("--lr pozitif ve sonlu, --dropout [0, 1) aralığında olmalı.")
    if not 0 < args.val_fraction < 1:
        p.error("--val-fraction (0, 1) aralığında olmalı.")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    train_loader, val_loader = get_training_loaders(
        augment=args.augment, batch_size=args.batch_size, root=args.data_dir,
        num_workers=args.workers, val_fraction=args.val_fraction, seed=args.seed)
    model = SmallCNN(use_bn=args.bn, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    results_dir, models_dir = Path(args.results_dir), Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_acc": []}
    checkpoint_path = models_dir / f"{args.name}.pt"
    best_acc, start = -1.0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        scheduler.step()

        train_loss = running / len(train_loader.dataset)
        acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)
        print(f"Epoch {epoch:02d}/{args.epochs}  loss={train_loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(),
                        "model_config": {"use_bn": args.bn, "dropout": args.dropout},
                        "epoch": epoch, "best_val_acc": best_acc}, checkpoint_path)

    # Model seçimi tamamlandıktan sonra test kümesini yalnızca bir kez kullan.
    best_model = load_model(checkpoint_path, device)
    test_loader = get_test_loader(root=args.data_dir, num_workers=args.workers)
    test_acc = evaluate(best_model, test_loader, device)
    result = {"name": args.name, "args": vars(args), "best_val_acc": best_acc,
              "test_acc": test_acc, "selection_metric": "val_acc",
              "seconds": round(time.time() - start), "history": history}
    with open(results_dir / f"{args.name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Bitti. En iyi doğrulama: {best_acc:.4f}; son test: {test_acc:.4f}")


if __name__ == "__main__":
    main()
