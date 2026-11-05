"""results/*.json dosyalarından karşılaştırma tablosu ve grafik üretir,
ve en iyi modelin confusion matrix'ini çizer.

    python src/evaluate.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import CLASSES, get_test_loader
from checkpoint import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deneyleri karşılaştırır ve grafik üretir.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Grafik dizini; varsayılan sonuç dizini")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--curves-only", action="store_true", help="Model veya veri olmadan tablo ve eğitim eğrileri")
    parser.add_argument("--name", help="Yalnızca belirtilen deneyi değerlendir")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers sıfır veya pozitif olmalı.")
    runs = []
    for path in sorted(args.results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            runs.append(json.load(stream))
    if args.name:
        runs = [run for run in runs if run["name"] == args.name]
    if not runs:
        raise SystemExit("Önce src/train.py ile en az bir deney çalıştır.")
    def selection_score(run):
        return run["best_val_acc"] if "best_val_acc" in run else run["best_acc"]

    mixed = len({"best_val_acc" in run for run in runs}) > 1
    if mixed and not args.curves_only:
        parser.error("Eski test seçimi ve yeni doğrulama sonuçları birlikte sıralanamaz. "
                     "--name ile bir deney veya ayrı --results-dir seçin.")
    runs.sort(key=selection_score)
    if any("best_val_acc" not in run for run in runs):
        print("Not: Eski deneyler test doğruluğuyla seçilmiştir; yeni doğrulama sonuçlarıyla doğrudan karşılaştırmayın.")

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) README'ye yapıştırılacak tablo
    print("| Deney | Augment | BatchNorm | Dropout | Seçim ölçütü | En iyi seçim doğruluğu | Son test |")
    print("|---|---|---|---|---|---|---|")
    for r in runs:
        a = r["args"]
        metric = "Doğrulama" if "best_val_acc" in r else "Test (eski)"
        final_test = f"{r['test_acc'] * 100:.2f}%" if "test_acc" in r else "—"
        print(f"| {r['name']} | {a['augment']} | {a['bn']} | {a['dropout']} | "
              f"{metric} | {selection_score(r)*100:.2f}% | {final_test} |")

    # 2) Eğri grafiği
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in runs:
        ax[0].plot(r["history"]["train_loss"], label=r["name"])
        metric = "val_acc" if "best_val_acc" in r else "test_acc"
        label = f"{r['name']} ({'val' if metric == 'val_acc' else 'test/eski'})"
        ax[1].plot([value * 100 for value in r["history"][metric]], label=label)
    ax[0].set(title="Eğitim kaybı", xlabel="Epoch", ylabel="Loss")
    ax[1].set(title="Doğruluk (eğri etiketindeki küme)", xlabel="Epoch", ylabel="%")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "curves.png", dpi=150)
    plt.close(fig)
    if args.curves_only:
        print(f"Kaydedildi: {output_dir / 'curves.png'}")
        return

    # 3) En iyi modelin confusion matrix'i
    best = runs[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.models_dir / f"{best['name']}.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"Model bulunamadı: {checkpoint}. Önce modeli eğitin veya --curves-only kullanın.")
    model = load_model(checkpoint, device)
    test_loader = get_test_loader(root=args.data_dir, num_workers=args.workers)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            labels += y.tolist()
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix – {best['name']}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\nKaydedildi: {output_dir / 'curves.png'}, {output_dir / 'confusion_matrix.png'} (en iyi: {best['name']})")


if __name__ == "__main__":
    main()
