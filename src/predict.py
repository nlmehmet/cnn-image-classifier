"""Bir görüntüyü kayıtlı CNN ile CIFAR-10 sınıflarından birine sınıflandırır."""
import argparse
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms

from checkpoint import load_model
from data import CLASSES, MEAN, STD


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Tahmin edilecek görüntü")
    parser.add_argument("--model", type=Path, required=True, help="Eğitilmiş .pt model kaydı")
    parser.add_argument("--top-k", type=int, default=3, help="Gösterilecek sınıf sayısı (1–10)")
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= len(CLASSES):
        parser.error("--top-k 1 ile 10 arasında olmalı.")
    for path in (args.image, args.model):
        if not path.is_file():
            parser.error(f"Dosya bulunamadı: {path}")
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    try:
        with Image.open(args.image) as image:
            tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
    except (OSError, UnidentifiedImageError) as error:
        parser.error(f"Görüntü okunamadı: {error}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_model(args.model, device)
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as error:
        parser.error(f"Model yüklenemedi: {error}")
    with torch.inference_mode():
        scores, indices = model(tensor.to(device)).softmax(dim=1)[0].topk(args.top_k)
    print("Sınıf | Model skoru (softmax)")
    for score, index in zip(scores.tolist(), indices.tolist()):
        print(f"{CLASSES[index]} | %{score * 100:.2f}")


if __name__ == "__main__":
    main()
