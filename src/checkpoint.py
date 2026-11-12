"""Yeni ve eski model kayıtları için ortak yükleme işlevi."""
import torch

from model import SmallCNN


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        config = checkpoint["model_config"]
    else:
        # Eski kayıtlar yalnızca state_dict içerir. Dropout tahminde etkisizdir.
        state = checkpoint
        config = {"use_bn": any(key.endswith("running_mean") for key in state),
                  "dropout": 0.0}
    model = SmallCNN(**config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
