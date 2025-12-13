from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None

from slu.models.m1_cnn_transformer import CNNTransformerSLU
from slu.models.m2_transformer_tiny import TransformerTinySLU


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LogMelDataset(Dataset):
    def __init__(
        self,
        df,
        feature_key: str,
        product_map: dict,
        qty_map: dict,
        augment: bool = False,
        augment_cfg: dict | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_key = feature_key
        self.product_map = product_map
        self.qty_map = qty_map
        self.augment = augment
        cfg = augment_cfg or {}
        self.freq_mask_param = int(cfg.get("freq_mask_param", 0))
        self.time_mask_param = int(cfg.get("time_mask_param", 0))
        self.time_shift_max = int(cfg.get("time_shift_max", 0))
        self.noise_std = float(cfg.get("noise_std", 0.0))

    def __len__(self) -> int:
        return len(self.df)

    def _freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        # spec shape: (C, n_mels, T)
        max_width = self.freq_mask_param
        if max_width <= 0:
            return spec
        _, freq_bins, _ = spec.shape
        width = int(torch.randint(0, max_width + 1, (1,)).item())
        if width == 0 or width >= freq_bins:
            return spec
        start = int(torch.randint(0, freq_bins - width + 1, (1,)).item())
        spec[:, start : start + width, :] = 0.0
        return spec

    def _time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        # spec shape: (C, n_mels, T)
        max_width = self.time_mask_param
        if max_width <= 0:
            return spec
        _, _, time_steps = spec.shape
        width = int(torch.randint(0, max_width + 1, (1,)).item())
        if width == 0 or width >= time_steps:
            return spec
        start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
        spec[:, :, start : start + width] = 0.0
        return spec

    def _apply_augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: (1, n_mels, T)
        spec = tensor
        if torch.rand(1).item() < 0.8 and self.freq_mask_param > 0:
            spec = self._freq_mask(spec.clone())
        if torch.rand(1).item() < 0.8 and self.time_mask_param > 0:
            spec = self._time_mask(spec.clone())
        if torch.rand(1).item() < 0.5 and self.time_shift_max > 0:
            shift = int(torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item())
            spec = torch.roll(spec, shifts=shift, dims=-1)
        if torch.rand(1).item() < 0.5 and self.noise_std > 0.0:
            spec = spec + torch.randn_like(spec) * self.noise_std
        return spec

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_path = Path(row[self.feature_key])
        logmel = np.load(feat_path)
        tensor = torch.from_numpy(logmel).unsqueeze(0).float()  # 1, n_mels, T

        if self.augment:
            tensor = self._apply_augmentation(tensor)

        product = torch.tensor(self.product_map[str(row["product"])] if str(row["product"]) in self.product_map else self.product_map[row["product"]])
        qty = torch.tensor(self.qty_map[str(row["quantity"])] if str(row["quantity"]) in self.qty_map else self.qty_map[row["quantity"]])
        length = torch.tensor(tensor.shape[-1])
        return tensor, length, product, qty


def pad_collate(batch: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    tensors, lengths, products, qtys = zip(*batch)
    max_len = max(l.item() for l in lengths)
    padded = []
    for t in tensors:
        if t.shape[-1] < max_len:
            pad_width = (0, max_len - t.shape[-1])
            t = torch.nn.functional.pad(t, pad_width)
        padded.append(t)
    feats = torch.stack(padded, dim=0)
    lengths = torch.stack(lengths, dim=0)
    products = torch.stack(products, dim=0)
    qtys = torch.stack(qtys, dim=0)
    return feats, lengths, products, qtys


def build_label_maps(df, product_path: Path, qty_path: Path) -> tuple[dict, dict]:
    # build or load existing
    if product_path.exists() and qty_path.exists():
        product_map = json.loads(product_path.read_text())
        qty_map = json.loads(qty_path.read_text())
        return product_map, qty_map
    product_vals = sorted(df["product"].unique())
    qty_vals = sorted(df["quantity"].unique())
    product_map = {str(v): i for i, v in enumerate(product_vals)}
    qty_map = {str(v): i for i, v in enumerate(qty_vals)}
    product_path.parent.mkdir(parents=True, exist_ok=True)
    qty_path.parent.mkdir(parents=True, exist_ok=True)
    product_path.write_text(json.dumps(product_map, indent=2), encoding="utf-8")
    qty_path.write_text(json.dumps(qty_map, indent=2), encoding="utf-8")
    return product_map, qty_map


def compute_class_weights(df, column: str, label_map: dict) -> torch.Tensor:
    # Convert to string to align with label_map keys and avoid int casting failures
    counts = df[column].astype(str).value_counts().to_dict()
    weights = []
    for k, idx in label_map.items():
        count = counts.get(str(k), 0)
        weight = 1.0 / math.sqrt(count + 1e-6)
        weights.append(weight)
    w = torch.tensor(weights, dtype=torch.float)
    w = w / w.mean()
    return w


def get_model(model_cfg: dict, num_products: int, num_quantities: int) -> nn.Module:
    name = model_cfg.get("model_name", "cnn_transformer")
    if name == "cnn_transformer":
        return CNNTransformerSLU(
            num_products=num_products,
            num_quantities=num_quantities,
            d_model=model_cfg.get("transformer", {}).get("d_model", 256),
            nhead=model_cfg.get("transformer", {}).get("nhead", 4),
            num_layers=model_cfg.get("transformer", {}).get("num_layers", 2),
        )
    if name == "transformer_tiny":
        return TransformerTinySLU(
            num_products=num_products,
            num_quantities=num_quantities,
            d_model=model_cfg.get("transformer", {}).get("d_model", 128),
            nhead=model_cfg.get("transformer", {}).get("nhead", 4),
            num_layers=model_cfg.get("transformer", {}).get("num_layers", 2),
        )
    raise ValueError(f"Unknown model_name: {name}")


def train_one_epoch(model, loader, criterion_prod, criterion_qty, optimizer, device):
    model.train()
    total_loss = 0.0
    for feats, lengths, prod, qty in loader:
        feats, lengths, prod, qty = feats.to(device), lengths.to(device), prod.to(device), qty.to(device)
        optimizer.zero_grad()
        out = model(feats)
        loss_prod = criterion_prod(out["product"], prod)
        loss_qty = criterion_qty(out["quantity"], qty)
        loss = loss_prod + loss_qty
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion_prod, criterion_qty, device):
    model.eval()
    total_loss = 0.0
    correct_prod = 0
    correct_qty = 0
    total = 0
    for feats, lengths, prod, qty in loader:
        feats, lengths, prod, qty = feats.to(device), lengths.to(device), prod.to(device), qty.to(device)
        out = model(feats)
        loss_prod = criterion_prod(out["product"], prod)
        loss_qty = criterion_qty(out["quantity"], qty)
        loss = loss_prod + loss_qty
        total_loss += loss.item() * feats.size(0)
        pred_prod = out["product"].argmax(dim=1)
        pred_qty = out["quantity"].argmax(dim=1)
        correct_prod += (pred_prod == prod).sum().item()
        correct_qty += (pred_qty == qty).sum().item()
        total += feats.size(0)
    return {
        "loss": total_loss / total,
        "acc_product": correct_prod / total,
        "acc_qty": correct_qty / total,
    }


def split_train_val(df, val_ratio: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def main(
    data_cfg: str = "configs/data.yaml",
    preprocess_cfg: str = "configs/preprocess.yaml",
    model_cfg: str = "configs/model_m1.yaml",
    train_cfg: str = "configs/train.yaml",
    metadata_features: str = "data/processed/metadata_features.csv",
) -> None:
    data_conf = load_yaml(Path(data_cfg))
    preprocess_conf = load_yaml(Path(preprocess_cfg))
    model_conf = load_yaml(Path(model_cfg))
    train_conf = load_yaml(Path(train_cfg))

    device = torch.device("cuda" if torch.cuda.is_available() and train_conf.get("device", "auto") != "cpu" else "cpu")

    wandb_conf = train_conf.get("wandb", {})
    env_mode = os.getenv("WANDB_MODE", "").lower()
    env_disabled = os.getenv("WANDB_DISABLED", "").lower() == "true"
    env_key = os.getenv("WANDB_API_KEY")
    can_offline = env_mode in {"offline", "dryrun", "disabled"}
    use_wandb = bool(
        wandb_conf.get("enable", False)
        and wandb is not None
        and not env_disabled
        and (env_key or can_offline)
    )
    if wandb_conf.get("enable", False) and not use_wandb:
        print("[wandb] disabled: no API key and not in offline/dryrun mode; set WANDB_MODE=offline to enable")

    # load data
    df = torch.tensor([])
    import pandas as pd

    df = pd.read_csv(metadata_features)
    if "feature_path" not in df.columns:
        raise ValueError("metadata_features.csv harus memiliki kolom feature_path")

    product_map, qty_map = build_label_maps(
        df,
        Path(data_conf.get("label_maps", {}).get("product", "data/processed/product2id.json")),
        Path(data_conf.get("label_maps", {}).get("quantity", "data/processed/qty2id.json")),
    )

    train_df, val_df = split_train_val(df, val_ratio=0.1, seed=train_conf.get("seed", 42))

    augment_conf = train_conf.get("augment", {})
    train_ds = LogMelDataset(
        train_df,
        "feature_path",
        product_map,
        qty_map,
        augment=bool(augment_conf.get("enable", False)),
        augment_cfg=augment_conf,
    )
    val_ds = LogMelDataset(val_df, "feature_path", product_map, qty_map, augment=False)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=train_conf.get("batch_size", 16),
        shuffle=True,
        num_workers=train_conf.get("num_workers", 2),
        collate_fn=pad_collate,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_conf.get("batch_size", 16),
        shuffle=False,
        num_workers=train_conf.get("num_workers", 2),
        collate_fn=pad_collate,
        pin_memory=pin_mem,
    )

    model = get_model(model_conf, num_products=len(product_map), num_quantities=len(qty_map)).to(device)

    class_weighting = train_conf.get("class_weighting", True)
    prod_weights = compute_class_weights(df, "product", product_map) if class_weighting else None
    qty_weights = compute_class_weights(df, "quantity", qty_map) if class_weighting else None

    criterion_prod = nn.CrossEntropyLoss(weight=prod_weights.to(device) if prod_weights is not None else None)
    criterion_qty = nn.CrossEntropyLoss(weight=qty_weights.to(device) if qty_weights is not None else None)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_conf.get("learning_rate", 3e-4),
        weight_decay=train_conf.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_conf.get("max_epochs", 30))

    if use_wandb:
        wandb.init(
            project=wandb_conf.get("project", "slu-mlops"),
            name=wandb_conf.get("run_name"),
            config={
                "model_cfg": model_cfg,
                "train_cfg": train_cfg,
                "data_cfg": data_cfg,
                "preprocess_cfg": preprocess_cfg,
                "batch_size": train_conf.get("batch_size", 16),
                "learning_rate": train_conf.get("learning_rate", 3e-4),
                "weight_decay": train_conf.get("weight_decay", 1e-4),
                "class_weighting": train_conf.get("class_weighting", True),
                "max_epochs": train_conf.get("max_epochs", 30),
                "augment": train_conf.get("augment", {}),
            },
            mode=env_mode if env_mode else None,
            settings=wandb.Settings(start_method="thread"),
        )

    best_val = float("inf")
    patience = train_conf.get("early_stop_patience", 5)
    wait = 0
    checkpoint_dir = Path(train_conf.get("checkpoint_dir", "artifacts"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_conf.get("max_epochs", 30) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion_prod, criterion_qty, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion_prod, criterion_qty, device)
        scheduler.step()
        val_loss = val_stats["loss"]
        lr = optimizer.param_groups[0].get("lr", train_conf.get("learning_rate", 3e-4))
        log_msg = (
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"acc_prod={val_stats['acc_product']:.3f} | acc_qty={val_stats['acc_qty']:.3f} | lr={lr:.6f}"
        )
        print(log_msg)
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "acc_product": val_stats["acc_product"],
                    "acc_quantity": val_stats["acc_qty"],
                    "lr": lr,
                }
            )
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            if use_wandb:
                wandb.summary["best_val_loss"] = best_val
                wandb.summary["best_acc_product"] = val_stats["acc_product"]
                wandb.summary["best_acc_quantity"] = val_stats["acc_qty"]
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
