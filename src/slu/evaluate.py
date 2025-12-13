from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from slu.train import (
    LogMelDataset,
    pad_collate,
    split_train_val,
    get_model,
    load_yaml,
    compute_class_weights,
)


def load_label_map(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    data = json.loads(path.read_text())
    return {str(k): int(v) for k, v in data.items()}


def invert_label_map(label_map: dict) -> List[str]:
    inverse = {v: k for k, v in label_map.items()}
    return [inverse[i] for i in range(len(inverse))]


def plot_confusion(cm: np.ndarray, labels: List[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_model(
    model,
    loader,
    criterion_prod,
    criterion_qty,
    device,
) -> Tuple[dict, list, list, list, list]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct_prod = 0
    correct_qty = 0
    y_true_prod: list[int] = []
    y_pred_prod: list[int] = []
    y_true_qty: list[int] = []
    y_pred_qty: list[int] = []

    with torch.no_grad():
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
            y_true_prod.extend(prod.cpu().tolist())
            y_pred_prod.extend(pred_prod.cpu().tolist())
            y_true_qty.extend(qty.cpu().tolist())
            y_pred_qty.extend(pred_qty.cpu().tolist())

    stats = {
        "loss": total_loss / total if total else 0.0,
        "acc_product": correct_prod / total if total else 0.0,
        "acc_quantity": correct_qty / total if total else 0.0,
    }
    return stats, y_true_prod, y_pred_prod, y_true_qty, y_pred_qty


def run_evaluation(
    registry: str = "artifacts/registry/latest.json",
    metadata_features: str = "data/processed/metadata_features.csv",
    data_cfg: str = "configs/data.yaml",
    preprocess_cfg: str = "configs/preprocess.yaml",
    model_cfg: str | None = None,
    train_cfg: str = "configs/train.yaml",
    metrics_path: str = "artifacts/metrics.json",
    confusion_product: str = "artifacts/confusion_product.png",
    confusion_quantity: str = "artifacts/confusion_qty.png",
    report_path: str = "artifacts/classification_report.txt",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    registry_path = Path(registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = json.loads(registry_path.read_text()) if registry_path.exists() else {}

    model_cfg_path = Path(model_cfg or registry_data.get("config_path", "configs/model_m1.yaml"))
    data_cfg_path = Path(data_cfg)
    preprocess_cfg_path = Path(preprocess_cfg)
    train_cfg_path = Path(train_cfg)

    model_path = Path(registry_data.get("model_path", "artifacts/best_model.pt"))
    product_map_path = Path(registry_data.get("label_map_product", "data/processed/product2id.json"))
    qty_map_path = Path(registry_data.get("label_map_quantity", "data/processed/qty2id.json"))

    data_conf = load_yaml(data_cfg_path)
    preprocess_conf = load_yaml(preprocess_cfg_path)
    train_conf = load_yaml(train_cfg_path)
    model_conf = load_yaml(model_cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() and train_conf.get("device", "auto") != "cpu" else "cpu")

    df = pd.read_csv(metadata_features)
    if "feature_path" not in df.columns:
        raise ValueError("metadata_features.csv harus memiliki kolom feature_path")

    product_map = load_label_map(product_map_path)
    qty_map = load_label_map(qty_map_path)

    _, val_df = split_train_val(df, val_ratio=val_ratio, seed=seed)

    val_ds = LogMelDataset(val_df, "feature_path", product_map, qty_map)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_conf.get("batch_size", 16),
        shuffle=False,
        num_workers=train_conf.get("num_workers", 2),
        collate_fn=pad_collate,
        pin_memory=True,
    )

    model = get_model(model_conf, num_products=len(product_map), num_quantities=len(qty_map)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    class_weighting = train_conf.get("class_weighting", True)
    prod_weights = compute_class_weights(df, "product", product_map) if class_weighting else None
    qty_weights = compute_class_weights(df, "quantity", qty_map) if class_weighting else None

    criterion_prod = torch.nn.CrossEntropyLoss(weight=prod_weights.to(device) if prod_weights is not None else None)
    criterion_qty = torch.nn.CrossEntropyLoss(weight=qty_weights.to(device) if qty_weights is not None else None)

    stats, y_true_prod, y_pred_prod, y_true_qty, y_pred_qty = evaluate_model(
        model, val_loader, criterion_prod, criterion_qty, device
    )

    labels_prod = invert_label_map(product_map)
    labels_qty = invert_label_map(qty_map)

    cm_prod = confusion_matrix(y_true_prod, y_pred_prod, labels=list(range(len(labels_prod))))
    cm_qty = confusion_matrix(y_true_qty, y_pred_qty, labels=list(range(len(labels_qty))))

    plot_confusion(cm_prod, labels_prod, Path(confusion_product), "Confusion Matrix - Product")
    plot_confusion(cm_qty, labels_qty, Path(confusion_quantity), "Confusion Matrix - Quantity")

    f1_prod = f1_score(y_true_prod, y_pred_prod, average="macro") if y_true_prod else 0.0
    f1_qty = f1_score(y_true_qty, y_pred_qty, average="macro") if y_true_qty else 0.0

    metrics = {
        "val_loss": stats["loss"],
        "acc_product": stats["acc_product"],
        "acc_quantity": stats["acc_quantity"],
        "f1_product": f1_prod,
        "f1_quantity": f1_qty,
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_prod = classification_report(y_true_prod, y_pred_prod, target_names=labels_prod, zero_division=0)
    report_qty = classification_report(y_true_qty, y_pred_qty, target_names=labels_qty, zero_division=0)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "Product head\n" + report_prod + "\n\nQuantity head\n" + report_qty + "\n", encoding="utf-8"
    )

    registry_payload = {
        "model_path": str(model_path.as_posix()),
        "config_path": str(model_cfg_path.as_posix()),
        "data_cfg": str(data_cfg_path.as_posix()),
        "preprocess_cfg": str(preprocess_cfg_path.as_posix()),
        "train_cfg": str(train_cfg_path.as_posix()),
        "label_map_product": str(product_map_path.as_posix()),
        "label_map_quantity": str(qty_map_path.as_posix()),
        "metrics": metrics,
        "artifacts": {
            "metrics": str(Path(metrics_path).as_posix()),
            "confusion_product": str(Path(confusion_product).as_posix()),
            "confusion_quantity": str(Path(confusion_quantity).as_posix()),
            "classification_report": str(report_path.as_posix()),
        },
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


def main(cli_args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="artifacts/registry/latest.json")
    parser.add_argument("--metadata_features", default="data/processed/metadata_features.csv")
    parser.add_argument("--data_cfg", default="configs/data.yaml")
    parser.add_argument("--preprocess_cfg", default="configs/preprocess.yaml")
    parser.add_argument("--model_cfg", default=None)
    parser.add_argument("--train_cfg", default="configs/train.yaml")
    parser.add_argument("--metrics_path", default="artifacts/metrics.json")
    parser.add_argument("--confusion_product", default="artifacts/confusion_product.png")
    parser.add_argument("--confusion_quantity", default="artifacts/confusion_qty.png")
    parser.add_argument("--report_path", default="artifacts/classification_report.txt")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(cli_args)

    run_evaluation(
        registry=args.registry,
        metadata_features=args.metadata_features,
        data_cfg=args.data_cfg,
        preprocess_cfg=args.preprocess_cfg,
        model_cfg=args.model_cfg,
        train_cfg=args.train_cfg,
        metrics_path=args.metrics_path,
        confusion_product=args.confusion_product,
        confusion_quantity=args.confusion_quantity,
        report_path=args.report_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
