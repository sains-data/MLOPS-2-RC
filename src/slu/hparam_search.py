from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd
import torch

from slu.train import (
    LogMelDataset,
    pad_collate,
    build_label_maps,
    compute_class_weights,
    get_model,
    load_yaml,
    train_one_epoch,
    evaluate,
    split_train_val,
)


def run_trial(trial, cfg_paths: dict) -> float:
    data_cfg = load_yaml(cfg_paths["data_cfg"])
    model_cfg = load_yaml(cfg_paths["model_cfg"])
    train_cfg = load_yaml(cfg_paths["train_cfg"])

    # Hyperparameters to tune
    lr = trial.suggest_loguniform("learning_rate", 5e-5, 5e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24])

    df = pd.read_csv(cfg_paths["metadata_features"])
    if "feature_path" not in df.columns:
        raise ValueError("metadata_features.csv harus memiliki kolom feature_path")

    product_map, qty_map = build_label_maps(
        df,
        Path(data_cfg.get("label_maps", {}).get("product", "data/processed/product2id.json")),
        Path(data_cfg.get("label_maps", {}).get("quantity", "data/processed/qty2id.json")),
    )

    train_df, val_df = split_train_val(df, val_ratio=train_cfg.get("val_ratio", 0.1), seed=train_cfg.get("seed", 42))

    train_ds = LogMelDataset(train_df, "feature_path", product_map, qty_map)
    val_ds = LogMelDataset(val_df, "feature_path", product_map, qty_map)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.get("device", "auto") != "cpu" else "cpu")

    model = get_model(model_cfg, num_products=len(product_map), num_quantities=len(qty_map)).to(device)

    class_weighting = train_cfg.get("class_weighting", True)
    prod_weights = compute_class_weights(df, "product", product_map) if class_weighting else None
    qty_weights = compute_class_weights(df, "quantity", qty_map) if class_weighting else None

    criterion_prod = torch.nn.CrossEntropyLoss(weight=prod_weights.to(device) if prod_weights is not None else None)
    criterion_qty = torch.nn.CrossEntropyLoss(weight=qty_weights.to(device) if qty_weights is not None else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.get("max_epochs", 20))

    best_val = float("inf")
    patience = train_cfg.get("early_stop_patience", 5)
    wait = 0
    max_epochs = min(train_cfg.get("max_epochs", 20), cfg_paths.get("max_epochs", 20))

    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, criterion_prod, criterion_qty, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion_prod, criterion_qty, device)
        scheduler.step()
        val_loss = val_stats["loss"]
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_val


def run_hparam_search(
    n_trials: int = 10,
    study_name: str = "slu-optuna",
    storage: str | None = None,
    data_cfg: str = "configs/data.yaml",
    model_cfg: str = "configs/model_m1.yaml",
    train_cfg: str = "configs/train.yaml",
    metadata_features: str = "data/processed/metadata_features.csv",
    max_epochs: int = 15,
):
    cfg_paths = {
        "data_cfg": Path(data_cfg),
        "model_cfg": Path(model_cfg),
        "train_cfg": Path(train_cfg),
        "metadata_features": metadata_features,
        "max_epochs": max_epochs,
    }

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(lambda t: run_trial(t, cfg_paths), n_trials=n_trials)

    print("Best trial:")
    best = study.best_trial
    print(f"  Value: {best.value}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


def main(cli_args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--study_name", default="slu-optuna")
    parser.add_argument(
        "--storage", default=None, help="Optuna storage URL if you want persistence (e.g., sqlite:///optuna.db)"
    )
    parser.add_argument("--data_cfg", default="configs/data.yaml")
    parser.add_argument("--model_cfg", default="configs/model_m1.yaml")
    parser.add_argument("--train_cfg", default="configs/train.yaml")
    parser.add_argument("--metadata_features", default="data/processed/metadata_features.csv")
    parser.add_argument("--max_epochs", type=int, default=15)
    args = parser.parse_args(cli_args)

    run_hparam_search(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        data_cfg=args.data_cfg,
        model_cfg=args.model_cfg,
        train_cfg=args.train_cfg,
        metadata_features=args.metadata_features,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
