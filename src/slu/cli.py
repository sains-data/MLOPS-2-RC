from __future__ import annotations

from pathlib import Path
import typer

from slu import data_ingest, preprocess as preproc, train as train_mod, evaluate as eval_mod, hparam_search as hparam_search_module

app = typer.Typer(help="Command-line interface for the SLU pipeline")


@app.command()
def ingest(
    config: Path = typer.Option("configs/data.yaml", help="Path to data ingestion config (YAML)."),
) -> None:
    """Build consolidated metadata and label maps."""
    data_ingest.main(str(config))


@app.command()
def preprocess(
    metadata: Path = typer.Option("data/processed/metadata.csv", help="Combined metadata CSV to featurize."),
    config: Path = typer.Option("configs/preprocess.yaml", help="Preprocessing config (YAML)."),
) -> None:
    """Convert audio to log-mel features and cache them."""
    preproc.main(metadata_path=str(metadata), cfg_path=str(config))


@app.command()
def train(
    data_cfg: Path = typer.Option("configs/data.yaml", help="Data config (YAML)."),
    preprocess_cfg: Path = typer.Option("configs/preprocess.yaml", help="Preprocess config (YAML)."),
    model_cfg: Path = typer.Option("configs/model_m1.yaml", help="Model config (YAML)."),
    train_cfg: Path = typer.Option("configs/train.yaml", help="Training config (YAML)."),
    metadata_features: Path = typer.Option("data/processed/metadata_features.csv", help="Featurized metadata CSV."),
) -> None:
    """Train the SLU model defined in the configs."""
    train_mod.main(
        data_cfg=str(data_cfg),
        preprocess_cfg=str(preprocess_cfg),
        model_cfg=str(model_cfg),
        train_cfg=str(train_cfg),
        metadata_features=str(metadata_features),
    )


@app.command()
def evaluate(
    registry: Path = typer.Option("artifacts/registry/latest.json", help="Registry JSON to update."),
    metadata_features: Path = typer.Option("data/processed/metadata_features.csv", help="Featurized metadata CSV."),
    data_cfg: Path = typer.Option("configs/data.yaml", help="Data config (YAML)."),
    preprocess_cfg: Path = typer.Option("configs/preprocess.yaml", help="Preprocess config (YAML)."),
    model_cfg: Path | None = typer.Option(None, help="Model config override (YAML)."),
    train_cfg: Path = typer.Option("configs/train.yaml", help="Training config (YAML)."),
    metrics_path: Path = typer.Option("artifacts/metrics.json", help="Where to save metrics JSON."),
    confusion_product: Path = typer.Option("artifacts/confusion_product.png", help="Confusion matrix for product head."),
    confusion_quantity: Path = typer.Option("artifacts/confusion_qty.png", help="Confusion matrix for quantity head."),
    report_path: Path = typer.Option("artifacts/classification_report.txt", help="Classification report output path."),
    val_ratio: float = typer.Option(0.1, help="Validation split ratio."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Evaluate the latest trained model and write metrics/artifacts."""
    eval_mod.run_evaluation(
        registry=str(registry),
        metadata_features=str(metadata_features),
        data_cfg=str(data_cfg),
        preprocess_cfg=str(preprocess_cfg),
        model_cfg=str(model_cfg) if model_cfg is not None else None,
        train_cfg=str(train_cfg),
        metrics_path=str(metrics_path),
        confusion_product=str(confusion_product),
        confusion_quantity=str(confusion_quantity),
        report_path=str(report_path),
        val_ratio=val_ratio,
        seed=seed,
    )


@app.command(name="hparam-search")
def hparam_search_command(
    n_trials: int = typer.Option(10, help="Number of Optuna trials."),
    study_name: str = typer.Option("slu-optuna", help="Optuna study name."),
    storage: str | None = typer.Option(None, help="Optuna storage URL (e.g., sqlite:///optuna.db)."),
    data_cfg: Path = typer.Option("configs/data.yaml", help="Data config (YAML)."),
    model_cfg: Path = typer.Option("configs/model_m1.yaml", help="Model config (YAML)."),
    train_cfg: Path = typer.Option("configs/train.yaml", help="Training config (YAML)."),
    metadata_features: Path = typer.Option("data/processed/metadata_features.csv", help="Featurized metadata CSV."),
    max_epochs: int = typer.Option(15, help="Max epochs per trial."),
) -> None:
    """Run Optuna hyperparameter search."""
    hparam_search_module.run_hparam_search(
        n_trials=n_trials,
        study_name=study_name,
        storage=storage,
        data_cfg=str(data_cfg),
        model_cfg=str(model_cfg),
        train_cfg=str(train_cfg),
        metadata_features=str(metadata_features),
        max_epochs=max_epochs,
    )


if __name__ == "__main__":
    app()
