import hydra
import json
import os
import subprocess
from datetime import datetime, timezone
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.datamodule import JumpCPDataModule
from src.models.dann_classifier import DANNClassifier
from src.models.erm_classifier import ERMClassifier
from src.models.in_classifier import InstanceNormClassifier
from src.models.ttt_bn_classifier import TTTBatchNormClassifier
from src.utils.logger import build_logger
import torch

torch.backends.cudnn.benchmark = True        # let cuDNN pick the fastest conv kernels
torch.set_flush_denormal(True)               # avoid slowdowns from tiny denormal numbers
torch.set_float32_matmul_precision("high")   # TF32 on Ampere/Ada/Hopper (RTX 30/40/50xx);
                                             # harmless no-op on Pascal (P40) and older


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # ---------------- Data ----------------
    dm = JumpCPDataModule(**cfg.data)
    dm.setup()  # explicit call so num_classes / domain_names are available before model build

    # ---------------- Model ----------------
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_kwargs, dict)
    framework = str(model_kwargs.pop("framework")).lower()

    # domain_names: readable batch labels for per-domain test metrics
    domain_names = (list(dm.domain_encoder.classes_)
                    if cfg.data.get("return_domain") else None)

    if framework == "erm":
        model = ERMClassifier(num_classes=dm.num_classes,
                              domain_names=domain_names, **model_kwargs)
    elif framework == "dann":
        model = DANNClassifier(num_classes=dm.num_classes,
                               num_domains=dm.num_domains,
                               domain_names=domain_names, **model_kwargs)
    elif framework == "instance_norm":
        model = InstanceNormClassifier(num_classes=dm.num_classes,
                                       domain_names=domain_names, **model_kwargs)
    elif framework == "ttt_bn":
        model = TTTBatchNormClassifier(num_classes=dm.num_classes,
                                       domain_names=domain_names, **model_kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # ---------------- Run identity (used by logger + checkpoint) ----------------
    fold_index = cfg.data.get("fold_index", 0)

    # Derive a short fold-strategy label from the fold config file metadata.
    # e.g. strategy=kfold, k=5 → "k5"  |  strategy=lobo → "lobo"
    fold_label = "fold"  # fallback
    try:
        with open(cfg.data.fold_config_file) as _f:
            _fc = json.load(_f)
        _strategy = _fc.get("strategy", "")
        _k = _fc.get("k")
        fold_label = "lobo" if _strategy == "lobo" else (f"k{_k}" if _k else _strategy)
    except Exception:
        pass

    run_name = f"jumpcp-{framework}-{fold_label}-fold{fold_index}"

    # ---------------- Logger ----------------
    logger = build_logger(cfg.logger, run_name=run_name)

    # Log run metadata to W&B if available
    run_ts = os.environ.get("RUN_TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    try:
        if logger and hasattr(logger, "experiment"):
            exp = logger.experiment
            try:
                # Environment & version info for reproducibility
                git_hash = None
                try:
                    git_hash = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=cfg.base_dir, stderr=subprocess.DEVNULL
                    ).decode().strip()
                except Exception:
                    pass

                gpu_name = None
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)

                import albumentations
                wb_meta = {
                    "run_timestamp": run_ts,
                    "git_commit": git_hash,
                    "env/torch": torch.__version__,
                    "env/pytorch_lightning": pl.__version__,
                    "env/cuda": torch.version.cuda,
                    "env/cudnn": str(torch.backends.cudnn.version()),
                    "env/albumentations": albumentations.__version__,
                    "env/gpu": gpu_name,
                }
                split_summary = getattr(dm, "split_summary", {})
                if split_summary:
                    wb_meta["data_split"] = split_summary
                exp.config.update(wb_meta)
            except Exception:
                pass
    except Exception:
        pass

    # ---------------- Callbacks ----------------
    ckpt_dir = cfg.train.get("checkpoint_dir")
    ckp = ModelCheckpoint(
        dirpath=ckpt_dir,  # None → PL default (Hydra output dir); set on HPC to avoid $HOME quota
        monitor="val_acc", mode="max", save_top_k=1,
        filename=f"{framework}_{fold_label}_fold{fold_index}_{{epoch:02d}}_{{val_acc:.3f}}"
    )
    es = EarlyStopping(
        monitor="val_acc", mode="max", patience=cfg.train.early_stopping_patience
    )

    # ---------------- Trainer ----------------
    trainer_kwargs = dict(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        enable_checkpointing=cfg.train.enable_checkpointing,
        callbacks=[ckp, es],
        max_epochs=cfg.train.max_epochs,
        logger=logger,
    )

    # optional tiny-overfit / debug knobs
    for k in [
        "max_steps",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "overfit_batches",
        "gradient_clip_val",
    ]:
        v = cfg.train.get(k)
        if v is not None:
            trainer_kwargs[k] = v

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dm)

    # Load best checkpoint manually with weights_only=False to avoid
    # PyTorch >=2.6 blocking numpy types in PL checkpoint state dicts.
    best_ckpt = torch.load(ckp.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    test_results = trainer.test(model, datamodule=dm)

    # Flush W&B (critical for offline mode — ensures test metrics are persisted)
    if logger and hasattr(logger, "experiment"):
        logger.experiment.finish()

    # ---------------- Save results.json ----------------
    # test_results is a list[dict] — one dict per test dataloader.
    metrics = test_results[0] if test_results else {}

    # Reproducibility: capture git + environment info
    git_hash = None
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cfg.base_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    results = {
        "run_name":        run_name,
        "method":          framework,
        "fold_strategy":   fold_label,
        "fold_index":      fold_index,
        "fold_config_file": str(cfg.data.fold_config_file),
        "backbone":        str(cfg.model.backbone),
        "best_checkpoint": str(ckp.best_model_path),
        "test_metrics":    {k: float(v) for k, v in metrics.items()},
        "completed_at":    datetime.now(timezone.utc).isoformat(),
        "run_timestamp":   run_ts,
        "git_commit":      git_hash,
        "seed":            cfg.seed,
        "precision":       cfg.train.precision,
        "max_epochs":      cfg.train.max_epochs,
        "lr":              cfg.model.lr,
        "batch_size":      cfg.data.batch_size,
        "env": {
            "torch":              torch.__version__,
            "pytorch_lightning":  pl.__version__,
            "cuda":               torch.version.cuda,
            "gpu":                gpu_name,
        },
    }

    # 1) Hydra output dir (cwd during the run)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 2) Timestamped results dir for easy aggregation
    results_dir = os.path.join(cfg.base_dir, "outputs", "results", run_ts)
    os.makedirs(results_dir, exist_ok=True)
    flat_path = os.path.join(results_dir, f"{run_name}.json")
    with open(flat_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
