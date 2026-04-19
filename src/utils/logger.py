"""Logger factory for JUMP-CP experiments.

Two logger modes:
  wandb  — W&B online (default) or offline (set offline: true in config or
            WANDB_MODE=offline env var).  Offline runs are synced afterwards:
              wandb sync outputs/wandb/offline-run-*
  none   — disable logging entirely (useful for quick debug / HPC smoke tests)
"""

import os
from pytorch_lightning.loggers import WandbLogger


def build_logger(cfg_logger, run_name: str | None = None):
    t = cfg_logger.type.lower()

    if t == "none":
        return False

    if t == "wandb":
        # Offline mode: config flag OR environment variable
        offline = cfg_logger.wandb.offline or os.environ.get("WANDB_MODE") == "offline"
        if offline:
            os.environ["WANDB_MODE"] = "offline"

        # Pull model info for auto-tagging (best-effort)
        try:
            root_cfg = cfg_logger._get_parent()
            model_cfg = root_cfg.model
            backbone  = str(model_cfg.backbone)
            framework = str(model_cfg.framework)
            in_chans  = int(model_cfg.in_channels)
        except Exception:
            backbone = framework = in_chans = None

        # Run name: explicit override wins, else compose from config
        base_name = getattr(cfg_logger, "name", None)
        if run_name:
            composed_name = run_name
        elif base_name and framework:
            composed_name = f"{base_name}-{framework}-{backbone}"
        else:
            composed_name = base_name

        # Tags: user tags + auto-derived tags
        user_tags = list(cfg_logger.tags) if cfg_logger.get("tags") else []
        auto_tags = [t for t in [backbone, framework, f"{in_chans}ch" if in_chans else None] if t]
        tags = list(dict.fromkeys(user_tags + auto_tags))

        logger = WandbLogger(
            project=cfg_logger.wandb.project,
            name=composed_name,
            entity=cfg_logger.wandb.entity if cfg_logger.wandb.entity else None,
            group=cfg_logger.wandb.group,
            job_type=cfg_logger.wandb.job_type,
            log_model=cfg_logger.wandb.log_model,
            tags=tags if tags else None,
        )
        try:
            logger.experiment.config.update(dict(cfg_logger._get_parent()._content))
        except Exception:
            pass
        return logger

    raise ValueError(f"Unknown logger type: {cfg_logger.type!r}  (valid: wandb, none)")
