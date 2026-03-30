import logging
import os
from typing import Dict, Optional

from utils.config import build_runtime_config

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


class ExperimentTracker:
    def __init__(self, args, output_dir: str, distributed_rank: int = 0):
        self.logger = logging.getLogger('pas.experiment')
        self.output_dir = output_dir
        self.enabled = False
        self._run = None

        use_wandb = bool(getattr(args, 'use_wandb', False)) and distributed_rank == 0
        if not use_wandb:
            return
        if wandb is None:
            self.logger.warning('Weights & Biases logging requested but `wandb` is not installed. Continuing without W&B.')
            return

        os.makedirs(output_dir, exist_ok=True)
        run_name = getattr(args, 'wandb_run_name', None) or os.path.basename(output_dir.rstrip(os.sep))
        self._run = wandb.init(
            project=getattr(args, 'wandb_project', 'PAS'),
            entity=getattr(args, 'wandb_entity', None),
            name=run_name,
            group=getattr(args, 'wandb_group', None),
            mode=getattr(args, 'wandb_mode', 'online'),
            tags=getattr(args, 'wandb_tags', None),
            notes=getattr(args, 'wandb_notes', None),
            dir=output_dir,
            config=build_runtime_config(args),
            reinit=True,
        )
        if getattr(args, 'wandb_log_code', False):
            try:
                wandb.run.log_code(root=os.path.dirname(output_dir))
            except Exception as exc:  # pragma: no cover
                self.logger.warning('Unable to log code to W&B: %s', exc)
        self.enabled = True

    def log(self, metrics: Dict[str, object], step: Optional[int] = None, commit: bool = True):
        if not self.enabled or self._run is None or not metrics:
            return
        wandb.log(metrics, step=step, commit=commit)

    def finish(self):
        if self.enabled and self._run is not None:
            wandb.finish()
            self._run = None
            self.enabled = False

