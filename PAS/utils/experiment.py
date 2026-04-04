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
        self._comparison_history = {}

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
            # PAS logs canonical metrics directly through wandb.log(...);
            # disabling TB sync avoids a second mirrored namespace such as
            # train_metrics/* when TensorBoard event files are present.
            sync_tensorboard=False,
        )
        self._define_default_metrics()
        self._upload_run_configs(args, run_name)
        if getattr(args, 'wandb_log_code', False):
            try:
                wandb.run.log_code(root=os.path.dirname(output_dir))
            except Exception as exc:  # pragma: no cover
                self.logger.warning('Unable to log code to W&B: %s', exc)
        self.enabled = True

    def _upload_run_configs(self, args, run_name: str):
        if self._run is None:
            return
        config_paths = []
        for filename in ('configs.yaml', 'resolved_config.yaml'):
            local_path = os.path.join(self.output_dir, filename)
            if os.path.isfile(local_path):
                config_paths.append((local_path, filename))
        source_config = getattr(args, 'config_file', None)
        if source_config and os.path.isfile(source_config):
            config_paths.append((source_config, 'source_config.yaml'))
        if not config_paths:
            return
        try:
            artifact = wandb.Artifact(name=f'{run_name}-configs', type='run_config')
            for local_path, artifact_name in config_paths:
                artifact.add_file(local_path=local_path, name=artifact_name)
            self._run.log_artifact(artifact)
        except Exception as exc:  # pragma: no cover
            self.logger.warning('Unable to upload run config files to W&B: %s', exc)

    def _define_default_metrics(self):
        if self._run is None:
            return
        try:
            wandb.define_metric('train/step')
            wandb.define_metric('train/*', step_metric='train/step')
            wandb.define_metric('debug/*', step_metric='train/step')
            wandb.define_metric('val/epoch')
            wandb.define_metric('val/*', step_metric='val/epoch')
        except Exception as exc:  # pragma: no cover
            self.logger.warning('Unable to define W&B metric axes: %s', exc)

    def log(self, metrics: Dict[str, object], step: Optional[int] = None, commit: bool = True):
        if not self.enabled or self._run is None or not metrics:
            return
        if step is None:
            wandb.log(metrics, commit=commit)
            return
        wandb.log(metrics, step=step, commit=commit)

    def log_comparison_charts(
        self,
        epoch: float,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        if not self.enabled or self._run is None:
            return
        plot_api = getattr(wandb, 'plot', None)
        line_series = getattr(plot_api, 'line_series', None) if plot_api is not None else None
        if line_series is None:
            return

        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}
        metric_names = sorted(set(train_metrics.keys()) | set(val_metrics.keys()))
        if not metric_names:
            return

        epoch_value = float(epoch)
        payload = {}
        for metric_name in metric_names:
            history = self._comparison_history.setdefault(
                metric_name,
                {'epochs': [], 'train': [], 'val': []},
            )
            history['epochs'].append(epoch_value)
            history['train'].append(train_metrics.get(metric_name))
            history['val'].append(val_metrics.get(metric_name))

            keys = []
            ys = []
            if any(value is not None for value in history['train']):
                keys.append('train')
                ys.append([float('nan') if value is None else float(value) for value in history['train']])
            if any(value is not None for value in history['val']):
                keys.append('val')
                ys.append([float('nan') if value is None else float(value) for value in history['val']])
            if not keys:
                continue

            try:
                payload[f'plots/{metric_name}'] = line_series(
                    xs=list(history['epochs']),
                    ys=ys,
                    keys=keys,
                    title=f'{metric_name}: train vs val',
                    xname='epoch',
                )
            except Exception as exc:  # pragma: no cover
                self.logger.warning('Unable to build W&B comparison chart for %s: %s', metric_name, exc)

        if payload:
            wandb.log(payload)

    def finish(self):
        if self.enabled and self._run is not None:
            wandb.finish()
            self._run = None
            self.enabled = False
