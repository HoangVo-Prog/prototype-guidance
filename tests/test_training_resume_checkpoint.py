import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None

try:
    from utils.checkpoint import Checkpointer
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    Checkpointer = None


if torch is not None:
    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)
else:
    class _TinyModel:  # pragma: no cover - only used when torch is unavailable
        pass


class _DummyScaler:
    def __init__(self):
        self._state = {"scale": 1024.0, "growth_interval": 2000}

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


def _build_optimizer_and_scheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return optimizer, scheduler


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@unittest.skipIf(
    torch is None or Checkpointer is None,
    "torch runtime is required for checkpoint resume tests",
)
class TrainingResumeCheckpointTests(unittest.TestCase):
    def test_resumable_checkpoint_restores_full_training_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _seed_all(77)
            model = _TinyModel()
            optimizer, scheduler = _build_optimizer_and_scheduler(model)
            scaler = _DummyScaler()

            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            checkpointer = Checkpointer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=str(tmp_path),
                save_to_disk=True,
            )
            checkpointer.save_training_checkpoint(
                "checkpoint_training_latest",
                epoch=7,
                global_step=1152,
                checkpoint_kind="latest",
                metric_name="R1",
                metric_mode="max",
                best_metric_state={
                    "name": "R1",
                    "mode": "max",
                    "value": 66.75,
                    "best_epoch": 7,
                    "selected_row": "host-t2i",
                },
                latest_metric_state={
                    "name": "R1",
                    "mode": "max",
                    "value": 66.75,
                    "epoch": 7,
                    "selected_row": "host-t2i",
                },
                early_stopping_state={"bad_epochs": 1, "should_stop": False},
                modular_best_metric_value_by_group={"host": 66.75},
                scaler=scaler,
                config_snapshot={"training": {"epochs": 60}},
            )

            expected_py = random.random()
            expected_np = float(np.random.rand())
            expected_torch = float(torch.rand(1).item())

            _ = [random.random() for _ in range(3)]
            _ = np.random.rand(3)
            _ = torch.rand(3)

            resumed_model = _TinyModel()
            resumed_optimizer, resumed_scheduler = _build_optimizer_and_scheduler(resumed_model)
            resumed_scaler = _DummyScaler()
            resumed_checkpointer = Checkpointer(
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                save_dir=str(tmp_path),
                save_to_disk=True,
            )

            bundle = resumed_checkpointer.resume_training(
                str(tmp_path / "checkpoint_training_latest.pth"),
                strict=True,
                restore_rng=True,
                scaler=resumed_scaler,
            )

            self.assertEqual(bundle["start_epoch"], 8)
            self.assertEqual(bundle["global_step"], 1152)
            self.assertTrue(bundle["optimizer_restored"])
            self.assertTrue(bundle["scheduler_restored"])
            self.assertTrue(bundle["scaler_restored"])
            self.assertTrue(bundle["rng_restored"])
            self.assertEqual(bundle["warnings"], [])
            self.assertEqual(bundle["training_state"]["best_metric_state"]["selected_row"], "host-t2i")
            self.assertEqual(resumed_scaler.state_dict(), scaler.state_dict())

            for old_param, new_param in zip(model.parameters(), resumed_model.parameters()):
                self.assertTrue(torch.allclose(old_param.detach(), new_param.detach()))

            self.assertAlmostEqual(random.random(), expected_py, places=7)
            self.assertAlmostEqual(float(np.random.rand()), expected_np, places=7)
            self.assertAlmostEqual(float(torch.rand(1).item()), expected_torch, places=7)

    def test_resume_backward_compat_weights_only_non_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model = _TinyModel()
            weights_only_path = tmp_path / "weights_only.pth"
            torch.save({"model": model.state_dict(), "epoch": 3}, str(weights_only_path))

            resumed_model = _TinyModel()
            resumed_optimizer, resumed_scheduler = _build_optimizer_and_scheduler(resumed_model)
            checkpointer = Checkpointer(
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                save_dir=str(tmp_path),
                save_to_disk=True,
            )

            bundle = checkpointer.resume_training(str(weights_only_path), strict=False, restore_rng=False, scaler=None)
            self.assertFalse(bundle["optimizer_restored"])
            self.assertFalse(bundle["scheduler_restored"])
            self.assertEqual(bundle["start_epoch"], 4)
            self.assertIn("optimizer_state_missing", bundle["warnings"])
            self.assertIn("scheduler_state_missing", bundle["warnings"])
            self.assertIn("training_state_missing", bundle["warnings"])

    def test_resume_backward_compat_weights_only_strict_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model = _TinyModel()
            weights_only_path = tmp_path / "weights_only_strict.pth"
            torch.save({"model": model.state_dict(), "epoch": 3}, str(weights_only_path))

            resume_model = _TinyModel()
            resume_optimizer, resume_scheduler = _build_optimizer_and_scheduler(resume_model)
            checkpointer = Checkpointer(
                model=resume_model,
                optimizer=resume_optimizer,
                scheduler=resume_scheduler,
                save_dir=str(tmp_path),
                save_to_disk=True,
            )

            with self.assertRaisesRegex(RuntimeError, "Strict resume requested"):
                checkpointer.resume_training(str(weights_only_path), strict=True, restore_rng=False, scaler=None)


if __name__ == "__main__":
    unittest.main()
