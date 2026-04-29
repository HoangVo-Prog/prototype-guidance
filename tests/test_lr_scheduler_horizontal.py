import unittest

import torch

from solver.lr_scheduler import LRSchedulerWithWarmup


class HorizontalSchedulerTests(unittest.TestCase):
    def test_horizontal_scheduler_is_flat_after_warmup(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = LRSchedulerWithWarmup(
            optimizer=optimizer,
            milestones=[5, 10],
            gamma=0.1,
            mode="horizontal",
            warmup_factor=0.1,
            warmup_epochs=2,
            warmup_method="linear",
            total_epochs=20,
            target_lr=0.0,
            power=0.9,
        )

        observed = []
        for _ in range(7):
            optimizer.step()
            scheduler.step()
            observed.append(float(optimizer.param_groups[0]["lr"]))

        self.assertLess(observed[0], 0.1)
        for idx in range(2, len(observed)):
            self.assertAlmostEqual(observed[idx], 0.1, places=8)


if __name__ == "__main__":
    unittest.main()
