import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover - environment-dependent
    torch = None
    Sampler = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if torch is not None:
    from datasets.build import build_dataloader


class DummyDatasetFactory:
    def __init__(self, root=''):
        self.img_dir = 'unused_imgs'
        self.train = [
            (0, 0, 'dummy_a.jpg', 'caption a'),
            (0, 1, 'dummy_b.jpg', 'caption b'),
            (1, 2, 'dummy_c.jpg', 'caption c'),
            (1, 3, 'dummy_d.jpg', 'caption d'),
        ]
        self.train_id_container = {0, 1}
        split = {
            'image_pids': [0, 1],
            'img_paths': ['img_a.jpg', 'img_b.jpg'],
            'caption_pids': [0, 1],
            'captions': ['caption a', 'caption b'],
        }
        self.val = split
        self.test = split
        self.val_annos = [
            {'id': 10, 'file_path': 'img_a.jpg', 'captions': ['caption a']},
            {'id': 11, 'file_path': 'img_b.jpg', 'captions': ['caption b']},
        ]


if Sampler is not None:
    class DummyDistributedSampler(Sampler):
        def __init__(self, data_source, batch_size, num_instances):
            self.data_source = data_source
            self.batch_size = batch_size
            self.num_instances = num_instances

        def __iter__(self):
            return iter(range(len(self.data_source)))

        def __len__(self):
            return len(self.data_source)
else:
    DummyDistributedSampler = None


@unittest.skipUnless(torch is not None and Sampler is not None, 'Torch is required for dataloader tests.')
class DataloaderBuildTests(unittest.TestCase):
    def _build_args(self, **overrides):
        base = dict(
            dataset_name='RSTPReid',
            root_dir='unused',
            training=True,
            img_size=(4, 4),
            img_aug=False,
            txt_aug=False,
            text_length=6,
            sampler='identity',
            distributed=True,
            batch_size=4,
            num_instance=2,
            num_workers=0,
            val_dataset='test',
            test_batch_size=2,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_distributed_identity_sampler_builds_train_loader(self):
        with mock.patch.dict('datasets.build.__factory', {'RSTPReid': DummyDatasetFactory}, clear=False):
            with mock.patch('datasets.build.RandomIdentitySampler_DDP', DummyDistributedSampler):
                with mock.patch('datasets.build.get_world_size', return_value=2):
                    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(self._build_args())

        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsNotNone(train_loader.batch_sampler)
        self.assertIsInstance(train_loader.batch_sampler.sampler, DummyDistributedSampler)
        self.assertIsInstance(val_img_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_txt_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(getattr(train_loader, 'eval_loss_loader', None), torch.utils.data.DataLoader)
        self.assertEqual(num_classes, 2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

