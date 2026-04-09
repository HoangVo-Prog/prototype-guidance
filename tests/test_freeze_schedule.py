import os
import sys
import unittest
from types import SimpleNamespace


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.freeze_schedule import (  # noqa: E402
    FreezePhase,
    apply_loss_weight_overrides,
    apply_optimizer_lr_overrides,
    apply_phase_trainability,
    get_active_phase,
    parse_freeze_schedule_config,
)


class _FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = bool(requires_grad)


class _FakeModel:
    def __init__(self):
        self.lambda_host = 1.0
        self.prototype_head = SimpleNamespace(
            losses=SimpleNamespace(
                lambda_ret=1.0,
                lambda_diag=1.0,
                lambda_bal=0.01,
                lambda_div=0.01,
                use_loss_ret=False,
                use_loss_diag=False,
                use_balance_loss=False,
                use_diversity_loss=False,
            )
        )
        self._named_params = [
            ('base_model.visual.proj.weight', _FakeParameter(True)),               # host_backbone
            ('base_model.transformer.block.0.weight', _FakeParameter(True)),       # host_backbone
            ('host_head.core.weight', _FakeParameter(True)),                        # host_retrieval
            ('prototype_head.prototype_bank.prototypes', _FakeParameter(True)),     # prototype_bank
            ('prototype_head.image_projector.net.0.weight', _FakeParameter(True)),  # prototype_projector
            ('prototype_head.losses.class_proxies', _FakeParameter(True)),          # prototype_projector
            ('prototype_head.router.temperature', _FakeParameter(True)),            # routing
            ('prototype_head.contextualizer.q_proj.weight', _FakeParameter(True)),  # routing
            ('prototype_head.text_pool_query', _FakeParameter(True)),               # fusion
            ('prototype_head.token_pooler.scale', _FakeParameter(True)),            # fusion
            ('fusion_module.coefficient', _FakeParameter(True)),                    # fusion
            ('untracked.module.weight', _FakeParameter(True)),                      # untouched
        ]

    def named_parameters(self):
        for name, parameter in self._named_params:
            yield name, parameter


class _FakeOptimizer:
    def __init__(self):
        self.param_groups = [
            {'name': 'prototype_bank', 'lr': 0.001},
            {'name': 'prototype_projectors', 'lr': 0.001},
            {'name': 'class_proxies', 'lr': 0.001},
            {'name': 'prototype_routing', 'lr': 0.001},
            {'name': 'prototype_contextualization', 'lr': 0.001},
            {'name': 'prototype_pooling', 'lr': 0.001},
            {'name': 'host_projectors', 'lr': 0.001},
            {'name': 'image_backbone', 'lr': 0.001},
            {'name': 'text_backbone', 'lr': 0.001},
            {'name': 'other', 'lr': 0.001},
        ]


class FreezeScheduleTests(unittest.TestCase):
    def test_parse_schedule_accepts_and_selects_active_phase(self):
        phases = parse_freeze_schedule_config(
            [
                {
                    'name': 'warmup',
                    'epoch_start': 1,
                    'epoch_end': 2,
                    'trainable_groups': ['prototype_bank'],
                    'frozen_groups': ['host_backbone'],
                },
                {
                    'name': 'joint',
                    'epoch_start': 3,
                    'epoch_end': 5,
                    'trainable_groups': ['host_backbone'],
                    'frozen_groups': [],
                },
            ],
            num_epoch=5,
        )
        self.assertEqual([phase.name for phase in phases], ['warmup', 'joint'])
        self.assertEqual(get_active_phase(phases, 1).name, 'warmup')
        self.assertEqual(get_active_phase(phases, 4).name, 'joint')
        self.assertIsNone(get_active_phase(phases, 6))

    def test_parse_schedule_supports_zero_based_epochs(self):
        phases = parse_freeze_schedule_config(
            [
                {
                    'name': 'warmup',
                    'epoch_start': 0,
                    'epoch_end': 1,
                    'trainable_groups': ['prototype_bank'],
                },
                {
                    'name': 'joint',
                    'epoch_start': 2,
                    'epoch_end': 4,
                    'trainable_groups': ['host_backbone'],
                },
            ],
            num_epoch=5,
        )
        self.assertEqual(phases[0].epoch_start, 1)
        self.assertEqual(phases[0].epoch_end, 2)
        self.assertEqual(phases[1].epoch_start, 3)
        self.assertEqual(phases[1].epoch_end, 5)

    def test_parse_schedule_rejects_invalid_configs(self):
        with self.assertRaisesRegex(ValueError, 'overlapping phases'):
            parse_freeze_schedule_config(
                [
                    {'name': 'p1', 'epoch_start': 1, 'epoch_end': 3, 'trainable_groups': ['prototype_bank']},
                    {'name': 'p2', 'epoch_start': 3, 'epoch_end': 5, 'trainable_groups': ['host_backbone']},
                ],
                num_epoch=5,
            )

        with self.assertRaisesRegex(ValueError, 'must be >= 1'):
            parse_freeze_schedule_config(
                [
                    {'name': 'p1', 'epoch_start': -1, 'epoch_end': 1, 'trainable_groups': ['prototype_bank']},
                ],
                num_epoch=5,
            )

        with self.assertRaisesRegex(ValueError, 'Unsupported module group'):
            parse_freeze_schedule_config(
                [{'name': 'p1', 'epoch_start': 1, 'epoch_end': 2, 'trainable_groups': ['unknown_group']}],
                num_epoch=2,
            )

        with self.assertRaisesRegex(ValueError, 'is empty'):
            parse_freeze_schedule_config(
                [{'name': 'p1', 'epoch_start': 1, 'epoch_end': 1}],
                num_epoch=1,
            )

        with self.assertRaisesRegex(ValueError, 'but training.epochs=4'):
            parse_freeze_schedule_config(
                [{'name': 'p1', 'epoch_start': 1, 'epoch_end': 5, 'trainable_groups': ['prototype_bank']}],
                num_epoch=4,
            )

    def test_apply_phase_trainability_sets_expected_requires_grad(self):
        model = _FakeModel()
        phase = FreezePhase(
            name='warmup',
            epoch_start=1,
            epoch_end=2,
            trainable_groups=('prototype_bank', 'prototype_projector', 'routing', 'fusion'),
            frozen_groups=('host_backbone', 'host_retrieval'),
            lr_overrides={},
            loss_weights={},
        )
        summary = apply_phase_trainability(model, phase)

        self.assertGreater(summary['trainable']['prototype_bank'], 0)
        self.assertGreater(summary['frozen']['host_backbone'], 0)

        status = {name: param.requires_grad for name, param in model.named_parameters()}
        self.assertFalse(status['base_model.visual.proj.weight'])
        self.assertFalse(status['host_head.core.weight'])
        self.assertTrue(status['prototype_head.prototype_bank.prototypes'])
        self.assertTrue(status['prototype_head.image_projector.net.0.weight'])
        self.assertTrue(status['prototype_head.router.temperature'])
        self.assertTrue(status['prototype_head.text_pool_query'])
        self.assertTrue(status['untracked.module.weight'])

    def test_apply_loss_weight_overrides_updates_model_and_args(self):
        model = _FakeModel()
        args = SimpleNamespace(
            lambda_host=1.0,
            lambda_ret=1.0,
            lambda_diag=1.0,
            lambda_bal=0.01,
            lambda_div=0.01,
        )
        applied = apply_loss_weight_overrides(
            model,
            args,
            {
                'lambda_host': 0.0,
                'lambda_ret': 0.25,
                'lambda_diag': 0.5,
                'lambda_bal': 0.003,
                'lambda_div': 0.004,
            },
        )
        self.assertEqual(applied['lambda_host'], 0.0)
        self.assertEqual(model.lambda_host, 0.0)
        self.assertEqual(args.lambda_host, 0.0)
        self.assertEqual(model.prototype_head.losses.lambda_ret, 0.25)
        self.assertEqual(model.prototype_head.losses.lambda_diag, 0.5)
        self.assertEqual(model.prototype_head.losses.lambda_bal, 0.003)
        self.assertEqual(model.prototype_head.losses.lambda_div, 0.004)
        self.assertTrue(model.prototype_head.losses.use_loss_ret)
        self.assertTrue(model.prototype_head.losses.use_loss_diag)
        self.assertTrue(model.prototype_head.losses.use_balance_loss)
        self.assertTrue(model.prototype_head.losses.use_diversity_loss)

    def test_apply_loss_weight_overrides_disables_switches_when_zero(self):
        model = _FakeModel()
        args = SimpleNamespace(
            lambda_host=1.0,
            lambda_ret=1.0,
            lambda_diag=1.0,
            lambda_bal=0.01,
            lambda_div=0.01,
            use_loss_ret=True,
            use_loss_diag=True,
            use_balancing_loss=True,
            use_diversity_loss=True,
        )
        apply_loss_weight_overrides(
            model,
            args,
            {
                'lambda_ret': 0.0,
                'lambda_diag': 0.0,
                'lambda_bal': 0.0,
                'lambda_div': 0.0,
            },
        )
        self.assertFalse(model.prototype_head.losses.use_loss_ret)
        self.assertFalse(model.prototype_head.losses.use_loss_diag)
        self.assertFalse(model.prototype_head.losses.use_balance_loss)
        self.assertFalse(model.prototype_head.losses.use_diversity_loss)
        self.assertFalse(args.use_loss_ret)
        self.assertFalse(args.use_loss_diag)
        self.assertFalse(args.use_balancing_loss)
        self.assertFalse(args.use_diversity_loss)

    def test_apply_optimizer_lr_overrides_maps_logical_groups(self):
        optimizer = _FakeOptimizer()
        hit_counts = apply_optimizer_lr_overrides(
            optimizer,
            {
                'host_backbone': 1e-5,
                'host_retrieval': 2e-5,
                'prototype_bank': 3e-5,
                'prototype_projector': 4e-5,
                'routing': 5e-5,
                'fusion': 6e-5,
            },
        )
        groups = {group['name']: group for group in optimizer.param_groups}

        self.assertEqual(groups['image_backbone']['lr'], 1e-5)
        self.assertEqual(groups['text_backbone']['lr'], 1e-5)
        self.assertEqual(groups['host_projectors']['lr'], 2e-5)
        self.assertEqual(groups['other']['lr'], 0.001)  # unchanged
        self.assertEqual(groups['prototype_bank']['lr'], 3e-5)
        self.assertEqual(groups['prototype_projectors']['lr'], 4e-5)
        self.assertEqual(groups['class_proxies']['lr'], 4e-5)
        self.assertEqual(groups['prototype_routing']['lr'], 5e-5)
        self.assertEqual(groups['prototype_contextualization']['lr'], 5e-5)
        self.assertEqual(groups['prototype_pooling']['lr'], 6e-5)

        self.assertEqual(groups['prototype_bank']['initial_lr'], 3e-5)
        self.assertGreaterEqual(hit_counts['host_backbone'], 2)
        self.assertEqual(hit_counts['host_retrieval'], 1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
