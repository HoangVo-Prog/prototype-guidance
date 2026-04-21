import logging
import unittest

from processor.processor import _initialize_early_stopping_state, _update_early_stopping_monitor
from utils.metrics import collect_monitored_eval_rows, summarize_epoch_monitor


def _eval_rows(*rows):
    return {'rows': list(rows), 'authority': {'row_roles': {str(row['task']): 'host' for row in rows}}}


class EarlyStoppingMonitorTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('pas.train.test.early_stopping')

    def test_collect_monitored_rows_by_bucket_and_pattern(self):
        eval_result = _eval_rows(
            {'task': 'host-t2i', 'R1': 66.1},
            {'task': 'global-t2i', 'R1': 66.2},
            {'task': 'global+grab(0.3)-t2i', 'R1': 66.7},
            {'task': 'global+grab(0.5)-t2i', 'R1': 66.8},
        )
        rows = collect_monitored_eval_rows(
            eval_result,
            monitored_bucket='host',
            monitored_task_pattern='*grab*',
        )
        self.assertEqual([row['row_name'] for row in rows], ['global+grab(0.3)-t2i', 'global+grab(0.5)-t2i'])

    def test_summarize_epoch_monitor_uses_deterministic_tie_break(self):
        rows = collect_monitored_eval_rows(
            _eval_rows(
                {'task': 'z-row', 'R1': 66.8},
                {'task': 'a-row', 'R1': 66.8},
            ),
            monitored_bucket='host',
        )
        summary = summarize_epoch_monitor(rows, metric_name='R1', mode='max')
        self.assertEqual(summary['best_row_name'], 'a-row')
        self.assertAlmostEqual(float(summary['best_value']), 66.8, places=6)

    def test_early_stopping_resets_when_best_row_name_changes(self):
        config = {
            'enabled': True,
            'metric': 'R1',
            'mode': 'max',
            'patience': 2,
            'min_delta': 0.0,
            'start_epoch': 1,
            'monitored_bucket': 'host',
            'monitored_task_pattern': None,
            'stop_on_nan': False,
        }
        state = _initialize_early_stopping_state()
        state = _update_early_stopping_monitor(
            logger=self.logger,
            eval_epoch=9,
            eval_result=_eval_rows(
                {'task': 'global+grab(0.5)-t2i', 'R1': 66.75},
                {'task': 'global+grab(0.3)-t2i', 'R1': 66.70},
            ),
            config=config,
            state=state,
        )
        self.assertEqual(state['best_row_name'], 'global+grab(0.5)-t2i')
        self.assertEqual(state['bad_epochs'], 0)

        state = _update_early_stopping_monitor(
            logger=self.logger,
            eval_epoch=10,
            eval_result=_eval_rows(
                {'task': 'global+grab(0.5)-t2i', 'R1': 66.74},
                {'task': 'global+grab(0.3)-t2i', 'R1': 66.71},
            ),
            config=config,
            state=state,
        )
        self.assertEqual(state['bad_epochs'], 1)
        self.assertFalse(state['should_stop'])

        state = _update_early_stopping_monitor(
            logger=self.logger,
            eval_epoch=11,
            eval_result=_eval_rows(
                {'task': 'global+grab(0.5)-t2i', 'R1': 66.90},
                {'task': 'global+grab(0.3)-t2i', 'R1': 66.95},
            ),
            config=config,
            state=state,
        )
        self.assertEqual(state['best_row_name'], 'global+grab(0.3)-t2i')
        self.assertAlmostEqual(float(state['best_value']), 66.95, places=6)
        self.assertEqual(state['best_epoch'], 11)
        self.assertEqual(state['bad_epochs'], 0)
        self.assertFalse(state['should_stop'])

    def test_early_stopping_can_trigger_on_invalid_rows_when_stop_on_nan(self):
        config = {
            'enabled': True,
            'metric': 'R1',
            'mode': 'max',
            'patience': 5,
            'min_delta': 0.0,
            'start_epoch': 1,
            'monitored_bucket': 'host',
            'monitored_task_pattern': None,
            'stop_on_nan': True,
        }
        state = _initialize_early_stopping_state()
        state = _update_early_stopping_monitor(
            logger=self.logger,
            eval_epoch=1,
            eval_result=_eval_rows({'task': 'host-t2i', 'R1': float('nan')}),
            config=config,
            state=state,
        )
        self.assertTrue(state['should_stop'])
        self.assertIn('early_stopping_stop_on_nan=true', str(state.get('stop_reason', '')))


if __name__ == '__main__':
    unittest.main()
