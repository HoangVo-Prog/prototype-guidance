import unittest

from model.prototype.head import PrototypeConditionedTextHead


class PrototypeInitScheduleTests(unittest.TestCase):
    def _build_head(self, recompute_start_epoch: int) -> PrototypeConditionedTextHead:
        return PrototypeConditionedTextHead(
            input_dim=8,
            num_prototypes=4,
            prototype_dim=8,
            projector_output_dim=8,
            num_classes=1,
            prototype_method_role='semantic_structure',
            prototype_semantic_enabled=True,
            semantic_structure_enabled=True,
            semantic_ramp_use_prototype=True,
            semantic_recompute_start_epoch=recompute_start_epoch,
            prototype_init='normalized_random',
            use_loss_ret=False,
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_gap=False,
            use_loss_diag=False,
            use_loss_semantic_pbt=False,
            use_loss_support=False,
            use_diversity_loss=False,
            use_balance_loss=False,
        )

    def test_prototype_init_is_deferred_until_recompute_start_epoch(self):
        head = self._build_head(recompute_start_epoch=3)

        self.assertFalse(head.prototype_bank.is_initialized())

        _ = head.get_prototype_context(return_debug=False, epoch=2, current_step=0)
        self.assertFalse(head.prototype_bank.is_initialized())

        _ = head.get_prototype_context(return_debug=False, epoch=3, current_step=0)
        self.assertTrue(head.prototype_bank.is_initialized())


if __name__ == '__main__':
    unittest.main()
