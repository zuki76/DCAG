"""Small CPU checks for the adapter generator and continual state."""

import io
import unittest

import torch

from llava.model.dcag import (
    DCAGConfig,
    DCAGGenerator,
    FisherSubspaceTracker,
    ResidualLoRAManager,
    TaskSliceManager,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_layer_and_projector_generation_match_all_slots(self):
        config = DCAGConfig(
            feature_dim=16,
            generator_hidden_dim=32,
            generator_heads=4,
            generator_layers=1,
            r=4,
            factored_head_dim=2,
            generator_dropout=0.0,
            domain_model_device="cpu",
            domain_model_dtype="float32",
        )
        generator = DCAGGenerator(config, num_hidden_layers=32).eval()
        with torch.no_grad():
            for head in [generator.U_head, *generator.V_heads.values()]:
                for parameter in head.parameters():
                    parameter.normal_(0, 0.02)
        features = torch.randn(1, 5, 16, requires_grad=True)
        complete = generator.forward_all_slots(features, domain_id=2)
        layer = generator.forward_for_layer(3, features, domain_id=2)
        torch.testing.assert_close(layer["U_layer"], complete["U"][:, 21:28])
        for bucket in (4096, 11008):
            target_ids = layer[f"V_{bucket}_targets"]
            global_ids = complete["bucket_slot_ids"][bucket]
            indices = [global_ids.index(21 + index) for index in target_ids]
            torch.testing.assert_close(
                layer[f"V_{bucket}"], complete["V"][str(bucket)][:, indices]
            )
        projector = generator.forward_for_projector(features, domain_id=2)
        torch.testing.assert_close(projector["U_proj"], complete["U"][:, 224:226])
        loss = layer["U_layer"].square().mean() + layer["V_4096"].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_task_slice_freeze_and_checkpoint_roundtrip(self):
        def manager():
            return TaskSliceManager(
                r=8,
                h_d_per_kind={"llm": 2},
                kind_per_slot={0: "llm"},
                d_out_per_slot={0: 6},
            )

        slices = manager()
        slices.start_task(0)
        slices.record_domain_for_task(0, 0)
        self.assertEqual(slices.B_effective(0, 0).abs().sum().item(), 0)
        with torch.no_grad():
            slices.C_slot_0_current.normal_()
        learned = slices.B_effective(0, 0).detach().clone()
        slices.end_task(0)
        torch.testing.assert_close(slices.B_effective(0, 0), learned)
        old_r = slices.R_slot_0_frozen.clone()
        old_c = slices.C_slot_0_frozen.clone()
        slices.start_task(1)
        slices.record_domain_for_task(1, 1)
        torch.testing.assert_close(
            old_r.T @ slices.R_slot_0_current, torch.zeros(2, 2), atol=1e-6, rtol=0
        )
        with torch.no_grad():
            slices.C_slot_0_current.add_(1)
        torch.testing.assert_close(slices.R_slot_0_frozen, old_r)
        torch.testing.assert_close(slices.C_slot_0_frozen, old_c)
        slices.end_task(1)
        payload = io.BytesIO()
        torch.save(slices.state_dict(), payload)
        payload.seek(0)
        restored = manager()
        for task in (0, 1):
            restored.start_task(task)
            restored.end_task(task)
        restored.load_state_dict(
            torch.load(payload, map_location="cpu", weights_only=False)
        )
        torch.testing.assert_close(restored.B_effective(0, 0), slices.B_effective(0, 0))
        self.assertEqual(restored.domain_to_task, {0: 0, 1: 1})

    def test_residual_retains_completed_domain(self):
        residual = ResidualLoRAManager(2, 4, {0: 5}, {0: 3})
        residual.start_task(0)
        with torch.no_grad():
            residual.residual_B_slot_0_current.normal_()
        pair = [x.detach().clone() for x in residual.current_AB(0)]
        residual.end_task(0)
        residual.start_task(1)
        for actual, expected in zip(residual.task_AB(0, 0), pair):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(residual.residual_B_slot_0_current.abs().sum().item(), 0)

    def test_fisher_does_not_invent_unobserved_directions(self):
        tracker = FisherSubspaceTracker(12, 4)
        self.assertEqual(tuple(tracker.extract(2, None).shape), (12, 0))
        tracker.update_signal(torch.randn(12))
        previous = torch.eye(12)[:, :2]
        directions = tracker.extract(2, previous)
        torch.testing.assert_close(
            previous.T @ directions, torch.zeros(2, 2), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            directions.T @ directions, torch.eye(2), atol=1e-5, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
