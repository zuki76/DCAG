"""Check domain embedding isolation and checkpoint compatibility."""

import unittest

import torch

from llava.model.dcag import DCAGConfig, DCAGController, DCAGGenerator


def small_config():
    return DCAGConfig(
        feature_dim=8,
        generator_hidden_dim=16,
        generator_heads=2,
        generator_layers=1,
        r=4,
        factored_head_dim=1,
        num_centroids=1,
        generator_dropout=0.0,
        domain_model_device="cpu",
        domain_model_dtype="float32",
        Wd_mode="none_disable_A_projection",
    )


class DomainEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_old_domains_ignore_weight_decay_and_existing_momentum(self):
        generator = DCAGGenerator(small_config(), 32)
        with torch.no_grad():
            generator.domain_embed.normal_()
        generator.start_domain(0)
        optimizer = torch.optim.AdamW(
            [generator.current_domain_embed], lr=0.01, weight_decay=0.05
        )
        generator._domain_embedding(0).sum().backward()
        optimizer.step()
        self.assertGreater(
            optimizer.state[generator.current_domain_embed]["exp_avg"].abs().sum(), 0
        )
        learned = generator.current_domain_embed.detach().clone()
        generator.start_domain(3)
        torch.testing.assert_close(generator.domain_embed[0], learned, rtol=0, atol=0)
        frozen = generator.domain_embed.clone()
        current_before = generator.current_domain_embed.detach().clone()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            generator._domain_embedding(3).square().sum().backward()
            optimizer.step()
            torch.testing.assert_close(generator.domain_embed, frozen, rtol=0, atol=0)
        self.assertFalse(torch.equal(current_before, generator.current_domain_embed))
        self.assertNotIn("domain_embed", dict(generator.named_parameters()))
        self.assertFalse(generator._domain_embedding(0).requires_grad)

    def test_all_forward_paths_use_only_the_active_embedding(self):
        generator = DCAGGenerator(small_config(), 32)
        generator.start_domain(2)
        features = torch.randn(1, 3, 8)
        paths = (
            lambda domain: generator.forward_all_slots(features, domain_id=domain)["U"],
            lambda domain: generator.forward_for_layer(0, features, domain_id=domain)[
                "U_layer"
            ],
            lambda domain: generator.forward_for_projector(features, domain_id=domain)[
                "U_proj"
            ],
        )
        for forward in paths:
            with self.subTest(path=paths.index(forward)):
                generator.zero_grad(set_to_none=True)
                forward(2)[..., 0].sum().backward()
                self.assertIsNotNone(generator.current_domain_embed.grad)
                self.assertGreater(generator.current_domain_embed.grad.abs().sum(), 0)
                generator.zero_grad(set_to_none=True)
                forward(0)[..., 0].sum().backward()
                self.assertIsNone(generator.current_domain_embed.grad)

    def test_active_and_completed_checkpoint_roundtrip(self):
        generator = DCAGGenerator(small_config(), 32)
        generator.start_domain(4)
        with torch.no_grad():
            generator.current_domain_embed.normal_()
        expected = generator.current_domain_embed.detach().clone()
        active = DCAGGenerator(small_config(), 32)
        active.load_state_dict(generator.state_dict(), strict=True)
        active.eval()
        torch.testing.assert_close(
            active._domain_embedding(4), expected, rtol=0, atol=0
        )
        active.start_domain(4)
        torch.testing.assert_close(
            active.current_domain_embed, expected, rtol=0, atol=0
        )
        generator.freeze_domain_embedding()
        saved = {
            **{
                k: v.detach().clone()
                for k, v in generator.named_parameters()
                if v.requires_grad
            },
            **{
                k: v.detach().clone()
                for k, v in generator.named_buffers()
                if k not in generator._non_persistent_buffers_set
            },
        }
        restored = DCAGGenerator(small_config(), 32)
        restored.load_state_dict(saved, strict=True)
        restored.start_domain(1)
        torch.testing.assert_close(
            restored._domain_embedding(4), expected, rtol=0, atol=0
        )
        self.assertFalse(restored._domain_embedding(4).requires_grad)

    def test_legacy_table_loads_without_changing_predictions(self):
        generator = DCAGGenerator(small_config(), 32)
        with torch.no_grad():
            generator.domain_embed.normal_()
        legacy = generator.state_dict()
        legacy.pop("current_domain_embed")
        legacy.pop("domain_embed_active_id")
        restored = DCAGGenerator(small_config(), 32)
        restored.load_state_dict(legacy, strict=True)
        features = torch.randn(1, 3, 8)
        for domain in range(5):
            torch.testing.assert_close(
                restored.forward_all_slots(features, domain_id=domain)["trunk"],
                generator.forward_all_slots(features, domain_id=domain)["trunk"],
                rtol=0,
                atol=0,
            )
        restored.start_domain(2)
        torch.testing.assert_close(
            restored.current_domain_embed, legacy["domain_embed"][2]
        )

    def test_controller_uses_explicit_domain_and_freezes_at_task_end(self):
        controller = DCAGController(small_config(), 32)
        controller.start_task(0, domain_id=3)
        self.assertEqual(controller.current_domain_id, 3)
        self.assertEqual(controller.generator.domain_embed_active_id.item(), 3)
        with torch.no_grad():
            controller.generator.current_domain_embed.fill_(1.25)
        with self.assertRaisesRegex(ValueError, "Batch domain"):
            controller.set_batch_context(None, torch.tensor([0]))
        controller.snapshot_and_end_task()
        torch.testing.assert_close(
            controller.generator.domain_embed[3],
            torch.full((16,), 1.25),
            rtol=0,
            atol=0,
        )
        self.assertEqual(controller.generator.domain_embed_active_id.item(), -1)
        controller.start_task(1, domain_id=0)
        self.assertFalse(controller.generator._domain_embedding(3).requires_grad)


if __name__ == "__main__":
    unittest.main()
