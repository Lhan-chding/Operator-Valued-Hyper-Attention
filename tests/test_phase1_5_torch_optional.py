import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; torch implementation tests skipped.")
class Phase15TorchOptionalTests(unittest.TestCase):
    def test_episode_generator_and_ovha_forward_backward(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.ovha import OVHAMetaOperator

        zoo = MetadataFreeOperatorZoo(seed=5)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=2,
            context_points=4,
            support_points=16,
            query_points=8,
            family="compositional_mixed_family",
            split="iid",
            mode="operator_transfer",
            device="cpu",
        )
        model = OVHAMetaOperator(d_model=32, memory_tokens=3)
        output = model(batch)
        loss = torch.nn.functional.mse_loss(output.y_hat, batch.target_y)
        loss.backward()

        self.assertEqual(output.y_hat.shape, batch.target_y.shape)
        self.assertEqual(output.primitive_weights.shape[:2], batch.target_y.shape[:2])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(hidden.family, "compositional_mixed_family")


if __name__ == "__main__":
    unittest.main()
