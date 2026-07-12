import unittest

import torch

from ovha_rod.models.role_encoder import LatentRoleEncoder, role_diversity_loss


class LatentRoleEncoderTests(unittest.TestCase):
    def test_shapes_masking_and_normalization(self):
        torch.manual_seed(7)
        encoder = LatentRoleEncoder(d_model=16, num_heads=4)
        text = torch.randn(2, 6, 16)
        valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)

        state = encoder(text, valid)

        self.assertEqual(tuple(state.vectors.shape), (2, 4, 16))
        self.assertEqual(tuple(state.attention.shape), (2, 4, 6))
        padded_attention = state.attention.masked_select(~valid[:, None, :])
        self.assertTrue(torch.equal(padded_attention, torch.zeros_like(padded_attention)))
        self.assertTrue(torch.allclose(state.attention.sum(-1), torch.ones(2, 4), atol=1e-6))

    def test_padding_content_cannot_change_roles(self):
        torch.manual_seed(11)
        encoder = LatentRoleEncoder(d_model=8, num_heads=2).eval()
        text = torch.randn(1, 5, 8)
        valid = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        changed = text.clone()
        changed[:, 3:] = 1000.0

        first = encoder(text, valid)
        second = encoder(changed, valid)

        self.assertTrue(torch.allclose(first.vectors, second.vectors, atol=1e-6))
        self.assertTrue(torch.allclose(first.attention, second.attention, atol=1e-6))

    def test_all_padding_is_rejected(self):
        encoder = LatentRoleEncoder(d_model=8, num_heads=2)
        with self.assertRaisesRegex(ValueError, "valid text token"):
            encoder(torch.zeros(1, 3, 8), torch.zeros(1, 3, dtype=torch.bool))

    def test_diversity_loss_is_finite_and_differentiable(self):
        logits = torch.randn(2, 4, 5, requires_grad=True)
        attention = logits.softmax(-1)
        loss = role_diversity_loss(attention)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
