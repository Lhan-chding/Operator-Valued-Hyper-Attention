import math
import unittest
from unittest.mock import patch

import torch

from ovha_rod.models.positional_encoding import (
    DeterministicSinePositionalEncoding,
)


class DeterministicSinePositionalEncodingTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.tensor([
            [
                [False, False, True, True],
                [False, True, True, True],
                [False, False, False, True],
            ],
            [
                [False, True, True, True],
                [False, False, True, True],
                [True, True, True, True],
            ],
        ])

    def test_mask_path_matches_independent_oracle_with_integer_cumsum(self):
        encoder = DeterministicSinePositionalEncoding(
            num_feats=8, temperature=20, normalize=True, offset=0.25)
        original = torch.Tensor.cumsum

        def require_integer_cumsum(tensor, dim, *, dtype=None):
            self.assertFalse(tensor.is_floating_point())
            self.assertEqual(dtype, torch.int64)
            return original(tensor, dim=dim, dtype=dtype)

        with patch.object(
                torch.Tensor, "cumsum", new=require_integer_cumsum):
            actual = encoder(self.mask)
        expected = self._independent_oracle(
            self.mask, num_feats=8, temperature=20,
            normalize=True, offset=0.25)

        self.assertEqual(tuple(actual.shape), (2, 16, 3, 4))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_normalization_parameters_match_independent_oracle(self):
        cases = (
            dict(normalize=False, offset=0.0, scale=2 * math.pi,
                 temperature=10000),
            dict(normalize=True, offset=0.0, scale=2 * math.pi,
                 temperature=10000),
            dict(normalize=True, offset=0.25, scale=3.5,
                 temperature=20),
        )
        for options in cases:
            with self.subTest(options=options):
                encoder = DeterministicSinePositionalEncoding(
                    num_feats=8, **options)
                actual = encoder(self.mask)
                expected = self._independent_oracle(
                    self.mask, num_feats=8, **options)
                torch.testing.assert_close(
                    actual, expected, rtol=0.0, atol=0.0)

    def test_input_only_path_matches_all_valid_mask_and_ignores_content(self):
        encoder = DeterministicSinePositionalEncoding(
            num_feats=8, temperature=20, normalize=True)
        first_input = torch.randn(2, 16, 3, 4, requires_grad=True)
        second_input = torch.randn_like(first_input)
        expected = encoder(torch.zeros(2, 3, 4, dtype=torch.bool))

        first = encoder(mask=None, input=first_input)
        second = encoder(mask=None, input=second_input)

        torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)
        self.assertFalse(first.requires_grad)

    def test_encoding_preserves_downstream_feature_gradient(self):
        encoder = DeterministicSinePositionalEncoding(num_feats=8)
        feature = torch.randn(2, 16, 3, 4, requires_grad=True)
        position = encoder(self.mask)
        self.assertEqual(list(encoder.parameters()), [])
        self.assertFalse(position.requires_grad)

        (feature + position).sum().backward()
        torch.testing.assert_close(feature.grad, torch.ones_like(feature))

    def test_requires_mask_or_input_and_valid_normalized_scale(self):
        encoder = DeterministicSinePositionalEncoding(num_feats=8)
        with self.assertRaises(AssertionError):
            encoder(mask=None, input=None)
        with self.assertRaises(AssertionError):
            DeterministicSinePositionalEncoding(
                num_feats=8, normalize=True, scale="invalid")

    @staticmethod
    def _independent_oracle(
        mask,
        *,
        num_feats,
        temperature,
        normalize,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
    ):
        valid = (~mask).to(torch.float32)
        height, width = valid.shape[-2:]
        y_embed = torch.stack([
            valid[:, :index + 1, :].sum(dim=1)
            for index in range(height)
        ], dim=1)
        x_embed = torch.stack([
            valid[:, :, :index + 1].sum(dim=2)
            for index in range(width)
        ], dim=2)
        if normalize:
            y_embed = (
                (y_embed + offset) / (y_embed[:, -1:, :] + eps) * scale)
            x_embed = (
                (x_embed + offset) / (x_embed[:, :, -1:] + eps) * scale)
        dim_t = torch.arange(num_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4,
        ).view(mask.shape[0], height, width, -1)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
            dim=4,
        ).view(mask.shape[0], height, width, -1)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


if __name__ == "__main__":
    unittest.main()
