import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from ovha_rod.models.operators.relation_fields import (
    RelationFieldBank,
    _mass_strictly_after,
    _mass_strictly_before,
)


class RelationFieldBankTests(unittest.TestCase):
    def setUp(self):
        self.bank = RelationFieldBank(scales=(0.05, 0.15, 0.30))

    def _impulse(self, height=25, width=25):
        value = torch.zeros(1, height, width)
        value[0, height // 2, width // 2] = 1.0
        return value

    def test_directional_fields_move_to_expected_half_plane(self):
        context = self._impulse()
        center = context.shape[-1] // 2
        fields = self.bank(context)
        for relation, comparison in (
            ("left", lambda y, x: x < center),
            ("right", lambda y, x: x > center),
            ("above", lambda y, x: y < center),
            ("below", lambda y, x: y > center),
        ):
            field = fields[relation][:, 1]
            index = int(field.reshape(-1).argmax())
            y, x = divmod(index, field.shape[-1])
            with self.subTest(relation=relation, peak=(y, x)):
                self.assertTrue(comparison(y, x))

    def test_null_is_zero_and_all_fields_are_finite_nonnegative(self):
        fields = self.bank(self._impulse())
        self.assertTrue(torch.equal(fields["null"], torch.zeros_like(fields["null"])))
        for relation, field in fields.items():
            with self.subTest(relation=relation):
                self.assertTrue(torch.isfinite(field).all())
                self.assertGreaterEqual(float(field.min()), 0.0)

    def test_normalized_scale_is_resolution_consistent(self):
        low = self.bank(self._impulse(25, 25))["near"][:, 1:2]
        high = self.bank(self._impulse(50, 50))["near"][:, 1:2]
        resized = F.interpolate(high, size=(25, 25), mode="bilinear", align_corners=False)
        low = low / low.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        resized = resized / resized.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        self.assertLess(float((low - resized).abs().mean()), 0.08)

    def test_invalid_region_is_zero_and_cannot_wrap(self):
        context = self._impulse()
        valid = torch.zeros_like(context, dtype=torch.bool)
        valid[:, :15, :14] = True
        fields = self.bank(context, valid_mask=valid)
        for field in fields.values():
            self.assertTrue(torch.equal(field.masked_select(~valid[:, None]), torch.zeros_like(field.masked_select(~valid[:, None]))))

    def test_strict_mass_avoids_cumsum_and_preserves_values_and_gradients(self):
        values = torch.tensor(
            [[[0.2, 1.1, 0.4, 2.3],
              [0.7, 0.3, 1.5, 0.8],
              [1.2, 0.6, 0.9, 1.7]]],
            dtype=torch.float64,
        )
        upstream = torch.linspace(
            0.1, 1.2, values.numel(), dtype=values.dtype
        ).reshape_as(values)

        cases = (
            (_mass_strictly_before, -1, False),
            (_mass_strictly_before, -2, False),
            (_mass_strictly_after, -1, True),
            (_mass_strictly_after, -2, True),
        )
        for function, dim, after in cases:
            with self.subTest(function=function.__name__, dim=dim):
                expected_input = values.clone().requires_grad_()
                expected = self._strict_mass_reference(
                    expected_input, dim=dim, after=after)
                expected_gradient, = torch.autograd.grad(
                    (expected * upstream).sum(), expected_input)

                actual_input = values.clone().requires_grad_()
                forbidden = AssertionError(
                    "strict mass must not use torch.cumsum")
                with (
                    patch.object(torch, "cumsum", side_effect=forbidden),
                    patch.object(
                        torch.Tensor, "cumsum", side_effect=forbidden),
                ):
                    actual = function(actual_input, dim=dim)
                    actual_gradient, = torch.autograd.grad(
                        (actual * upstream).sum(), actual_input)

                torch.testing.assert_close(
                    actual, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(
                    actual_gradient, expected_gradient,
                    rtol=1e-12, atol=1e-12)

    def test_strict_mass_handles_singleton_spatial_axes(self):
        values = torch.tensor([[[2.0], [3.0]]])
        self.assertTrue(torch.equal(
            _mass_strictly_before(values, dim=-1),
            torch.zeros_like(values),
        ))
        self.assertTrue(torch.equal(
            _mass_strictly_after(values.transpose(-1, -2), dim=-2),
            torch.zeros_like(values.transpose(-1, -2)),
        ))

    @staticmethod
    def _strict_mass_reference(
        values: torch.Tensor,
        dim: int,
        *,
        after: bool,
    ) -> torch.Tensor:
        moved = values.movedim(dim, -1)
        extent = moved.shape[-1]
        weight = torch.ones(
            (extent, extent), dtype=values.dtype, device=values.device)
        weight = (
            torch.tril(weight, diagonal=-1)
            if after
            else torch.triu(weight, diagonal=1)
        )
        return (moved @ weight).movedim(-1, dim)


if __name__ == "__main__":
    unittest.main()
