import unittest

import torch
import torch.nn.functional as F

from ovha_rod.models.operators.relation_fields import RelationFieldBank


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


if __name__ == "__main__":
    unittest.main()
