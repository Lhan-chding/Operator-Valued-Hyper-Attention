import math
import unittest

from moat_ovha.models.primitives import (
    FourierPrimitive,
    LocalKernelPrimitive,
    Sample,
    SeparablePrimitive,
)


class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            Sample(point=0.0, value=0.0),
            Sample(point=0.5, value=1.0),
            Sample(point=1.0, value=0.0),
        ]
        self.queries = [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_primitives_return_one_value_per_query(self):
        primitives = [
            FourierPrimitive(modes=(1, 2, 3)),
            SeparablePrimitive(rank=3),
            LocalKernelPrimitive(bandwidth=0.35),
        ]

        for primitive in primitives:
            with self.subTest(primitive=primitive.name):
                values = primitive.apply(self.samples, self.queries)
                self.assertEqual(len(values), len(self.queries))
                self.assertTrue(all(math.isfinite(value) for value in values))

    def test_primitives_support_batch_inputs(self):
        primitive = LocalKernelPrimitive(bandwidth=0.35)
        batch = [self.samples, [Sample(point=s.point, value=s.value + 1.0) for s in self.samples]]

        values = primitive.apply_batch(batch, self.queries)

        self.assertEqual(len(values), 2)
        self.assertEqual(len(values[0]), len(self.queries))
        self.assertGreater(values[1][2], values[0][2])

    def test_context_condition_is_immutable_and_changes_output(self):
        primitive = FourierPrimitive(modes=(1, 2), scale=1.0)
        conditioned = primitive.context_condition({"scale": 1.5, "bias": 0.1})

        self.assertIsNot(primitive, conditioned)
        self.assertEqual(primitive.scale, 1.0)
        self.assertEqual(conditioned.scale, 1.5)
        base = primitive.apply(self.samples, [0.5])[0]
        changed = conditioned.apply(self.samples, [0.5])[0]
        self.assertNotAlmostEqual(base, changed)

    def test_finite_difference_gradient_is_stable(self):
        primitive = SeparablePrimitive(rank=2, scale=1.0)

        grad = primitive.finite_difference(
            self.samples,
            [0.25, 0.5],
            parameter="scale",
            epsilon=1e-4,
        )

        self.assertEqual(len(grad), 2)
        self.assertTrue(all(math.isfinite(value) for value in grad))
        self.assertGreater(sum(abs(value) for value in grad), 0.0)


if __name__ == "__main__":
    unittest.main()
