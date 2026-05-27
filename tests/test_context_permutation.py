import unittest

from moat_ovha.data.operator_zoo import OperatorZoo
from moat_ovha.models.memory import PoolingMemoryEncoder
from moat_ovha.models.ovha_layer import OVHALayer
from moat_ovha.models.primitives import FourierPrimitive, LocalKernelPrimitive, SeparablePrimitive
from moat_ovha.models.router import PrimitiveRouter


class ContextPermutationTests(unittest.TestCase):
    def test_pooling_memory_is_permutation_invariant(self):
        task = OperatorZoo(seed=11).make_task("separable", "train", context_size=5, resolution=10)
        encoder = PoolingMemoryEncoder(token_count=3)

        original = encoder.encode(task.context)
        shuffled = encoder.encode(tuple(reversed(task.context)))

        self.assertEqual(original.tokens, shuffled.tokens)
        self.assertEqual(original.summary, shuffled.summary)

    def test_predictions_are_stable_under_context_permutation(self):
        task = OperatorZoo(seed=11).make_task("separable", "train", context_size=5, resolution=10)
        encoder = PoolingMemoryEncoder(token_count=3)
        layer = OVHALayer(
            primitives=(
                FourierPrimitive(modes=(1, 2)),
                SeparablePrimitive(rank=2),
                LocalKernelPrimitive(bandwidth=0.25),
            ),
            router=PrimitiveRouter(top_k=None),
            hyper_adapter=None,
        )

        original = layer.forward(task.input_samples, task.query_points, encoder.encode(task.context))
        shuffled = layer.forward(
            task.input_samples,
            task.query_points,
            encoder.encode(tuple(reversed(task.context))),
        )

        self.assertEqual(original.predictions, shuffled.predictions)


if __name__ == "__main__":
    unittest.main()
