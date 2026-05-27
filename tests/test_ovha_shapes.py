import unittest

from moat_ovha.data.operator_zoo import OperatorZoo
from moat_ovha.models.hyper_adapter import LowRankHyperAdapter
from moat_ovha.models.memory import PerceiverMemoryEncoder, PoolingMemoryEncoder
from moat_ovha.models.ovha_layer import OVHALayer
from moat_ovha.models.primitives import FourierPrimitive, LocalKernelPrimitive, SeparablePrimitive
from moat_ovha.models.router import PrimitiveRouter


class OVHALayerShapeTests(unittest.TestCase):
    def setUp(self):
        self.task = OperatorZoo(seed=7).make_task(
            family="mixed",
            split="train",
            context_size=4,
            resolution=12,
        )
        self.primitives = (
            FourierPrimitive(modes=(1, 2)),
            SeparablePrimitive(rank=3),
            LocalKernelPrimitive(bandwidth=0.2),
        )

    def test_dense_layer_returns_predictions_and_diagnostics(self):
        memory = PoolingMemoryEncoder(token_count=3).encode(self.task.context)
        layer = OVHALayer(
            primitives=self.primitives,
            router=PrimitiveRouter(top_k=None),
            hyper_adapter=LowRankHyperAdapter(enabled=True),
        )

        result = layer.forward(self.task.input_samples, self.task.query_points, memory)

        self.assertEqual(len(result.predictions), len(self.task.query_points))
        self.assertEqual(len(result.primitive_weights), len(self.task.query_points))
        self.assertEqual(set(result.diagnostics), {"adapters", "entropy", "primitive_names"})
        for weights in result.primitive_weights:
            self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_top_k_routing_masks_inactive_primitives(self):
        memory = PoolingMemoryEncoder(token_count=2).encode(self.task.context)
        layer = OVHALayer(
            primitives=self.primitives,
            router=PrimitiveRouter(top_k=1),
            hyper_adapter=LowRankHyperAdapter(enabled=False),
        )

        result = layer.forward(self.task.input_samples, self.task.query_points, memory)

        for weights in result.primitive_weights:
            self.assertEqual(sum(1 for weight in weights if weight > 0.0), 1)
            self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_perceiver_memory_supports_masked_variable_context(self):
        encoder = PerceiverMemoryEncoder(latent_count=2)
        full_memory = encoder.encode(self.task.context)
        masked_memory = encoder.encode(self.task.context, mask=[True, False, True, True])

        self.assertEqual(len(full_memory.tokens), 2)
        self.assertEqual(len(masked_memory.tokens), 2)
        self.assertEqual(masked_memory.summary["context_count"], 3)


if __name__ == "__main__":
    unittest.main()
