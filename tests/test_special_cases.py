import unittest

from moat_ovha.models.memory import MemoryState
from moat_ovha.models.ovha_layer import OVHALayer
from moat_ovha.models.primitives import IdentityValuePrimitive, Sample
from moat_ovha.models.router import PrimitiveRouter
from moat_ovha.theory.special_cases import (
    deep_operator_kernel,
    fourier_kernel,
    graph_local_kernel,
    standard_attention,
)


class SpecialCaseTests(unittest.TestCase):
    def test_identity_primitive_recovers_standard_attention(self):
        samples = [
            Sample(point=0.0, value=1.0),
            Sample(point=1.0, value=3.0),
        ]
        queries = [0.25, 0.75]
        memory = MemoryState(tokens=((0.0, 1.0),), summary={"gain": 1.0, "context_count": 1})
        layer = OVHALayer(
            primitives=(IdentityValuePrimitive(temperature=0.5),),
            router=PrimitiveRouter(top_k=None),
            hyper_adapter=None,
        )

        result = layer.forward(samples, queries, memory)
        expected = [standard_attention(query, samples, temperature=0.5) for query in queries]

        self.assertEqual(result.predictions, expected)

    def test_kernel_helpers_expose_named_special_cases(self):
        self.assertAlmostEqual(fourier_kernel(0.5, 0.5, modes=(1,)), 1.0)
        self.assertGreater(deep_operator_kernel(0.5, 0.5, rank=3), 0.0)
        self.assertGreater(graph_local_kernel(0.5, 0.55, bandwidth=0.2), 0.0)
        self.assertEqual(graph_local_kernel(0.0, 1.0, bandwidth=0.2), 0.0)


if __name__ == "__main__":
    unittest.main()
