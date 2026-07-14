import unittest

import torch

from ovha_rod.models.operators.operator_memory import (
    OperatorMemory,
    OperatorMemoryState,
)


class OperatorMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.memory = OperatorMemory(d_model=8)
        self.query = torch.randn(2, 4, 8, requires_grad=True)
        self.valid = torch.tensor(
            [[True, True, False, True], [True, False, True, False]])

    def test_initialize_and_update_are_immutable_and_masked(self):
        state = self.memory.initialize(
            batch_size=2,
            query_count=4,
            device=self.query.device,
            dtype=self.query.dtype,
        )
        before = state.value.clone()
        updated = self.memory(state, self.query, self.valid)

        self.assertIsInstance(state, OperatorMemoryState)
        self.assertEqual(state.step, 0)
        self.assertEqual(updated.step, 1)
        self.assertTrue(torch.equal(state.value, before))
        self.assertTrue(torch.equal(
            updated.value[~self.valid], state.value[~self.valid]))
        self.assertGreater(updated.value[self.valid].abs().sum().item(), 0.0)
        with self.assertRaisesRegex(Exception, "cannot assign"):
            state.step = 9

    def test_sequential_update_is_finite_and_backpropagates(self):
        state = self.memory.initialize_like(self.query)
        first = self.memory(state, self.query, self.valid)
        second = self.memory(first, self.query * 0.5, self.valid)
        loss = second.value.square().mean()
        loss.backward()

        self.assertEqual(second.step, 2)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.query.grad)
        self.assertTrue(torch.isfinite(self.query.grad).all())

    def test_shape_mask_and_nonfinite_state_fail_fast(self):
        state = self.memory.initialize_like(self.query)
        with self.assertRaisesRegex(ValueError, "query"):
            self.memory(state, self.query[:, :3], self.valid[:, :3])
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.memory(state, self.query, self.valid.float())
        with self.assertRaisesRegex(ValueError, "finite"):
            OperatorMemoryState(
                value=torch.full((1, 2, 8), float("inf")), step=0)

    def test_constructor_initialization_and_state_boundaries_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "d_model"):
            OperatorMemory(d_model=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.memory.initialize(0, 2, self.query.device, self.query.dtype)
        with self.assertRaisesRegex(ValueError, "floating"):
            self.memory.initialize(1, 2, self.query.device, torch.int64)
        with self.assertRaisesRegex(ValueError, "shape"):
            OperatorMemoryState(value=torch.zeros(2, 8), step=0)
        with self.assertRaisesRegex(ValueError, "floating"):
            OperatorMemoryState(
                value=torch.zeros(1, 2, 8, dtype=torch.int64), step=0)
        with self.assertRaisesRegex(ValueError, "step"):
            OperatorMemoryState(value=torch.zeros(1, 2, 8), step=-1)

        state = self.memory.initialize_like(self.query)
        with self.assertRaisesRegex(ValueError, "device and dtype"):
            self.memory(state, self.query.double(), self.valid)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.memory.initialize_like(torch.zeros(2, 4, 7))
        with self.assertRaisesRegex(ValueError, "floating"):
            self.memory.initialize_like(torch.zeros(2, 4, 8, dtype=torch.int64))
        bad = self.query.detach().clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.memory.initialize_like(bad)


if __name__ == "__main__":
    unittest.main()
