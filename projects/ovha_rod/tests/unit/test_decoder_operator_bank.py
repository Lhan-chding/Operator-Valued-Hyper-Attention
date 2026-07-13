from dataclasses import FrozenInstanceError, replace
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

import ovha_rod.models.operators as operator_exports
from ovha_rod.models.operators.decoder_bank import (
    OPERATOR_NAMES,
    BankOutput,
    DecoderOperatorBank,
    DecoderOperatorContext,
)
from ovha_rod.models.operators.decoder_contracts import DecoderResidualState


ROOT = Path(__file__).resolve().parents[2]


class DecoderOperatorBankTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(83)
        self.d_model = 8
        self.parent = DecoderResidualState(
            query=torch.randn(2, 4, self.d_model),
            box_logits=torch.randn(2, 4, 4),
            referent_score=torch.randn(2, 4),
        )
        self.boxes = torch.tensor(
            [
                [[0.20, 0.25, 0.18, 0.20], [0.70, 0.28, 0.16, 0.22],
                 [0.35, 0.72, 0.25, 0.18], [0.82, 0.78, 0.12, 0.14]],
                [[0.15, 0.20, 0.10, 0.12], [0.48, 0.45, 0.30, 0.24],
                 [0.75, 0.65, 0.14, 0.20], [0.55, 0.85, 0.20, 0.10]],
            ]
        )
        self.valid = torch.tensor(
            [[True, True, True, False], [True, True, False, False]])
        self.relation_role = torch.randn(2, self.d_model)
        self.text = torch.randn(2, 5, self.d_model)
        self.text_valid = torch.tensor(
            [[True, False, True, True, False],
             [False, True, True, False, True]],
        )
        self.feature_maps = (
            torch.randn(2, self.d_model, 8, 8),
            torch.randn(2, self.d_model, 4, 4),
        )
        self.valid_ratios = torch.tensor(
            [[[1.0, 1.0], [0.75, 1.0]], [[0.875, 0.75], [1.0, 1.0]]],
        )

    def _context(self, **changes) -> DecoderOperatorContext:
        values = dict(
            parent=self.parent,
            boxes=self.boxes,
            valid=self.valid,
            layer_index=2,
            relation_role=self.relation_role,
            text=self.text,
            text_valid=self.text_valid,
            feature_maps=self.feature_maps,
            valid_ratios=self.valid_ratios,
        )
        values.update(changes)
        return DecoderOperatorContext(**values)

    def _bank(self, **changes) -> DecoderOperatorBank:
        values = dict(
            d_model=self.d_model,
            num_layers=6,
            router_hidden_dim=12,
            adapter_rank=2,
        )
        values.update(changes)
        return DecoderOperatorBank(**values)

    def test_frozen_context_and_output_copy_collection_boundaries(self):
        feature_maps = list(self.feature_maps)
        context = self._context(feature_maps=feature_maps)
        feature_maps.append(torch.randn(2, self.d_model, 2, 2))

        self.assertEqual(len(context.feature_maps), 2)
        self.assertIsInstance(context.feature_maps, tuple)
        with self.assertRaises(FrozenInstanceError):
            context.layer_index = 4

        output = self._bank()(context)
        self.assertIsInstance(output, BankOutput)
        with self.assertRaises(FrozenInstanceError):
            output.fused = context.parent
        with self.assertRaises(TypeError):
            output.residuals["late"] = output.residuals["qsro"]
        with self.assertRaises(TypeError):
            output.artifacts["late"] = torch.tensor(1.0)

    def test_default_bank_composes_all_operators_as_exact_parent_noop(self):
        context = self._context()
        output = self._bank()(context)

        self.assertEqual(tuple(output.residuals), OPERATOR_NAMES)
        self.assertEqual(output.memory_state.step, 1)
        self.assertEqual(tuple(output.router.weights.shape), (2, 4, 3))
        self.assertTrue(torch.allclose(
            output.router.weights[self.valid].sum(-1),
            torch.ones_like(output.router.weights[self.valid].sum(-1)),
        ))
        self.assertIn("tq_cato_transport", output.artifacts)
        self.assertTrue(torch.equal(output.fused.query, context.parent.query))
        self.assertTrue(torch.equal(
            output.fused.box_logits, context.parent.box_logits))
        self.assertTrue(torch.equal(
            output.fused.referent_score, context.parent.referent_score))
        self.assertTrue(torch.equal(
            output.fused.query[~self.valid], context.parent.query[~self.valid]))

    def test_ablation_and_dynamic_availability_skip_unavailable_inputs(self):
        bank = self._bank(enabled_operators=("tq_cato",))
        context = self._context(
            relation_role=None,
            feature_maps=(),
            valid_ratios=None,
        )
        output = bank(context)

        self.assertEqual(tuple(output.residuals), ("tq_cato",))
        self.assertTrue(torch.equal(
            output.availability[..., 0],
            torch.zeros_like(output.availability[..., 0]),
        ))
        self.assertTrue(torch.equal(
            output.availability[..., 2],
            torch.zeros_like(output.availability[..., 2]),
        ))

        availability = torch.tensor([False, True, True])
        dynamic = self._bank()(self._context(
            relation_role=None,
            operator_available=availability,
        ))
        self.assertNotIn("qsro", dynamic.residuals)
        self.assertTrue(torch.equal(
            dynamic.router.weights[..., 0],
            torch.zeros_like(dynamic.router.weights[..., 0]),
        ))

    def test_requested_operator_without_inputs_and_all_unavailable_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "relation_role.*qsro"):
            self._bank()(self._context(relation_role=None))
        with self.assertRaisesRegex(ValueError, "text.*tq_cato"):
            self._bank()(self._context(text=None, text_valid=None))
        with self.assertRaisesRegex(ValueError, "feature_maps.*ms_tleo"):
            self._bank()(self._context(feature_maps=(), valid_ratios=None))
        with self.assertRaisesRegex(ValueError, "available operator"):
            self._bank()(self._context(
                operator_available=torch.zeros(3, dtype=torch.bool)))

    def test_per_query_availability_masks_residual_and_router_exactly(self):
        availability = torch.ones(2, 4, 3, dtype=torch.bool)
        availability[0, 1, 0] = False
        availability[1, 0, 2] = False
        output = self._bank()(self._context(operator_available=availability))

        effective = availability & self.valid[..., None]
        self.assertTrue(torch.equal(output.availability, effective))
        self.assertEqual(
            int(torch.count_nonzero(output.router.weights[~effective])), 0)
        self.assertTrue(torch.equal(
            output.residuals["qsro"].query_delta[0, 1],
            torch.zeros(self.d_model),
        ))
        self.assertTrue(torch.equal(
            output.residuals["ms_tleo"].box_delta[1, 0],
            torch.zeros(4),
        ))

    def test_valid_ratios_change_only_multiscale_evidence_path(self):
        bank = self._bank()
        context = self._context()
        full_ratios = torch.ones_like(self.valid_ratios)
        full = bank(replace(context, valid_ratios=full_ratios))
        cropped = bank(context)

        self.assertFalse(torch.allclose(
            full.residuals["ms_tleo"].query_delta,
            cropped.residuals["ms_tleo"].query_delta,
        ))
        self.assertTrue(torch.allclose(
            full.residuals["qsro"].query_delta,
            cropped.residuals["qsro"].query_delta,
        ))
        self.assertTrue(torch.isfinite(
            cropped.artifacts["valid_ratio_mean"]))

    def test_query_permutation_is_equivariant_across_complete_bank(self):
        bank = self._bank()
        context = self._context()
        original = bank(context)
        order = torch.tensor([2, 0, 3, 1])
        permuted_parent = DecoderResidualState(
            query=context.parent.query[:, order],
            box_logits=context.parent.box_logits[:, order],
            referent_score=context.parent.referent_score[:, order],
        )
        permuted = bank(replace(
            context,
            parent=permuted_parent,
            boxes=context.boxes[:, order],
            valid=context.valid[:, order],
        ))

        self.assertTrue(torch.allclose(
            permuted.fused.query, original.fused.query[:, order], atol=1e-6))
        self.assertTrue(torch.allclose(
            permuted.router.weights,
            original.router.weights[:, order],
            atol=1e-6,
        ))
        for name in OPERATOR_NAMES:
            self.assertTrue(torch.allclose(
                permuted.residuals[name].query_delta,
                original.residuals[name].query_delta[:, order],
                atol=1e-6,
            ))
        self.assertTrue(torch.allclose(
            permuted.artifacts["tq_cato_transport"],
            original.artifacts["tq_cato_transport"][:, :, order],
            atol=1e-6,
        ))

    def test_nonzero_operator_gates_backward_is_finite(self):
        bank = self._bank()
        with torch.no_grad():
            bank.qsro.gate_head.bias.fill_(0.2)
            bank.tq_cato.gate_head.bias.fill_(0.2)
            bank.ms_tleo.gate_head.weight.fill_(0.01)
            bank.adapter.scale_basis.fill_(0.01)
            bank.adapter.shift_basis.fill_(0.01)
            bank.adapter.channel_basis.fill_(0.01)
            bank.rceo.output.weight.normal_(std=0.01)

        query = self.parent.query.clone().requires_grad_(True)
        boxes = self.boxes.clone().requires_grad_(True)
        text = self.text.clone().requires_grad_(True)
        features = tuple(value.clone().requires_grad_(True)
                         for value in self.feature_maps)
        parent = DecoderResidualState(
            query=query,
            box_logits=self.parent.box_logits.clone().requires_grad_(True),
            referent_score=self.parent.referent_score.clone().requires_grad_(True),
        )
        output = bank(self._context(
            parent=parent,
            boxes=boxes,
            text=text,
            feature_maps=features,
        ))
        loss = (
            output.fused.query.square().mean()
            + output.fused.box_logits.square().mean()
            + output.fused.referent_score.square().mean()
            + output.router.weights.square().mean()
            + output.memory_state.value.square().mean()
            + output.reliability.log_prior.square().mean()
        )
        loss.backward()

        gradients = (
            query.grad,
            boxes.grad,
            text.grad,
            *(feature.grad for feature in features),
            bank.qsro.gate_head.bias.grad,
            bank.tq_cato.gate_head.bias.grad,
            bank.ms_tleo.gate_head.weight.grad,
            bank.router.network[-1].weight.grad,
            bank.memory.candidate.weight.grad,
            bank.adapter.scale_basis.grad,
            bank.rceo.output.weight.grad,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all()
                            for gradient in gradients))

    def test_public_boundary_is_inference_only_and_fail_fast(self):
        self.assertEqual(
            set(inspect.signature(DecoderOperatorBank.forward).parameters),
            {"self", "context"},
        )
        self.assertEqual(
            set(inspect.signature(DecoderOperatorContext).parameters),
            {
                "parent", "boxes", "valid", "layer_index", "relation_role",
                "text", "text_valid", "feature_maps", "valid_ratios",
                "memory_state", "operator_available",
            },
        )

        bad_boxes = self.boxes.clone()
        bad_boxes[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "boxes.*finite"):
            self._context(boxes=bad_boxes)
        with self.assertRaisesRegex(ValueError, "valid_ratios.*range"):
            self._context(valid_ratios=torch.full_like(self.valid_ratios, 1.1))
        with self.assertRaisesRegex(ValueError, "levels"):
            self._context(valid_ratios=self.valid_ratios[:, :1])
        with self.assertRaisesRegex(ValueError, "text_valid"):
            self._context(text_valid=None)
        with self.assertRaisesRegex(ValueError, "operator_available"):
            self._context(operator_available=torch.ones(2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "enabled_operators"):
            self._bank(enabled_operators=("unknown",))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._bank(enabled_operators=("qsro", "qsro"))

    def test_result_contract_rejects_invalid_mapping_payloads(self):
        output = self._bank()(self._context())
        with self.assertRaisesRegex(ValueError, "residual"):
            replace(output, residuals={"qsro": object()})
        with self.assertRaisesRegex(ValueError, "artifact"):
            replace(output, artifacts={"bad": object()})
        with self.assertRaisesRegex(ValueError, "availability"):
            replace(output, availability=torch.ones(2, 4, 2, dtype=torch.bool))

    def test_lazy_exports_and_formal_integration_remain_isolated(self):
        self.assertIs(operator_exports.DecoderOperatorBank, DecoderOperatorBank)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        script = (
            "import sys; import ovha_rod; "
            "assert 'ovha_rod.models.operators.decoder_bank' not in sys.modules; "
            "from ovha_rod.models.operators import DecoderOperatorBank; "
            "assert 'ovha_rod.models.operators.decoder_bank' in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        for config in sorted((ROOT / "configs").glob("ovha_rod_swin_t_5e_*.py")):
            source = config.read_text(encoding="utf-8").lower()
            self.assertNotIn("decoderoperatorbank", source)
            self.assertNotIn("decoder_bank", source)


if __name__ == "__main__":
    unittest.main()
