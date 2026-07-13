from dataclasses import FrozenInstanceError, replace
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch

import ovha_rod.models.operators as operator_exports
from ovha_rod.models.operators.decoder_bank import (
    OPERATOR_NAMES,
    BankOutput,
    DecoderOperatorBank,
    DecoderOperatorContext,
)
from ovha_rod.models.operators.decoder_contracts import DecoderResidualState
from ovha_rod.models.operators.operator_memory import OperatorMemoryState
from ovha_rod.models.operators.operator_router import OperatorRouterResult
from ovha_rod.models.operators.hyper_adapter import HyperAdapterResult
from ovha_rod.models.operators.rceo import RCEOResult


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
            operator_available=availability,
        ))
        self.assertIn("qsro", dynamic.residuals)
        self.assertFalse(dynamic.residuals["qsro"].valid.any())
        self.assertTrue(torch.equal(
            dynamic.router.weights[..., 0],
            torch.zeros_like(dynamic.router.weights[..., 0]),
        ))

        with self.assertRaisesRegex(ValueError, "relation_role.*qsro"):
            self._bank()(self._context(
                relation_role=None,
                operator_available=availability,
            ))

    def test_requested_operator_without_inputs_and_all_unavailable_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "relation_role.*qsro"):
            self._bank()(self._context(relation_role=None))
        with self.assertRaisesRegex(ValueError, "text.*tq_cato"):
            self._bank()(self._context(text=None, text_valid=None))
        with self.assertRaisesRegex(ValueError, "feature_maps.*ms_tleo"):
            self._bank()(self._context(feature_maps=(), valid_ratios=None))
        with self.assertRaises(RuntimeError):
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

    def test_unavailable_queries_cannot_pollute_primitive_competition(self):
        bank = self._bank(
            enabled_operators=("qsro", "tq_cato"),
            use_router=False,
            use_memory=False,
            use_hyper_adapter=False,
            use_rceo=False,
        )
        available = torch.ones(2, 4, 3, dtype=torch.bool)
        available[..., 2] = False
        available[0, 1, 0] = False
        available[0, 2, 1] = False
        context = self._context(
            feature_maps=(),
            valid_ratios=None,
            operator_available=available,
        )
        reference = bank(context)

        qsro_mutation = context.parent.query.clone()
        qsro_mutation[0, 1] = 1e4
        qsro_parent = replace(context.parent, query=qsro_mutation)
        qsro_changed = bank(replace(context, parent=qsro_parent))
        qsro_valid = reference.residuals["qsro"].valid
        self.assertTrue(torch.allclose(
            qsro_changed.residuals["qsro"].query_delta[qsro_valid],
            reference.residuals["qsro"].query_delta[qsro_valid],
            atol=1e-6,
            rtol=1e-6,
        ))

        tq_mutation = context.parent.query.clone()
        tq_mutation[0, 2] = -1e4
        tq_parent = replace(context.parent, query=tq_mutation)
        tq_changed = bank(replace(context, parent=tq_parent))
        tq_valid = reference.residuals["tq_cato"].valid
        self.assertTrue(torch.allclose(
            tq_changed.residuals["tq_cato"].query_delta[tq_valid],
            reference.residuals["tq_cato"].query_delta[tq_valid],
            atol=1e-6,
            rtol=1e-6,
        ))
        invalid_tq_columns = ~available[..., 1]
        transport = reference.artifacts["tq_cato_transport"]
        self.assertTrue(torch.equal(
            transport.transpose(1, 2)[invalid_tq_columns],
            torch.zeros_like(transport.transpose(1, 2)[invalid_tq_columns]),
        ))

    def test_hyper_adapter_modulation_preserves_zero_primitive_gates(self):
        bank = self._bank()
        with torch.no_grad():
            bank.adapter.coefficients.weight.zero_()
            bank.adapter.coefficients.bias.fill_(1.0)
            bank.adapter.scale_basis.fill_(0.2)
            bank.adapter.shift_basis.fill_(0.2)
            bank.adapter.channel_basis.fill_(0.2)

        output = bank(self._context())

        self.assertGreater(
            output.adaptation.channel_delta[
                self.valid].abs().sum().detach().item(),
            0.0,
        )
        for residual in output.residuals.values():
            self.assertTrue(torch.equal(
                residual.gate_logits, torch.zeros_like(residual.gate_logits)))
        self.assertTrue(torch.equal(output.fused.query, self.parent.query))
        self.assertTrue(torch.equal(
            output.fused.box_logits, self.parent.box_logits))
        self.assertTrue(torch.equal(
            output.fused.referent_score, self.parent.referent_score))

    def test_valid_ratios_change_only_multiscale_evidence_path(self):
        bank = self._bank()
        context = self._context()
        full_ratios = torch.ones_like(self.valid_ratios)
        full = bank(replace(context, valid_ratios=full_ratios))
        with mock.patch.object(
            bank.ms_tleo, "forward", wraps=bank.ms_tleo.forward
        ) as ms_tleo_forward:
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
        self.assertIs(
            ms_tleo_forward.call_args.kwargs["valid_ratios"],
            context.valid_ratios,
        )

    def test_infrastructure_ablation_flags_have_neutral_outputs(self):
        bank = self._bank(
            use_router=False,
            use_memory=False,
            use_hyper_adapter=False,
            use_rceo=False,
        )
        previous = OperatorMemoryState(
            value=torch.randn_like(self.parent.query),
            step=5,
        )
        output = bank(self._context(memory_state=previous))

        expected_weights = torch.full_like(output.router.weights, 1.0 / 3.0)
        expected_weights = torch.where(
            self.valid[..., None], expected_weights, torch.zeros_like(expected_weights))
        self.assertTrue(torch.equal(output.router.weights, expected_weights))
        self.assertTrue(torch.equal(
            output.router.logits, torch.zeros_like(output.router.logits)))
        self.assertEqual(output.memory_state.step, 0)
        self.assertTrue(torch.equal(
            output.memory_state.value, torch.zeros_like(output.memory_state.value)))
        self.assertTrue(torch.equal(
            output.adaptation.scale, torch.zeros_like(output.adaptation.scale)))
        self.assertTrue(torch.equal(
            output.adaptation.shift, torch.zeros_like(output.adaptation.shift)))
        self.assertTrue(torch.equal(
            output.adaptation.channel_delta,
            torch.zeros_like(output.adaptation.channel_delta),
        ))
        self.assertTrue(torch.equal(
            output.reliability.log_prior,
            torch.zeros_like(output.reliability.log_prior),
        ))
        self.assertTrue(torch.equal(
            output.reliability.reliability[self.valid],
            torch.full_like(output.reliability.reliability[self.valid], 0.5),
        ))
        self.assertTrue(torch.equal(
            output.reliability.reliability[~self.valid],
            torch.zeros_like(output.reliability.reliability[~self.valid]),
        ))

    def test_uniform_router_respects_dynamic_operator_availability(self):
        available = torch.tensor([True, False, True])
        output = self._bank(use_router=False)(self._context(
            operator_available=available))

        expected = torch.tensor([0.5, 0.0, 0.5]).expand_as(
            output.router.weights)
        expected = torch.where(
            self.valid[..., None], expected, torch.zeros_like(expected))
        self.assertTrue(torch.equal(output.router.weights, expected))
        self.assertIn("tq_cato", output.residuals)
        self.assertFalse(output.residuals["tq_cato"].valid.any())
        self.assertTrue(torch.equal(
            output.artifacts["tq_cato_transport"],
            torch.zeros_like(output.artifacts["tq_cato_transport"]),
        ))

    def test_dynamic_availability_keeps_static_module_execution_graph(self):
        bank = self._bank()
        availability = torch.ones(2, 4, 3, dtype=torch.bool)
        availability[..., 0] = False
        with mock.patch.object(
            bank.qsro, "forward", wraps=bank.qsro.forward
        ) as qsro_forward:
            output = bank(self._context(operator_available=availability))

        qsro_forward.assert_called_once()
        self.assertEqual(tuple(output.residuals), OPERATOR_NAMES)
        self.assertFalse(output.residuals["qsro"].valid.any())

    def test_inactive_tq_samples_do_not_dilute_scalar_diagnostics(self):
        bank = self._bank(
            enabled_operators=("tq_cato",),
            use_router=False,
            use_memory=False,
            use_hyper_adapter=False,
            use_rceo=False,
        )
        mixed_valid = self.valid.clone()
        mixed_valid[1] = False
        availability = torch.zeros(2, 4, 3, dtype=torch.bool)
        availability[0, :, 1] = mixed_valid[0]
        mixed = bank(self._context(
            valid=mixed_valid,
            relation_role=None,
            feature_maps=(),
            valid_ratios=None,
            operator_available=availability,
        ))
        single_parent = DecoderResidualState(
            query=self.parent.query[:1],
            box_logits=self.parent.box_logits[:1],
            referent_score=self.parent.referent_score[:1],
        )
        single = bank(self._context(
            parent=single_parent,
            boxes=self.boxes[:1],
            valid=mixed_valid[:1],
            relation_role=None,
            text=self.text[:1],
            text_valid=self.text_valid[:1],
            feature_maps=(),
            valid_ratios=None,
            operator_available=availability[:1],
        ))

        for name in ("transport_entropy", "transport_row_error"):
            self.assertTrue(torch.allclose(
                mixed.residuals["tq_cato"].diagnostics[name],
                single.residuals["tq_cato"].diagnostics[name],
                atol=1e-6,
                rtol=1e-6,
            ))

    def test_each_disabled_infrastructure_module_is_not_executed(self):
        cases = (
            ("use_router", "router"),
            ("use_memory", "memory"),
            ("use_hyper_adapter", "adapter"),
            ("use_rceo", "rceo"),
        )
        for flag, module_name in cases:
            with self.subTest(flag=flag):
                bank = self._bank(**{flag: False})
                with mock.patch.object(
                    getattr(bank, module_name),
                    "forward",
                    side_effect=AssertionError(f"{module_name} must stay disabled"),
                ):
                    output = bank(self._context())
                self.assertTrue(torch.equal(
                    output.fused.query, self.parent.query))

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
        self.assertTrue(all(gradient.abs().sum().detach().item() > 0
                            for gradient in gradients[-7:]))

    def test_trainable_boundary_uses_inference_available_inputs_and_fail_fast(self):
        module_doc = inspect.getdoc(sys.modules[DecoderOperatorBank.__module__])
        self.assertIn("trainable", module_doc.lower())
        self.assertNotIn("inference-only", module_doc.lower())
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
        constructor = inspect.signature(DecoderOperatorBank).parameters
        for flag in (
            "use_router", "use_memory", "use_hyper_adapter", "use_rceo"
        ):
            self.assertIn(flag, constructor)
            self.assertIs(constructor[flag].default, True)
        self.assertIs(constructor["debug_contracts"].default, False)

        bad_boxes = self.boxes.clone()
        bad_boxes[0, 0, 0] = float("nan")
        bad_context = self._context(boxes=bad_boxes)
        with self.assertRaisesRegex(ValueError, "boxes.*finite"):
            self._bank(debug_contracts=True)._validate_context(bad_context)
        bad_ratios = self._context(
            valid_ratios=torch.full_like(self.valid_ratios, 1.1))
        with self.assertRaisesRegex(ValueError, "valid_ratios.*range"):
            self._bank(debug_contracts=True)._validate_context(bad_ratios)
        with self.assertRaisesRegex(ValueError, "levels"):
            self._context(valid_ratios=self.valid_ratios[:, :1])
        with self.assertRaisesRegex(ValueError, "text_valid"):
            self._context(text_valid=None)
        with self.assertRaisesRegex(ValueError, "operator_available"):
            self._context(operator_available=torch.ones(2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "operator_available"):
            self._context(operator_available=object())
        with self.assertRaisesRegex(ValueError, "enabled_operators"):
            self._bank(enabled_operators=("unknown",))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._bank(enabled_operators=("qsro", "qsro"))
        for name in ("num_layers", "router_hidden_dim", "adapter_rank"):
            for value in (0, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self._bank(**{name: value})
        for flag in (
            "use_router", "use_memory", "use_hyper_adapter", "use_rceo",
            "debug_contracts",
        ):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, flag):
                    self._bank(**{flag: 1})

    def test_formal_bank_contract_path_has_no_tensor_to_host_checks(self):
        hot_methods = (
            DecoderOperatorContext.__post_init__,
            DecoderOperatorContext._validate_feature_maps,
            BankOutput.__post_init__,
            DecoderOperatorBank.forward,
            DecoderOperatorBank._effective_availability,
        )
        for method in hot_methods:
            source = inspect.getsource(method)
            with self.subTest(method=method.__qualname__):
                self.assertNotIn(".item(", source)
                self.assertNotIn("bool(", source)
                self.assertNotIn("torch.isfinite", source)
        context_source = inspect.getsource(DecoderOperatorContext)
        output_source = inspect.getsource(BankOutput)
        self.assertNotIn("_validate_float_like", context_source)
        self.assertNotIn("_validate_float_like", output_source)

    def test_numeric_output_checks_are_debug_only(self):
        output = self._bank()(self._context())
        bad_router = OperatorRouterResult(
            weights=torch.full_like(output.router.weights, float("nan")),
            logits=torch.zeros_like(output.router.logits),
        )
        unchecked = replace(output, router=bad_router)
        self.assertTrue(torch.isnan(unchecked.router.weights).all())
        with self.assertRaisesRegex(ValueError, "router.*finite"):
            replace(
                output,
                router=bad_router,
                debug_contracts=True,
            )

    def test_result_contract_rejects_invalid_mapping_payloads(self):
        output = self._bank()(self._context())
        with self.assertRaisesRegex(ValueError, "residual"):
            replace(output, residuals={"qsro": object()})
        with self.assertRaisesRegex(ValueError, "artifact"):
            replace(output, artifacts={"bad": object()})
        with self.assertRaisesRegex(ValueError, "availability"):
            replace(output, availability=torch.ones(2, 4, 2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "availability"):
            replace(output, availability=object())
        with self.assertRaisesRegex(ValueError, "debug_contracts"):
            replace(output, debug_contracts=1)

        wrong_residual = replace(
            output.residuals["qsro"],
            query_delta=torch.zeros(2, 3, self.d_model),
            box_delta=torch.zeros(2, 3, 4),
            score_delta=torch.zeros(2, 3),
            gate_logits=torch.zeros(2, 3, 3),
            valid=torch.ones(2, 3, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "residual.*shape"):
            replace(output, residuals={"qsro": wrong_residual})

        wrong_router = OperatorRouterResult(
            weights=torch.zeros(2, 4, 3), logits=torch.zeros(9))
        with self.assertRaisesRegex(ValueError, "router.*shape"):
            replace(output, router=wrong_router)

        bad_reliability = RCEOResult(
            reliability=torch.zeros(2, 4, 2),
            log_prior=torch.zeros(2, 4, 2),
        )
        with self.assertRaisesRegex(ValueError, "reliability.*shape"):
            replace(output, reliability=bad_reliability)
        bad_adaptation = HyperAdapterResult(
            scale=torch.zeros(2, 4, 3, 7),
            shift=torch.zeros(2, 4, 3, 7),
            channel_delta=torch.zeros(2, 4, 3, 3),
        )
        with self.assertRaisesRegex(ValueError, "adaptation.*shape"):
            replace(output, adaptation=bad_adaptation)
        bad_memory = OperatorMemoryState(
            value=torch.zeros(2, 3, self.d_model), step=0)
        with self.assertRaisesRegex(ValueError, "memory_state.*shape"):
            replace(output, memory_state=bad_memory)

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
