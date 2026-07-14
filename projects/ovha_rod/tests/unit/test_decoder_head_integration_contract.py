import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / "ovha_rod/models/detectors/ovha_grounding_dino.py"
HEAD = ROOT / "ovha_rod/models/dense_heads/ovha_grounding_dino_head.py"
PINNED_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"missing {class_name}.{method_name}")


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = _method(path, class_name, method_name)
    return ast.get_source_segment(source, node) or ""


class DecoderDetectorStaticContractTests(unittest.TestCase):
    def test_bank_context_carries_raw_multimodal_and_recurrent_state(self):
        file_source = DETECTOR.read_text(encoding="utf-8")
        forward_transformer = _method_source(
            DETECTOR, "OVHAGroundingDINO", "forward_transformer")
        forward_decoder = _method_source(
            DETECTOR, "OVHAGroundingDINO", "forward_decoder")
        for snippet in (
            "DecoderOperatorBank(**decoder_cfg)",
            "DecoderOperatorContext(",
            "DecoderResidualState(",
            "feature_maps=feature_maps",
            "valid_ratios=valid_ratios",
            "boxes=matching_parent_logits.sigmoid()",
            "fused_query_parent_box_logits=fused_query_parent_box_logits",
            "relation_role=relation_role",
            "text=memory_text",
            "text_valid=text_valid",
            "memory_state=memory_state",
            "memory_state = bank_output.memory_state",
            "bank_output = self.decoder_operator(decoder_context)",
        ):
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, file_source)
        self.assertIn("feature_maps=tuple(img_feats)", forward_transformer)
        self.assertNotIn("self.decoder_operator(\n                query=", forward_decoder)

    def test_forward_transformer_disabled_path_is_direct_parent_call(self):
        source = _method_source(
            DETECTOR, "OVHAGroundingDINO", "forward_transformer")
        self.assertIn("if self.decoder_operator is None:", source)
        self.assertIn("return super().forward_transformer(", source)

    def test_disabled_path_is_direct_parent_call_and_has_no_operator_state(self):
        source = _method_source(
            DETECTOR, "OVHAGroundingDINO", "forward_decoder")
        self.assertIn("if self.decoder_operator is None:", source)
        self.assertIn("return super().forward_decoder(", source)
        self.assertLess(
            source.index("return super().forward_decoder("),
            source.index("for lid, layer in enumerate(self.decoder.layers):"),
        )

    def test_disabled_pre_decoder_does_not_emit_bank_only_arguments(self):
        source = _method_source(
            DETECTOR, "OVHAGroundingDINO", "pre_decoder")
        self.assertIn("if self.decoder_operator is not None:", source)
        self.assertIn("decoder_inputs_dict.update(", source)
        unconditional = source[:source.index("if self.decoder_operator is not None:")]
        self.assertNotIn("relation_role=relation_role", unconditional)
        self.assertNotIn("text_valid=text_token_mask", unconditional)

    def test_referent_score_is_recurrent_without_detach(self):
        source = _method_source(
            DETECTOR, "OVHAGroundingDINO", "forward_decoder")
        initialize = source.index("matching_referent_score = query.new_zeros(")
        parent = source.index("referent_score=matching_referent_score")
        update = source.index(
            "matching_referent_score = bank_output.fused.referent_score")
        self.assertLess(initialize, parent)
        self.assertLess(parent, update)
        recurrence = source[parent:update + len(
            "matching_referent_score = bank_output.fused.referent_score")]
        self.assertNotIn("detach", recurrence)

    def test_operator_diagnostics_are_collected_and_assigned_once(self):
        source = _method_source(
            DETECTOR, "OVHAGroundingDINO", "forward_decoder")
        self.assertIn("self.last_decoder_operator_diagnostics = {}", source)
        self.assertIn("bank_outputs.append(bank_output)", source)
        method = _method(
            DETECTOR, "OVHAGroundingDINO", "forward_decoder")
        assignments = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "summarize_decoder_bank_outputs"
        ]
        self.assertEqual(len(assignments), 1)

    def test_enabled_path_pins_and_reproduces_decoder_loop_without_wrapper(self):
        file_source = DETECTOR.read_text(encoding="utf-8")
        source = _method_source(
            DETECTOR, "OVHAGroundingDINO", "forward_decoder")
        snippets = (
            PINNED_COMMIT,
            "for lid, layer in enumerate(self.decoder.layers):",
            "coordinate_to_encoding(reference_points_input[:, :, 0, :])",
            "query_pos = self.decoder.ref_point_head(query_sine_embed)",
            "reg_branches[lid](query)",
            "reg_branches[lid](fused_query)",
            "inverse_sigmoid(reference_points, eps=1e-3)",
            "reference_points = new_reference_points.detach()",
            "matching_query_count=self.num_queries",
            "operator_box_deltas",
            "operator_referent_scores",
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, file_source if snippet == PINNED_COMMIT else source)
        self.assertNotIn("self.decoder(", source)

    def test_formal_configs_do_not_enable_decoder_operators(self):
        for path in sorted((ROOT / "configs").glob("ovha_rod_swin_t_5e_*.py")):
            with self.subTest(config=path.name):
                source = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("decoder_operator_cfg", source)
                self.assertNotIn("operator_box_deltas", source)


class DecoderHeadStaticContractTests(unittest.TestCase):
    def test_forward_loss_predict_accept_optional_operator_outputs(self):
        expected = {
            "forward": ("operator_box_deltas", "operator_referent_scores"),
            "loss": ("operator_box_deltas", "operator_referent_scores"),
            "predict": ("operator_box_deltas", "operator_referent_scores"),
        }
        for method_name, optional_names in expected.items():
            method = _method(HEAD, "OVHAGroundingDINOHead", method_name)
            arguments = method.args.args
            defaults = [None] * (len(arguments) - len(method.args.defaults))
            defaults.extend(method.args.defaults)
            by_name = {
                argument.arg: default
                for argument, default in zip(arguments, defaults)
            }
            for optional_name in optional_names:
                with self.subTest(method=method_name, name=optional_name):
                    self.assertIn(optional_name, by_name)
                    self.assertIsInstance(by_name[optional_name], ast.Constant)
                    self.assertIsNone(by_name[optional_name].value)

    def test_forward_none_path_delegates_exactly_to_parent(self):
        source = _method_source(
            HEAD, "OVHAGroundingDINOHead", "forward")
        self.assertIn(
            "if operator_box_deltas is None and operator_referent_scores is None:",
            source,
        )
        self.assertIn("return super().forward(", source)


if __name__ == "__main__":
    unittest.main()
