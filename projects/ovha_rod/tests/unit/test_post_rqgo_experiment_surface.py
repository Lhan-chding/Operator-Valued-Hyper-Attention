import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "post_rqgo"


EXPECTED_VARIANTS = {
    "full": {
        "enabled_operators": ("qsro", "tq_cato", "ms_tleo"),
        "use_router": True,
        "use_memory": True,
        "use_hyper_adapter": True,
        "use_rceo": True,
    },
    "qsro_only": {
        "enabled_operators": ("qsro",),
        "use_router": True,
        "use_memory": True,
        "use_hyper_adapter": True,
        "use_rceo": True,
    },
    "tq_cato_only": {
        "enabled_operators": ("tq_cato",),
        "use_router": True,
        "use_memory": True,
        "use_hyper_adapter": True,
        "use_rceo": True,
    },
    "ms_tleo_only": {
        "enabled_operators": ("ms_tleo",),
        "use_router": True,
        "use_memory": True,
        "use_hyper_adapter": True,
        "use_rceo": True,
    },
    "no_router": {
        "enabled_operators": ("qsro", "tq_cato", "ms_tleo"),
        "use_router": False,
        "use_memory": True,
        "use_hyper_adapter": True,
        "use_rceo": True,
    },
    "no_memory": {
        "enabled_operators": ("qsro", "tq_cato", "ms_tleo"),
        "use_router": True,
        "use_memory": False,
        "use_hyper_adapter": True,
        "use_rceo": True,
    },
    "no_hyper_adapter": {
        "enabled_operators": ("qsro", "tq_cato", "ms_tleo"),
        "use_router": True,
        "use_memory": True,
        "use_hyper_adapter": False,
        "use_rceo": True,
    },
    "no_rceo": {
        "enabled_operators": ("qsro", "tq_cato", "ms_tleo"),
        "use_router": True,
        "use_memory": True,
        "use_hyper_adapter": True,
        "use_rceo": False,
    },
}


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


class PostRQGOExperimentSurfaceTests(unittest.TestCase):
    def test_all_complete_and_ablation_configs_exist(self):
        observed = {
            path.stem.removeprefix("ovha_rod_swin_t_5e_refcoco_bank_")
            for path in CONFIG_ROOT.glob(
                "ovha_rod_swin_t_5e_refcoco_bank_*.py")
        }
        self.assertEqual(observed, set(EXPECTED_VARIANTS))

    def test_each_config_declares_a_complete_decoder_operator_contract(self):
        required = {
            "enabled",
            "enabled_operators",
            "router_hidden_dim",
            "adapter_rank",
            "use_router",
            "use_memory",
            "use_hyper_adapter",
            "use_rceo",
        }
        for name, expected in EXPECTED_VARIANTS.items():
            path = CONFIG_ROOT / (
                f"ovha_rod_swin_t_5e_refcoco_bank_{name}.py")
            with self.subTest(name=name):
                model = _literal_assignment(path, "model")
                contract = model["decoder_operator_cfg"]
                self.assertEqual(set(contract), required)
                self.assertIs(contract["enabled"], True)
                self.assertEqual(contract["router_hidden_dim"], 128)
                self.assertEqual(contract["adapter_rank"], 16)
                for key, value in expected.items():
                    self.assertEqual(contract[key], value)

    def test_configs_inherit_the_locked_refcoco_rqgo_protocol(self):
        expected_base = "../ovha_rod_swin_t_5e_refcoco.py"
        for path in sorted(CONFIG_ROOT.glob("*.py")):
            with self.subTest(path=path):
                self.assertEqual(_literal_assignment(path, "_base_"),
                                 expected_base)

    def test_configs_add_decoder_operator_optimizer_group(self):
        for path in sorted(CONFIG_ROOT.glob("*.py")):
            with self.subTest(path=path):
                optim = _literal_assignment(path, "optim_wrapper")
                custom_keys = optim["paramwise_cfg"]["custom_keys"]
                self.assertEqual(
                    custom_keys["decoder_operator"], {"lr_mult": 1.0})

    def test_formal_phase1_configs_keep_decoder_bank_disabled(self):
        for path in sorted((ROOT / "configs").glob(
                "ovha_rod_swin_t_5e_*.py")):
            with self.subTest(path=path):
                self.assertNotIn("decoder_operator_cfg", path.read_text())

    def test_post_rqgo_configs_do_not_introduce_oracle_or_cache_inputs(self):
        forbidden = (
            "proposal_cache",
            "region_features.npy",
            "ground_truth_box",
            "gt_box",
            "clip_crop",
            "oracle_candidate",
        )
        source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in sorted(CONFIG_ROOT.glob("*.py")))
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_diagnostics_track_decoder_bank_gradients_and_scalars(self):
        source = (
            ROOT / "ovha_rod/hooks/operator_diagnostics_hook.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_decoder_operator_grad_squared", source)
        self.assertIn("decoder_operator_gradient_norm", source)
        self.assertIn("last_decoder_operator_diagnostics", source)
        self.assertIn("non-finite decoder operator gradient norm", source)

    def test_documentation_marks_cuda_acceptance_as_pending(self):
        documentation = (
            (ROOT / "README.md").read_text(encoding="utf-8")
            + (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
        ).lower()
        self.assertIn("post-rqgo", documentation)
        self.assertIn("cuda", documentation)
        self.assertIn("pending", documentation)


if __name__ == "__main__":
    unittest.main()
