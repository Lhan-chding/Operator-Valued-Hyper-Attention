import ast
import unittest
from pathlib import Path

import torch

from ovha_rod.models.scoring import masked_expression_score, referent_focal_loss


ROOT = Path(__file__).resolve().parents[2]


class ScoringTests(unittest.TestCase):
    def test_expression_score_ignores_padding(self):
        logits = torch.tensor([[[1.0, 2.0, 999.0], [0.0, 0.0, -999.0]]])
        valid = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        score = masked_expression_score(logits, valid, temperature=1.0)
        expected = torch.logsumexp(logits[..., :2], dim=-1)
        self.assertTrue(torch.allclose(score, expected))

    def test_referent_focal_loss_is_finite(self):
        score = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
        positive = torch.tensor([[1, 0, 0]], dtype=torch.bool)
        loss = referent_focal_loss(score, positive)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(score.grad.abs().sum()), 0.0)


class StaticIntegrationContractTests(unittest.TestCase):
    def test_required_phase1_files_exist_and_compile(self):
        required = [
            ROOT / "ovha_rod/models/detectors/ovha_grounding_dino.py",
            ROOT / "ovha_rod/models/detectors/deterministic_grounding_dino.py",
            ROOT / "ovha_rod/models/positional_encoding.py",
            ROOT / "ovha_rod/models/dense_heads/ovha_grounding_dino_head.py",
            ROOT / "configs/ovha_rod_swin_t_5e_refcoco.py",
            ROOT / "configs/ovha_rod_swin_t_5e_refcoco_plus.py",
            ROOT / "configs/ovha_rod_swin_t_5e_refcocog.py",
            ROOT / "scripts/server_preflight.py",
            ROOT / "scripts/run_phase1_server.sh",
            ROOT / "scripts/two_batch_smoke.py",
            ROOT / "scripts/gpu_guard.py",
            ROOT / "scripts/port_guard.py",
            ROOT / "scripts/prepare_work_dir.py",
            ROOT / "scripts/resume_guard.py",
            ROOT / "scripts/run_lock.py",
            ROOT / "ovha_rod/runtime_contracts.py",
            ROOT / "ovha_rod/hooks/checkpoint_provenance_hook.py",
            ROOT / "README.md",
            ROOT / "environment/mmdetection.lock",
        ]
        for path in required:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)
                if path.suffix == ".py":
                    ast.parse(path.read_text())

    def test_configs_are_val_only_and_do_not_use_cached_candidates(self):
        for path in sorted((ROOT / "configs").glob("*.py")):
            source = path.read_text().lower()
            with self.subTest(path=path):
                self.assertNotIn("testa", source)
                self.assertNotIn("testb", source)
                self.assertNotIn("region_features.npy", source)
                self.assertNotIn("proposal_cache", source)
                self.assertNotIn("clip crop", source)
                self.assertIn("max_epochs=5", source.replace(" ", ""))
                self.assertIn("backbone=dict(init_cfg=none)",
                              source.replace(" ", ""))
                self.assertIn("load_from = none", source)

    def test_detector_keeps_parent_token_max_for_topk(self):
        source = (ROOT / "ovha_rod/models/detectors/ovha_grounding_dino.py").read_text()
        self.assertIn("enc_outputs_class.max(-1)[0]", source)
        self.assertNotIn("logsumexp(enc_outputs_class", source)
        self.assertIn(
            "valid = memory_valid_mask(output_memory, memory_mask)", source)

    def test_dense_head_imports_instance_list_from_pinned_mmdet_api(self):
        source = (
            ROOT / "ovha_rod/models/dense_heads/ovha_grounding_dino_head.py"
        ).read_text()
        self.assertIn("from mmdet.structures import SampleList", source)
        self.assertIn("from mmdet.utils import InstanceList", source)
        self.assertNotIn(
            "from mmdet.structures import InstanceList", source)

    def test_two_batch_smoke_pins_cublas_before_mmengine_import(self):
        smoke = (ROOT / "scripts/two_batch_smoke.py").read_text()
        assignment = (
            'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"')
        self.assertIn(assignment, smoke)
        self.assertLess(smoke.index("import os"), smoke.index(assignment))
        self.assertLess(smoke.index(assignment), smoke.index("from mmengine.config"))
        self.assertIn("ALLOWED_CFG_OPTIONS", smoke)
        self.assertIn("require_selected_gpus_idle", smoke)
        self.assertIn("validate_locked_checkpoint", smoke)
        self.assertIn("validate_local_bert", smoke)
        self.assertIn("prepare_fresh_private_work_dir", smoke)
        self.assertIn("sys.path.insert", smoke)
        self.assertLess(smoke.index("sys.path.insert"),
                        smoke.index("from ovha_rod.runtime_contracts"))
        self.assertIn('"train_dataloader.persistent_workers"', smoke)
        self.assertIn('"optim_wrapper.accumulative_counts"', smoke)
        self.assertIn('"custom_hooks.0.warmup_iters"', smoke)
        self.assertIn("Phase 1 smoke requires an OVHA config", smoke)
        self.assertGreaterEqual(smoke.count("require_selected_gpus_idle"), 3)
        self.assertIn('os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"', smoke)
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', smoke)
        self.assertIn(
            'parser.add_argument("--batch-size",', smoke)
        self.assertIn("choices=(1, 2, 4, 8, 16, 32)", smoke)
        self.assertIn(
            'config.train_dataloader.batch_size = args.batch_size', smoke)
        self.assertNotIn('"train_dataloader.batch_size",', smoke)
        self.assertIn("runner.train_dataloader.batch_size", smoke)
        self.assertIn('"max_memory_reserved_bytes"', smoke)
        self.assertIn('config.train_dataloader.num_workers = 0', smoke)
        self.assertIn('config.train_dataloader.persistent_workers = False', smoke)
        self.assertIn('config.optim_wrapper.type = "OptimWrapper"', smoke)
        self.assertIn('hook.get("type") != "CheckpointProvenanceHook"', smoke)
        self.assertIn("tokenizer root must equal the locked local BERT root", smoke)

    def test_two_batch_smoke_uses_full_precision(self):
        smoke = (ROOT / "scripts/two_batch_smoke.py").read_text()
        self.assertNotIn("AmpOptimWrapper", smoke)
        self.assertNotIn("bfloat16", smoke)
        self.assertNotIn('"optim_wrapper.loss_scale",', smoke)
        self.assertIn('"precision": "fp32"', smoke)
        self.assertIn('"amp_enabled": False', smoke)

    def test_two_batch_smoke_disables_validation_loop(self):
        smoke = (ROOT / "scripts/two_batch_smoke.py").read_text()
        runner_build = smoke.index("runner = Runner.from_cfg(config)")
        for assignment in (
                "config.val_cfg = None",
                "config.val_dataloader = None",
                "config.val_evaluator = None"):
            with self.subTest(assignment=assignment):
                self.assertIn(assignment, smoke)
                self.assertLess(smoke.index(assignment), runner_build)

    def test_server_runner_exports_cublas_before_launch_commands(self):
        runner = (ROOT / "scripts/run_phase1_server.sh").read_text()
        export = 'export CUBLAS_WORKSPACE_CONFIG=":4096:8"'
        self.assertIn(export, runner)
        self.assertLess(runner.index(export), runner.index("PREFLIGHT=("))
        self.assertLess(runner.index(export), runner.index("COMMAND=("))
        self.assertIn("randomness.deterministic=True", runner)
        self.assertIn("CUDA_VISIBLE_DEVICES", runner)
        self.assertIn("CUDA_DEVICE_ORDER", runner)
        self.assertIn("gpu_guard.py", runner)
        self.assertIn("prepare_work_dir.py", runner)
        self.assertGreaterEqual(runner.count("gpu_guard.py"), 2)
        self.assertIn("--master-port", runner)
        self.assertIn("port_guard.py", runner)
        self.assertIn("--resume", runner)
        self.assertIn("run_lock.py", runner)
        self.assertIn("default_hooks.checkpoint.by_epoch=True", runner)
        self.assertIn("default_hooks.checkpoint.interval=1", runner)
        self.assertIn("default_hooks.checkpoint.save_last=True", runner)

        documentation = (
            (ROOT / "README.md").read_text()
            + (ROOT / "RUNBOOK.md").read_text()
        )
        self.assertIn("CUDA_VISIBLE_DEVICES", documentation)
        self.assertIn("epoch-boundary resume", documentation.lower())

    def test_server_runner_uses_fail_fast_full_precision(self):
        runner = (ROOT / "scripts/run_phase1_server.sh").read_text()
        self.assertNotIn("AmpOptimWrapper", runner)
        self.assertNotIn("bfloat16", runner)
        self.assertNotIn("optim_wrapper.loss_scale", runner)
        self.assertIn(
            '"optim_wrapper.clip_grad.error_if_nonfinite=True"', runner)
        self.assertIn('--expected-identity "amp=false"', runner)
        self.assertIn('--expected-identity "amp_dtype=none"', runner)

    def test_server_resume_identity_locks_amp_dtype(self):
        runner = (ROOT / "scripts/run_phase1_server.sh").read_text()
        self.assertIn('--expected-identity "amp_dtype=', runner)

    def test_metric_requires_encoder_oracle_for_every_phase1_sample(self):
        source = (
            ROOT / "ovha_rod/evaluation/ovha_refexp_metric.py").read_text()
        self.assertIn(
            "if len(encoder_ious[name]) != len(values):", source)
        self.assertNotIn(
            "if encoder_ious[name] and len(encoder_ious[name])", source)
        self.assertIn("non-finite prediction boxes", source)
        self.assertIn("unexpected RefExp dataset_name", source)

    def test_full_training_aborts_on_nonfinite_diagnostics(self):
        source = (
            ROOT / "ovha_rod/hooks/operator_diagnostics_hook.py").read_text()
        raise_index = source.index("raise FloatingPointError")
        interval_index = source.index("(runner.iter + 1) % self.interval")
        self.assertLess(raise_index, interval_index)

    def test_preflight_requires_complete_backbone_checkpoint(self):
        source = (ROOT / "scripts/server_preflight.py").read_text()
        self.assertIn("checkpoint_backbone_key_coverage", source)
        self.assertIn("missing_backbone", source)
        self.assertIn("pre_decoder_none_memory_mask", source)

    def test_environment_is_pinned_to_a_commit(self):
        lock = (ROOT / "environment/mmdetection.lock").read_text()
        self.assertIn("cfd5d3a985b0249de009b67d04f37263e11cdf3d", lock)
        self.assertNotIn("@main", lock)

    def test_phase0_parent_configs_preserve_official_optimizer_schedule(self):
        for dataset in ("refcoco", "refcoco_plus", "refcocog"):
            path = ROOT / "configs" / f"phase0_parent_swin_t_5e_{dataset}.py"
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)
                source = path.read_text()
                self.assertNotIn("optim_wrapper", source)
                self.assertNotIn("param_scheduler", source)
                self.assertIn(
                    "backbone=dict(init_cfg=None)",
                    source.replace(" ", ""),
                )
                self.assertIn("OperatorDiagnosticsHook", source)

    def test_all_configs_use_deterministic_grounding_dino(self):
        for path in sorted((ROOT / "configs").glob("*swin_t_5e_*.py")):
            tree = ast.parse(path.read_text())
            model_call = next(
                node.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "model"
                    for target in node.targets
                )
            )
            model_type = next(
                ast.literal_eval(keyword.value)
                for keyword in model_call.keywords
                if keyword.arg == "type"
            )
            expected = (
                "OVHAGroundingDINO"
                if path.name.startswith("ovha_rod_")
                else "DeterministicGroundingDINO"
            )
            with self.subTest(path=path):
                self.assertEqual(model_type, expected)

    def test_preflight_requires_deterministic_positional_encoding(self):
        source = (ROOT / "scripts/server_preflight.py").read_text()
        self.assertIn("deterministic_positional_encoding", source)
        smoke = (ROOT / "scripts/two_batch_smoke.py").read_text()
        self.assertIn("DeterministicSinePositionalEncoding", smoke)
        self.assertIn("are_deterministic_algorithms_enabled", smoke)
        self.assertIn("deterministic positional CUDA probe", smoke)


if __name__ == "__main__":
    unittest.main()
