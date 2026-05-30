import json
import tempfile
import unittest
from pathlib import Path


class PublicAdapterDiagnosticsTests(unittest.TestCase):
    def test_summarize_reports_adapter_effect_by_family_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rows(
                root / "seed_81_ovha_full" / "eval_metrics" / "ovha_full" / "seed_81.jsonl",
                [
                    _eval_row("pdebench_burgers_1d", "iid", "ovha_full", 0.30),
                    _eval_row("pdebench_darcy_2d", "iid", "ovha_full", 0.60),
                ],
            )
            _write_rows(
                root / "seed_81_ovha_no_hyper_adapter" / "eval_metrics" / "ovha_no_hyper_adapter" / "seed_81.jsonl",
                [
                    _eval_row("pdebench_burgers_1d", "iid", "ovha_no_hyper_adapter", 0.25),
                    _eval_row("pdebench_darcy_2d", "iid", "ovha_no_hyper_adapter", 0.80),
                ],
            )
            _write_rows(
                root / "seed_81_ovha_full" / "diagnostics" / "ovha_full" / "seed_81.jsonl",
                [
                    _diagnostic_row("pdebench_burgers_1d", "iid", "ovha_full", 1.2, 0.7),
                    _diagnostic_row("pdebench_darcy_2d", "iid", "ovha_full", 0.8, 0.2),
                ],
            )

            from scripts.summarize_public_adapter_diagnostics import summarize

            report_path = summarize(root)
            report = report_path.read_text()

        self.assertIn("| pdebench_burgers_1d | iid | 0.300000 | 0.250000 | -0.050000 | adapter_hurts |", report)
        self.assertIn("| pdebench_darcy_2d | iid | 0.600000 | 0.800000 | 0.200000 | adapter_helps |", report)
        self.assertIn("| pdebench_burgers_1d | iid | ovha_full | 1.200000 | 0.700000 |", report)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _eval_row(family: str, split: str, model: str, relative_l2: float) -> dict[str, object]:
    return {
        "family": family,
        "dataset": family,
        "split": split,
        "model": model,
        "model_name": model,
        "relative_l2": relative_l2,
    }


def _diagnostic_row(
    family: str,
    split: str,
    model: str,
    adapter_norm: float,
    router_query_residual_norm: float,
) -> dict[str, object]:
    return {
        "family": family,
        "split": split,
        "model": model,
        "model_name": model,
        "primitive_entropy": 0.5,
        "router_context_prior_entropy": 0.3,
        "router_query_residual_norm": router_query_residual_norm,
        "adapter_norms": {"spectral": adapter_norm, "local": adapter_norm, "separable": adapter_norm},
        "adapter_stats": {"spectral": {"q_variance_scale": 0.01}},
    }


if __name__ == "__main__":
    unittest.main()
