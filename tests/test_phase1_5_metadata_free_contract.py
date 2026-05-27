import unittest

from moat_ovha_torch.data.episodes import (
    EpisodeHiddenInfo,
    MetaOperatorBatch,
    assert_no_metadata_leakage,
    batch_public_tensor_names,
    hash_model_inputs,
)


class Phase15MetadataFreeContractTests(unittest.TestCase):
    def test_public_batch_fields_exclude_hidden_metadata(self):
        public_names = set(batch_public_tensor_names())
        forbidden = {"family", "operator_id", "gain", "hint", "latent", "mixture", "oracle"}

        self.assertTrue({"context_u", "context_q", "context_y", "target_u", "target_q", "support_grid"} <= public_names)
        self.assertTrue(public_names.isdisjoint(forbidden))

    def test_hidden_metadata_does_not_change_model_input_hash(self):
        batch = MetaOperatorBatch(
            context_u=[[[[0.0], [1.0]]]],
            context_q=[[[[0.0], [0.5]]]],
            context_y=[[[[0.1], [0.2]]]],
            target_u=[[[0.0], [1.0]]],
            target_q=[[[0.25], [0.75]]],
            target_y=[[[0.3], [0.4]]],
            support_grid=[[[0.0], [1.0]]],
            context_mask=None,
            target_mask=None,
        )
        hidden_a = EpisodeHiddenInfo(
            family="spectral_family",
            latent_params={"gain": 1.5},
            mixture_weights={"spectral": 1.0},
            oracle_hints={"frequency": 2.0},
        )
        hidden_b = EpisodeHiddenInfo(
            family="local_green_family",
            latent_params={"gain": 9.0},
            mixture_weights={"local": 1.0},
            oracle_hints={"lengthscale": 0.1},
        )

        self.assertNotEqual(hidden_a, hidden_b)
        self.assertEqual(hash_model_inputs(batch), hash_model_inputs(batch))

    def test_leakage_guard_rejects_forbidden_keys(self):
        with self.assertRaises(ValueError):
            assert_no_metadata_leakage({"context_u": object(), "family": "spectral_family"})
        with self.assertRaises(ValueError):
            assert_no_metadata_leakage({"target_q": object(), "oracle_hints": {"gain": 1.0}})


if __name__ == "__main__":
    unittest.main()
