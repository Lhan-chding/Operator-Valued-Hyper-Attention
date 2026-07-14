import unittest

from torch import nn

from ovha_rod.models.operators.decoder_training import (
    configure_decoder_operator_only_training,
    enforce_decoder_operator_only_training_mode,
)


class _TrainingSurface(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.seed_operator = nn.Linear(4, 4)
        self.decoder_operator = nn.Sequential(nn.Linear(4, 4))


class DecoderOperatorOnlyTrainingTests(unittest.TestCase):
    def test_freezes_every_parameter_except_the_decoder_bank(self):
        model = _TrainingSurface()

        trainable = configure_decoder_operator_only_training(model)

        expected = {
            "decoder_operator.0.weight",
            "decoder_operator.0.bias",
        }
        self.assertEqual(set(trainable), expected)
        for name, parameter in model.named_parameters():
            with self.subTest(name=name):
                self.assertEqual(parameter.requires_grad, name in expected)

    def test_requires_a_nonempty_decoder_operator(self):
        with self.assertRaisesRegex(ValueError, "decoder_operator"):
            configure_decoder_operator_only_training(nn.Linear(4, 4))

        model = _TrainingSurface()
        model.decoder_operator = nn.Identity()
        with self.assertRaisesRegex(ValueError, "trainable parameters"):
            configure_decoder_operator_only_training(model)

    def test_requires_a_torch_module(self):
        with self.assertRaisesRegex(TypeError, "torch module"):
            configure_decoder_operator_only_training(object())

    def test_keeps_frozen_base_in_eval_and_decoder_bank_in_train(self):
        model = _TrainingSurface()
        model.train()

        enforce_decoder_operator_only_training_mode(model)

        self.assertTrue(model.training)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.seed_operator.training)
        self.assertTrue(model.decoder_operator.training)
        self.assertTrue(model.decoder_operator[0].training)

        with self.assertRaisesRegex(TypeError, "torch module"):
            enforce_decoder_operator_only_training_mode(object())
        model.decoder_operator = None
        with self.assertRaisesRegex(ValueError, "initialized module"):
            enforce_decoder_operator_only_training_mode(model)


if __name__ == "__main__":
    unittest.main()
