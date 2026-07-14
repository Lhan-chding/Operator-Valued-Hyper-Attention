import unittest

from torch import nn

from ovha_rod.models.operators.decoder_training import (
    configure_decoder_operator_only_training,
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


if __name__ == "__main__":
    unittest.main()
