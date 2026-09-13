# -*- coding: utf-8 -*-
import _isolate  # noqa: F401

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "需要 PyTorch")
class NetworksTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        from app import networks

        self.networks = networks

    def test_mlp_classification_forward_shape(self):
        net = self.networks.build_model(
            "tabular_classification", "mlp",
            {"hidden_sizes": "16,8", "activation": "relu", "dropout": 0.0},
            num_features=8, num_classes=3,
        )
        out = net(self.torch.randn(4, 8))
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertGreater(self.networks.count_parameters(net), 0)

    def test_mlp_regression_forward_shape(self):
        net = self.networks.build_model(
            "tabular_regression", "mlp",
            {"hidden_sizes": "16", "dropout": 0.0},
            num_features=5, num_classes=2,
        )
        out = net(self.torch.randn(2, 5))
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_image_models_forward(self):
        for model in ("cnn", "resnet18"):
            with self.subTest(model=model):
                net = self.networks.build_model(
                    "image_classification", model,
                    {"pretrained": False, "freeze_backbone": False, "dropout": 0.0},
                    num_classes=4,
                )
                out = net(self.torch.randn(2, 3, 32, 32))
                self.assertEqual(tuple(out.shape), (2, 4))
                self.assertGreater(self.networks.count_parameters(net), 0)

    def test_text_models_forward(self):
        x = self.torch.randint(1, 50, (2, 16))
        for model in ("lstm", "gru", "textcnn", "transformer"):
            with self.subTest(model=model):
                net = self.networks.build_model(
                    "text_classification", model,
                    {"embedding_dim": 32, "hidden_dim": 16, "num_filters": 8,
                     "d_model": 32, "nhead": 4, "num_layers": 1, "dim_feedforward": 64,
                     "dropout": 0.0},
                    num_classes=4, num_tokens=60, max_seq_len=16,
                )
                out = net(x)
                self.assertEqual(tuple(out.shape), (2, 4))
                self.assertGreater(self.networks.count_parameters(net), 0)

    def test_unknown_model_and_task_raise(self):
        with self.assertRaises(ValueError):
            self.networks.build_model("tabular_classification", "nope")
        with self.assertRaises(ValueError):
            self.networks.build_model("video_classification", "cnn")


if __name__ == "__main__":
    unittest.main()
