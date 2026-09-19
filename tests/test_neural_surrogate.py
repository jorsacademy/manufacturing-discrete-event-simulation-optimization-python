import importlib.util
import unittest

import numpy as np

from manufacturing_des_optimization import LineDesign

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "install requirements-surrogate.txt to run neural surrogate tests")
class NeuralSurrogateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from neural_surrogate_optimization import (
            SurrogateSearchConfig,
            design_features,
            expanded_candidate_designs,
            fit_surrogate,
            predict_surrogate,
        )
        cls.Config = SurrogateSearchConfig
        cls.design_features = staticmethod(design_features)
        cls.expanded_candidate_designs = staticmethod(expanded_candidate_designs)
        cls.fit_surrogate = staticmethod(fit_surrogate)
        cls.predict_surrogate = staticmethod(predict_surrogate)

    def test_expanded_design_space_and_features(self):
        cfg = self.Config(buffer_min=1, buffer_max=3, technicians=(1, 2), initial_designs=2, epochs=2)
        designs = self.expanded_candidate_designs(cfg)
        self.assertEqual(len(designs), 18)
        x = self.design_features([LineDesign(1, 1, 1), LineDesign(3, 3, 2)], cfg)
        self.assertEqual(x.shape, (2, 3))
        self.assertTrue(np.all((0.0 <= x) & (x <= 1.0)))

    def test_surrogate_fits_tiny_smooth_sample(self):
        cfg = self.Config(buffer_min=1, buffer_max=3, technicians=(1, 2), initial_designs=2, epochs=20, hidden_dim=16)
        designs = [LineDesign(1, 1, 1), LineDesign(2, 2, 1), LineDesign(3, 3, 2), LineDesign(1, 3, 2)]
        x = self.design_features(designs, cfg)
        y = np.asarray([1.0, 2.0, 4.0, 2.5], dtype=np.float32)
        model, mean, std = self.fit_surrogate(x, y, cfg)
        pred = self.predict_surrogate(model, x, mean, std)
        self.assertEqual(pred.shape, (4,))
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
