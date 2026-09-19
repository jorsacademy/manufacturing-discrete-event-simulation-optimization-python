import numpy as np

from manufacturing_des_optimization import LineDesign
from neural_surrogate_optimization import SurrogateSearchConfig, design_features, expanded_candidate_designs, fit_surrogate, predict_surrogate


def test_expanded_design_space_and_features():
    cfg = SurrogateSearchConfig(buffer_min=1, buffer_max=3, technicians=(1, 2), initial_designs=2, epochs=2)
    designs = expanded_candidate_designs(cfg)
    assert len(designs) == 18
    x = design_features([LineDesign(1, 1, 1), LineDesign(3, 3, 2)], cfg)
    assert x.shape == (2, 3)
    assert np.all((0.0 <= x) & (x <= 1.0))


def test_surrogate_fits_tiny_smooth_sample():
    cfg = SurrogateSearchConfig(buffer_min=1, buffer_max=3, technicians=(1, 2), initial_designs=2, epochs=20, hidden_dim=16)
    designs = [LineDesign(1, 1, 1), LineDesign(2, 2, 1), LineDesign(3, 3, 2), LineDesign(1, 3, 2)]
    x = design_features(designs, cfg)
    y = np.asarray([1.0, 2.0, 4.0, 2.5], dtype=np.float32)
    model, mean, std = fit_surrogate(x, y, cfg)
    pred = predict_surrogate(model, x, mean, std)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
