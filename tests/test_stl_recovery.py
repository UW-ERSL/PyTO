"""STL recovery CNN must work while torch's default dtype is float64 (set by pyto.autodiff.material_model)."""
import torch


def test_cnn_runs_with_float64_default_dtype():
    import pyto.autodiff.material_model  # noqa: F401  (sets the float64 default, as in the GUI)
    from pyto.io.topopt_stl_recovery import CNN3D
    assert torch.get_default_dtype() == torch.float64
    model = CNN3D()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    out = model(torch.rand(1, 1, 6, 5, 4, dtype=torch.float64))      # any input dtype is accepted
    assert out.shape == (1, 1, 6, 5, 4) and out.dtype == torch.float32
