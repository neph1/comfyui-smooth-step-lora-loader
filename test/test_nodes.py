import pytest
import torch
from nodes import _Smooth_Step_Lora_Loader_Base

class TestSmoothStepLoraLoaderBase:

    @pytest.fixture
    def loader(self):
        return _Smooth_Step_Lora_Loader_Base()

    def test_load_lora_no_strength(self, loader):
        model = torch.nn.Linear(10, 10)
        clip = torch.nn.Linear(10, 10)
        result = loader.load_lora(model, clip, "test_lora", 0, 0, 1.0)
        assert result == (model, clip)

    def test_smooth_step_function(self, loader):
        x = torch.tensor([0.0, 0.5, 1.0])
        expected = torch.tensor([0.0, 0.5, 1.0])
        result = loader.smooth_step_function(x)
        assert torch.allclose(result, expected)

    def test_smooth_step_lora(self, loader):
        sd = {
            "lora_up": torch.tensor([0.1, 0.2, 0.3]),
            "lora_down": torch.tensor([0.4, 0.5, 0.6]),
            "lora_mid": torch.tensor([0.7, 0.8, 0.9])
        }
        factor = 0.5
        result = loader.smooth_step_lora(sd, factor)
        assert "lora_up" in result
        assert "lora_down" in result
        assert "lora_mid" in result
        assert torch.allclose(result["lora_up"], torch.tensor([0.1, 0.2, 0.3]))
        assert torch.allclose(result["lora_down"], torch.tensor([0.4, 0.5, 0.6]))
        assert torch.allclose(result["lora_mid"], torch.tensor([0.7, 0.8, 0.9]))